import torch
import torch.nn.functional as F
from utils.util import forward_process_length, shift_logits, forward_process, batched_overlap_input, batched_dual_bd_noise_transition
from einops import rearrange

def compute_loss_by_config(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        config
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

# ==============================================================================
# Dual Block Diffusion Mask Utilities
# ==============================================================================

def _compute_dual_bd_params(seq_len: int, block_size: int):
    num_blocks = seq_len // block_size
    x_t_len = (seq_len // block_size - 1) * 2 * block_size
    x_0_len = seq_len
    return num_blocks, x_t_len, x_0_len

def _map_token_to_block(num_blocks, x_t_len, x_0_len, block_size, device):
    x_t_block_map = torch.zeros(x_t_len, dtype=torch.int32, device=device)
    x_0_block_map = torch.zeros(x_0_len, dtype=torch.int32, device=device)
    x_t_block_group_map = torch.zeros(x_t_len, dtype=torch.int32, device=device)
    x_0_block_group_map = torch.zeros(x_0_len, dtype=torch.int32, device=device)
    
    # Map x_t tokens to blocks
    for i in range(num_blocks - 1):
        left_block_id = i
        right_block_id = i + 1
        
        block_start_idx = left_block_id * 2
        left_block_start_idx = block_start_idx * block_size
        left_block_end_idx = (block_start_idx + 1) * block_size
        right_block_start_idx = left_block_end_idx
        right_block_end_idx = (block_start_idx + 2) * block_size
        
        x_t_block_map[left_block_start_idx:left_block_end_idx] = left_block_id
        x_t_block_map[right_block_start_idx:right_block_end_idx] = right_block_id
        x_t_block_group_map[left_block_start_idx:right_block_end_idx] = left_block_id
    
    # Map x_0 tokens to blocks
    for i in range(num_blocks):
        block_id = i
        block_start_idx = block_id * block_size
        block_end_idx = (block_id + 1) * block_size
        x_0_block_map[block_start_idx:block_end_idx] = block_id
        x_0_block_group_map[block_start_idx:block_end_idx] = block_id
    
    return (
        torch.cat([x_t_block_map, x_0_block_map], dim=0),
        torch.cat([x_t_block_group_map, x_0_block_group_map], dim=0)
    )

def dual_bd_attn_mask_student(
    batch_size: int,
    num_kv_heads: int,
    q_ids: torch.Tensor,
    kv_ids: torch.Tensor,
    block_size: int,
    seq_len: int
) -> torch.Tensor:
    num_blocks, x_t_len, x_0_len = _compute_dual_bd_params(seq_len, block_size)
    
    x_0_flag_q = (q_ids >= x_t_len)
    x_0_flag_kv = (kv_ids >= x_t_len)
    
    device = q_ids.device
    block_mapping, block_group_mapping = _map_token_to_block(
        num_blocks, x_t_len, x_0_len, block_size, device
    )
    
    block_mapping_q, block_group_mapping_q = [
        rearrange(mapping, "s -> s 1") for mapping in [block_mapping, block_group_mapping]
    ]
    block_mapping_kv, block_group_mapping_kv = [
        rearrange(mapping, "s -> 1 s") for mapping in [block_mapping, block_group_mapping]
    ]
    
    block_causal_template = (block_mapping_q >= block_mapping_kv)
    block_group_diagonal = (block_group_mapping_q == block_group_mapping_kv) & (~x_0_flag_q & ~x_0_flag_kv)
    
    offset_block_causal = (block_group_mapping_q > block_group_mapping_kv) & (~x_0_flag_q & x_0_flag_kv)
    group_diagonal_inner_causal = block_causal_template & block_group_diagonal
    block_causal = block_causal_template & (x_0_flag_q & x_0_flag_kv)
    
    return group_diagonal_inner_causal | offset_block_causal | block_causal

def dual_bd_attn_mask_teacher(
    batch_size: int,
    num_kv_heads: int,
    q_ids: torch.Tensor,
    kv_ids: torch.Tensor,
    block_size: int,
    seq_len: int
) -> torch.Tensor:
    num_blocks, x_t_len, x_0_len = _compute_dual_bd_params(seq_len, block_size)
    
    x_0_flag_q = (q_ids >= x_t_len)
    x_0_flag_kv = (kv_ids >= x_t_len)
    
    device = q_ids.device
    block_mapping, block_group_mapping = _map_token_to_block(
        num_blocks, x_t_len, x_0_len, block_size, device
    )
    
    block_mapping_q, block_group_mapping_q = [
        rearrange(mapping, "s -> s 1") for mapping in [block_mapping, block_group_mapping]
    ]
    block_mapping_kv, block_group_mapping_kv = [
        rearrange(mapping, "s -> 1 s") for mapping in [block_mapping, block_group_mapping]
    ]
    
    block_diagonal = (
        (block_group_mapping_q == block_group_mapping_kv) 
        & (~x_0_flag_q & ~x_0_flag_kv)
    )
    offset_full = (
        (block_mapping_kv < block_group_mapping_q) | (block_mapping_kv > (block_group_mapping_q + 1))
    ) & (x_0_flag_kv & ~x_0_flag_q)
    full = torch.full_like(block_diagonal, True) & (x_0_flag_kv & x_0_flag_q)
    
    return block_diagonal | offset_full | full

def dual_bd_attn_mask_generator(
    mask_flag_fn,
    dtype,
    device
) -> torch.Tensor:
    dual_bd_attn_mask_flag = mask_flag_fn().unsqueeze(0).unsqueeze(0)
    dual_bd_attn_mask_prototype = torch.zeros_like(
        dual_bd_attn_mask_flag,
        dtype=dtype,
        device=device
    )
    dual_bd_attn_mask_prototype.masked_fill_(
        dual_bd_attn_mask_flag.logical_not(), float("-inf")
    )
    return dual_bd_attn_mask_prototype

# ==============================================================================
# Compute Loss
# ==============================================================================

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    device = input_ids.device
    
    # 1. Overlap Input
    overlapped_input, overlapped_pos, maskable_mask = batched_overlap_input(input_ids, block_size, question_length)
    
    # 2. Noise Transition
    noise_range = [[0.2, 0.5], [0.5, 0.8]]
    noisy_input, p_mask = batched_dual_bd_noise_transition(
        overlapped_input, 
        maskable_mask, 
        mask_id, 
        block_size, 
        noise_range
    )
    
    # 3. Full Input Construction
    # Padding input_ids to match block structure if needed for clean part
    if L % block_size != 0:
        pad_len = block_size - (L % block_size)
        clean_input_ids = torch.cat([input_ids, torch.zeros((B, pad_len), dtype=input_ids.dtype, device=device)], dim=1)
        L = clean_input_ids.shape[1] # Update L to padded length
    else:
        clean_input_ids = input_ids
    
    full_input_ids = torch.cat([noisy_input, clean_input_ids], dim=1)
    
    clean_pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    full_pos_ids = torch.cat([overlapped_pos, clean_pos_ids], dim=1)
    
    # 4. Attention Mask
    full_seq_len = full_input_ids.shape[1]
    from functools import partial
    mask_flag_fn = partial(
        dual_bd_attn_mask_student,
        batch_size=B,
        num_kv_heads=1,
        q_ids=torch.arange(full_seq_len, device=device)[:, None],
        kv_ids=torch.arange(full_seq_len, device=device)[None, :],
        block_size=block_size,
        seq_len=L
    )
    
    attention_mask = dual_bd_attn_mask_generator(
        mask_flag_fn, 
        dtype=torch.float16, # Assuming float16 for training
        device=device
    )
    
    # 5. Forward Pass
    logits = denoiser(full_input_ids, attention_mask=attention_mask, position_ids=full_pos_ids).logits
    
    # 6. Extract Noisy Part Logits
    noisy_len = noisy_input.shape[1]
    noisy_logits = logits[:, :noisy_len, :]
    
    # 7. Shift Logits (Maintain original logic)
    logits = shift_logits(noisy_logits)
    
    # 8. Masked Indices
    masked_indices = (noisy_input == mask_id)
    
    # 9. Target (Clean Overlapped Input)
    target = overlapped_input
    
    # 10. Loss Calculation
    if self_align:
        mask_flag_fn_teacher = partial(
            dual_bd_attn_mask_teacher,
            batch_size=B,
            num_kv_heads=1,
            q_ids=torch.arange(full_seq_len, device=device)[:, None],
            kv_ids=torch.arange(full_seq_len, device=device)[None, :],
            block_size=block_size,
            seq_len=L
        )
        teacher_attn_mask = dual_bd_attn_mask_generator(
            mask_flag_fn_teacher,
            dtype=torch.float16,
            device=device
        )
        
        with torch.no_grad():
            with denoiser.disable_adapter():
                ref_logits = denoiser(full_input_ids, attention_mask=teacher_attn_mask, position_ids=full_pos_ids).logits
                ref_noisy_logits = ref_logits[:, :noisy_len, :]
                ref_logits = shift_logits(ref_noisy_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
    else:
        token_loss_2 = F.cross_entropy(logits[masked_indices], target[masked_indices], reduction='none') / p_mask[masked_indices]
    
    losses = {
        'loss': token_loss_2.mean(),
    }

    return losses 

def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)
    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 

def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    mask_id=126336
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                # ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 


def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
    B,seq_len= input_ids.shape
    # 初始化为全 -inf
    attn_mask = torch.full((B,1,seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)
    # 1. Prompt部分：每个token可以注意整个prompt
    for i in range(B):
        attn_mask[i,:,:,:prompt_length[i]] = 0.0  # 允许所有 token 看 prompt

        # 2. 块划分：从 prompt_length 开始划分 block
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            # print(block_start,block_size,seq_len)
            block_end = min(block_start + block_size, seq_len)

            # 块内全注意
            attn_mask[i,:,block_start:block_end, block_start:block_end] = 0.0

            # 块之间因果注意（只能看前面块）
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # 当前块可以看前面块
                attn_mask[i,:,block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask  # [seq_len, seq_len], float, 0.0 for allowed, -inf for disallowed
if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)