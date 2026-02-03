import torch
from torch.distributions import Uniform

def forward_process_block_fixed_p(x, mask_id, p_mask):
    B, L = x.shape
    if isinstance(p_mask, float):
        p_mask = torch.full((B, 1), p_mask, device=x.device)
    elif p_mask.ndim == 1:
        p_mask = p_mask[:, None]
    rand = torch.rand((B, L), device=x.device)
    mask = rand < p_mask
    x_masked = torch.where(mask, mask_id, x)
    return x_masked, mask

import torch

def generate_monotonic_pmasks(batch_size, max_blocks, device):
    """
    生成 shape (B, max_blocks) 的单调非降随机序列，每行第一个元素在[0,1]随机，后续不小于前一个
    """
    # 第一个block p_mask随机
    p0 = torch.rand(batch_size, 1, device=device)/2+0.2
    # print(p0)
    # 后续blocks生成增量 [0, 1]，加起来保证不超过1（之后用 clamp）
    increments = torch.rand(batch_size, max_blocks - 1, device=device) * (0.7 - p0)/ (max_blocks - 1)
    # print(increments)
    # 逐元素累加，保证非降
    cum_increments = torch.cumsum(increments, dim=1)
    # print(cum_increments)
    # 总 p_mask = p0 + 累积增量，保证不超过1
    p_masks = torch.cat([p0, p0 + cum_increments], dim=1)
    p_masks = torch.clamp(p_masks, max=1.0)
    # print(p_masks)
    return p_masks  # (B, max_blocks)


def forward_process_length(input_ids, mask_id, block_size, prompt_lengths, eos_id=None):
    """
    修改说明：
    1. 计算 last_non_eos_indices (即有效内容结束后的第一个位置，也就是 EOS 的位置)。
    2. 对中间内容进行单调递增概率 Mask (原有逻辑)。
    3. [新增] 检查 eos_id 是否存在，如果存在，强制将第一个 EOS 位置设为 mask_id，并将 p_mask 设为 1.0。
    """
    B, L = input_ids.shape
    device = input_ids.device
    noisy_batch = input_ids.clone()
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    p_mask_tensor = torch.zeros((B, L), device=device)

    # 1. 确定每个样本的有效内容结束位置 (last_non_eos_indices 指向第一个 EOS)
    if eos_id is not None:
        non_eos_mask = (input_ids != eos_id)
        # 确保 prompt 区域被视为非 EOS (避免 prompt 只有 EOS 的极端情况，虽不常见)
        for i in range(B):
            non_eos_mask[i, :prompt_lengths[i]] = True
        
        last_non_eos_indices = []
        for i in range(B):
            row_non_eos = torch.where(non_eos_mask[i])[0]
            if len(row_non_eos) > 0:
                last_non_eos_indices.append(row_non_eos[-1].item() + 1)
            else:
                last_non_eos_indices.append(prompt_lengths[i].item())
        last_non_eos_indices = torch.tensor(last_non_eos_indices, device=device)
    else:
        last_non_eos_indices = torch.full((B,), L, device=device)

    # 2. 计算需要加 mask 的有效区域长度和 block 数
    active_lens = torch.clamp(last_non_eos_indices - prompt_lengths, min=0)
    full_blocks = active_lens // block_size
    remainders = active_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()

    max_blocks = total_blocks.max().item()

    # 3. 生成 mask 比率
    if max_blocks > 0:
        p_masks = generate_monotonic_pmasks(B, max_blocks, device)  # (B, max_blocks)
    else:
        p_masks = None

    # 4. 应用 Mask 逻辑
    for i in range(B):
        prompt_len = prompt_lengths[i].item()
        
        # --- Part A: 正常的 Block 随机 Mask (原有逻辑) ---
        if p_masks is not None:
            num_blocks = total_blocks[i].item()
            for block_idx in range(num_blocks):
                start = prompt_len + block_idx * block_size
                end = min(start + block_size, last_non_eos_indices[i].item())
                
                if start >= end:
                    continue

                p_block = p_masks[i, block_idx].item()

                block_data = noisy_batch[i, start:end].unsqueeze(0)
                masked_block, mask = forward_process_block_fixed_p(block_data, mask_id, p_block)

                noisy_batch[i, start:end] = masked_block.squeeze(0)
                masked_indices[i, start:end] = mask.squeeze(0)
                p_mask_tensor[i, start:end] = p_block

        # --- Part B: [新增] 强制 Mask 第一个 EOS 位置 ---
        if eos_id is not None:
            # last_non_eos_indices[i] 刚好指向有效内容后的第一个位置 (即 EOS)
            eos_pos = last_non_eos_indices[i].item()
            
            # 确保位置在序列范围内 (防止序列填满没有EOS的情况)
            if eos_pos < L:
                noisy_batch[i, eos_pos] = mask_id       # 填入 [MASK] token
                masked_indices[i, eos_pos] = True       # 标记被 mask
                p_mask_tensor[i, eos_pos] = 1.0         # 该位置 mask 概率记为 1.0

    return noisy_batch, masked_indices, p_mask_tensor

# def forward_process_length(input_ids, mask_id, block_size, prompt_lengths, p_min=0.2, p_max=0.9):
#     """
#     返回每个 token 的实际 mask 概率 tensor（非prompt区域），其余为0。
#     """
#     B, L = input_ids.shape
#     device = input_ids.device
#     noisy_batch = input_ids.clone()
#     masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
#     p_mask_tensor = torch.zeros((B, L), device=device)  # 最终返回值

#     for i in range(B):
#         prompt_len = prompt_lengths[i].item()
#         non_prompt_len = L - prompt_len
#         full_blocks = non_prompt_len // block_size
#         remainder = non_prompt_len % block_size
#         total_blocks = full_blocks + (1 if remainder > 0 else 0)

#         for block_idx in range(total_blocks):
#             start = prompt_len + block_idx * block_size
#             end = min(start + block_size, L)

#             # block的 mask 概率（线性递增）
#             if total_blocks > 1:
#                 p_block = p_min + (p_max - p_min) * (block_idx / (total_blocks - 1))
#             else:
#                 p_block = p_max

#             block = noisy_batch[i, start:end].unsqueeze(0)
#             masked_block, mask = forward_process_block_fixed_p(block, mask_id, p_block)
#             noisy_batch[i, start:end] = masked_block.squeeze(0)
#             masked_indices[i, start:end] = mask.squeeze(0)

#             # 记录 p_mask 到 tensor 中
#             p_mask_tensor[i, start:end] = p_block

#     return noisy_batch, masked_indices, p_mask_tensor
def forward_process(input_ids,mask_id ,t_max=1.0, eps=1e-4):
    B, L = input_ids.shape
    # t = torch.rand(B, device=input_ids.device)
    dist = Uniform(0., t_max)
    t = dist.sample((B,)).to(input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, L)
    masked_indices = torch.rand((B, L), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    return noisy_batch, masked_indices, p_mask
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def shift_logits(logits):
    shifted_logits = torch.zeros_like(logits)
    shifted_logits[:, 1:, :] = logits[:, :-1, :]
    shifted_logits[:, 0, :] = 1.0

    return shifted_logits

def batched_overlap_input(input_ids, block_size, prompt_lengths):
    """
    Overlaps input ids for dual block diffusion.
    input_ids: (B, L)
    Returns:
        overlapped_input: (B, New_L)
        overlapped_pos: (B, New_L)
        overlapped_maskable: (B, New_L)
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    # Pad to multiple of block_size if needed
    remainder = L % block_size
    if remainder > 0:
        pad_len = block_size - remainder
        input_ids = torch.cat([input_ids, torch.zeros((B, pad_len), dtype=input_ids.dtype, device=device)], dim=1)
        L = input_ids.shape[1]
    
    num_blocks = L // block_size
    
    # We want pairs (0,1), (1,2), ...
    # Indices of blocks
    indices = []
    for i in range(num_blocks - 1):
        indices.extend([i, i+1])
    
    indices = torch.tensor(indices, device=device) # Shape (M,) where M = 2*(num_blocks-1)
    
    reshaped_input = input_ids.view(B, num_blocks, block_size)
    overlapped_input = reshaped_input[:, indices, :].reshape(B, -1)
    
    # Position IDs
    # Global position IDs: 0, 1, ..., L-1
    pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    reshaped_pos = pos_ids.view(B, num_blocks, block_size)
    overlapped_pos = reshaped_pos[:, indices, :].reshape(B, -1)
    
    # Maskable
    # True if pos >= prompt_len
    # prompt_lengths: (B,)
    maskable = pos_ids >= prompt_lengths.unsqueeze(1)
    reshaped_maskable = maskable.view(B, num_blocks, block_size)
    overlapped_maskable = reshaped_maskable[:, indices, :].reshape(B, -1)
    
    return overlapped_input, overlapped_pos, overlapped_maskable

def _batched_fixed_ratio_mask(block, block_maskable, ratio, mask_token_id):
    # block: (B, K)
    # block_maskable: (B, K)
    # ratio: float or (B,) tensor
    B, K = block.shape
    device = block.device
    
    rand_vals = torch.rand(B, K, device=device)
    # We want to select top k items where maskable is True
    # Set non-maskable to -1 (so they are not selected)
    # But rand is [0, 1]. So valid are [0, 1], invalid are -1.
    rand_vals = torch.where(block_maskable, rand_vals, torch.tensor(-1.0, device=device))
    
    # Determine how many to mask for each row
    num_valid = block_maskable.sum(dim=1) # (B,)
    if isinstance(ratio, float):
        num_to_mask = (num_valid.float() * ratio).long()
    else:
        num_to_mask = (num_valid.float() * ratio).long() # (B,)
    
    # We want to select top `num_to_mask` indices
    sorted_vals, sorted_indices = torch.sort(rand_vals, descending=True, dim=1)
    
    # Create a mask of selected indices
    # rank_matrix: 0, 1, 2, ... K-1
    rank_matrix = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
    
    if isinstance(num_to_mask, torch.Tensor) and num_to_mask.ndim == 1:
        limit = num_to_mask.unsqueeze(1)
    else:
        limit = num_to_mask
        
    selected_mask = rank_matrix < limit
    
    # Gather original indices
    mask_decision = torch.zeros_like(block, dtype=torch.bool)
    mask_decision.scatter_(1, sorted_indices, selected_mask)
    
    # Ensure invalid ones are not masked
    mask_decision = mask_decision & block_maskable
    
    masked_block = torch.where(mask_decision, mask_token_id, block)
    return masked_block

def batched_dual_bd_noise_transition(
    x_0, # (B, L_new)
    maskable_mask, # (B, L_new)
    mask_token_id,
    block_size,
    noise_range # [[low_min, low_max], [high_min, high_max]]
):
    B, L = x_0.shape
    device = x_0.device
    
    # Split into blocks. L is divisible by block_size because of construction.
    num_blocks = L // block_size
    
    x_reshaped = x_0.view(B, num_blocks, block_size)
    maskable_reshaped = maskable_mask.view(B, num_blocks, block_size)
    
    out_blocks_list = []
    p_mask_list = []
    
    left_range, right_range = noise_range
    
    # Iterate in pairs
    for b_idx in range(0, num_blocks, 2):
        if b_idx + 1 >= num_blocks:
            break
            
        block_left = x_reshaped[:, b_idx, :]
        maskable_left = maskable_reshaped[:, b_idx, :]
        
        block_right = x_reshaped[:, b_idx+1, :]
        maskable_right = maskable_reshaped[:, b_idx+1, :]
        
        # Sample p_left, p_right for each batch element
        p_left = torch.rand(B, device=device) * (left_range[1] - left_range[0]) + left_range[0]
        p_right = torch.rand(B, device=device) * (right_range[1] - right_range[0]) + right_range[0]
        
        masked_left = _batched_fixed_ratio_mask(block_left, maskable_left, p_left, mask_token_id)
        masked_right = _batched_fixed_ratio_mask(block_right, maskable_right, p_right, mask_token_id)
        
        out_blocks_list.append(masked_left)
        out_blocks_list.append(masked_right)
        
        # Create p_mask blocks (B, block_size)
        p_mask_left = p_left.unsqueeze(1).expand(B, block_size)
        p_mask_right = p_right.unsqueeze(1).expand(B, block_size)
        p_mask_list.append(p_mask_left)
        p_mask_list.append(p_mask_right)
        
    if len(out_blocks_list) > 0:
        x_masked = torch.stack(out_blocks_list, dim=1).view(B, -1)
        p_mask = torch.stack(p_mask_list, dim=1).view(B, -1)
    else:
        x_masked = x_0
        p_mask = torch.zeros_like(x_0, dtype=torch.float32)
        
    return x_masked, p_mask

if __name__ == '__main__':
    input_ids= torch.tensor([[1,5,4,3,25,6,7,9,5,8,7,6],[1,3,8,9,7,34,6,9,5,8,7,6]])
    mask_id=0
    block_size=3
    prompt_length=torch.tensor([2,1])
    noisy_batch, masked_indices,p_mask = forward_process_length(input_ids, mask_id, block_size, prompt_length)
    print("noisy_batch:", noisy_batch)
    print("masked_indices:", masked_indices)
    print("p_mask:", p_mask)
