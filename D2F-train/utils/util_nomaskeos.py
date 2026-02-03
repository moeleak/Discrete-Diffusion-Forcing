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
    修改后：仅对 prompt 之后、且包含非 EOS 内容的 block 加 mask。
    """
    B, L = input_ids.shape
    device = input_ids.device
    noisy_batch = input_ids.clone()
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    p_mask_tensor = torch.zeros((B, L), device=device)

    # 1. 确定每个样本的有效内容结束位置 (last_non_eos_idx)
    # 如果没有提供 eos_id，则默认处理到序列末尾
    if eos_id is not None:
        # 找到所有非 eos 的位置
        non_eos_mask = (input_ids != eos_id)
        # 为处理方便，将 prompt 区域也标为 True，确保有效长度至少包含 prompt
        for i in range(B):
            non_eos_mask[i, :prompt_lengths[i]] = True
        
        # 找到每个 batch 中最后一个 True 的索引
        # 使用 flip 找第一个非零元素比较高效
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
    # 有效区域 = last_non_eos_indices - prompt_lengths
    active_lens = torch.clamp(last_non_eos_indices - prompt_lengths, min=0)
    full_blocks = active_lens // block_size
    remainders = active_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()

    max_blocks = total_blocks.max().item()

    # 3. 生成 mask 比率 (仅针对有效 block 数)
    if max_blocks > 0:
        p_masks = generate_monotonic_pmasks(B, max_blocks, device)  # (B, max_blocks)
    else:
        return noisy_batch, masked_indices, p_mask_tensor

    # 4. 应用 Mask 逻辑
    for i in range(B):
        prompt_len = prompt_lengths[i].item()
        num_blocks = total_blocks[i].item() # 该样本实际需要 mask 的块数
        
        for block_idx in range(num_blocks):
            start = prompt_len + block_idx * block_size
            # 结束位置不能超过该样本的有效内容边界
            end = min(start + block_size, last_non_eos_indices[i].item())
            
            if start >= end:
                continue

            p_block = p_masks[i, block_idx].item()

            block_data = noisy_batch[i, start:end].unsqueeze(0)
            masked_block, mask = forward_process_block_fixed_p(block_data, mask_id, p_block)

            noisy_batch[i, start:end] = masked_block.squeeze(0)
            masked_indices[i, start:end] = mask.squeeze(0)
            p_mask_tensor[i, start:end] = p_block

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
if __name__ == '__main__':
    input_ids= torch.tensor([[1,5,4,3,25,6,7,9,5,8,7,6],[1,3,8,9,7,34,6,9,5,8,7,6]])
    mask_id=0
    block_size=3
    prompt_length=torch.tensor([2,1])
    noisy_batch, masked_indices,p_mask = forward_process_length(input_ids, mask_id, block_size, prompt_length)
    print("noisy_batch:", noisy_batch)
    print("masked_indices:", masked_indices)
    print("p_mask:", p_mask)
