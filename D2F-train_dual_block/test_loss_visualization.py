import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from contextlib import contextmanager

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.loss import compute_loss, dual_bd_attn_mask_student, dual_bd_attn_mask_teacher, dual_bd_attn_mask_generator
from utils.util import batched_overlap_input, batched_dual_bd_noise_transition

# Mock Denoiser
class MockDenoiser(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Default

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.head(x)
        return type('Output', (), {'logits': logits})()

    @contextmanager
    def disable_adapter(self):
        yield

# Helper to print sequence data nicely
def print_sequence_data(tensor, name, block_size=None):
    """Prints sequence data to console, handling batch dimension."""
    data = tensor.detach().cpu().numpy()
    print(f"\n[{name}] Shape: {data.shape}")
    
    if data.ndim == 1:
        data = data[None, :]
        
    B, L = data.shape
    for b in range(min(B, 3)): # Print max 3 batches
        row_str = []
        for i, val in enumerate(data[b]):
            # Add visual separator for blocks if block_size is provided
            if block_size and i > 0 and i % block_size == 0:
                row_str.append("|")
            
            if isinstance(val, (int, np.integer)):
                row_str.append(f"{val:3d}")
            else:
                row_str.append(f"{val:.2f}")
        
        print(f"  Batch {b}: " + " ".join(row_str))
    
    if B > 3:
        print(f"  ... (showing 3/{B})")

def visualize_mask(mask, title, save_path=None):
    """Visualizes the attention mask."""
    # Mask is likely (B, 1, Q, K) or (1, 1, Q, K)
    if mask.ndim == 4:
        mask = mask[0, 0] # Take first batch, first head
    
    if mask.dtype == torch.bool:
        mask_np = mask.detach().cpu().float().numpy()
    else:
        # 0.0 -> 1 (Allowed), -inf -> 0 (Blocked)
        mask_np = mask.detach().cpu().float().numpy()
        mask_np = (mask_np == 0.0).astype(float)

    plt.figure(figsize=(10, 8))
    plt.imshow(mask_np, cmap='Blues', origin='upper', interpolation='nearest')
    plt.title(title)
    plt.xlabel('Key Position (Context)')
    plt.ylabel('Query Position (Target)')
    plt.colorbar(label='Allowed (1) / Blocked (0)')
    
    # Grid lines
    if mask_np.shape[0] <= 64:
        # Pixel-perfect grid
        plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.gca().set_xticks(np.arange(-0.5, mask_np.shape[1], 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticks(np.arange(-0.5, mask_np.shape[0], 1))
        plt.gca().set_yticklabels([])
        plt.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved mask visualization: {save_path}")
    plt.close()

def main():
    output_dir = 'test_viz_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Outputs will be saved to: {os.path.abspath(output_dir)}")

    # Parameters
    vocab_size = 100
    hidden_size = 16
    B = 1
    block_size = 4
    # Example length
    L = 16 
    mask_id = 99
    eos_id = 98
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inputs: Random integers
    input_ids = torch.randint(1, vocab_size - 10, (B, L)).to(device)
    prompt_length = torch.tensor([4], dtype=torch.long).to(device)
    
    # Model
    denoiser = MockDenoiser(vocab_size, hidden_size).to(device)
    
    print("="*60)
    print("STEP 1: Data Preparation Check")
    print("="*60)
    
    print_sequence_data(input_ids, "Original Input IDs", block_size)
    print_sequence_data(prompt_length, "Prompt Lengths")

    # 1. Overlap Input
    overlapped_input, overlapped_pos, maskable_mask = batched_overlap_input(input_ids, block_size, prompt_length)
    
    print_sequence_data(overlapped_input, "Overlapped Input (x_t structure)", block_size)
    print_sequence_data(overlapped_pos, "Overlapped Position IDs", block_size)
    print_sequence_data(maskable_mask.int(), "Maskable Mask (1=Response)", block_size)
    
    # 2. Noise Transition
    noise_range = [[0.1, 0.3], [0.5, 0.8]]
    noisy_input, p_mask = batched_dual_bd_noise_transition(
        overlapped_input, 
        maskable_mask, 
        mask_id, 
        block_size, 
        noise_range
    )
    
    print_sequence_data(noisy_input, f"Noisy Input (Mask ID={mask_id})", block_size)
    print_sequence_data(p_mask, "P Mask (Noise Levels)", block_size)
    
    # 3. Full Input Construction
    # Padding if needed
    if L % block_size != 0:
        pad_len = block_size - (L % block_size)
        clean_input_ids = torch.cat([input_ids, torch.zeros((B, pad_len), dtype=input_ids.dtype, device=device)], dim=1)
        L_padded = clean_input_ids.shape[1]
    else:
        clean_input_ids = input_ids
        L_padded = L
        
    full_input_ids = torch.cat([noisy_input, clean_input_ids], dim=1)
    print_sequence_data(full_input_ids, "Full Model Input [Noisy || Clean]", block_size)
    
    print("\n" + "="*60)
    print("STEP 2: Attention Mask Visualization")
    print("="*60)
    
    full_seq_len = full_input_ids.shape[1]
    
    # Student Mask
    from functools import partial
    mask_flag_fn_student = partial(
        dual_bd_attn_mask_student,
        batch_size=B,
        num_kv_heads=1,
        q_ids=torch.arange(full_seq_len, device=device)[:, None],
        kv_ids=torch.arange(full_seq_len, device=device)[None, :],
        block_size=block_size,
        seq_len=L_padded
    )
    
    student_mask = dual_bd_attn_mask_generator(
        mask_flag_fn_student, 
        dtype=torch.float32, 
        device=device
    )
    
    print(f"Student Mask Shape: {student_mask.shape}")
    visualize_mask(student_mask, "Student Attention Mask", os.path.join(output_dir, "student_attn_mask.png"))

    # Teacher Mask
    mask_flag_fn_teacher = partial(
        dual_bd_attn_mask_teacher,
        batch_size=B,
        num_kv_heads=1,
        q_ids=torch.arange(full_seq_len, device=device)[:, None],
        kv_ids=torch.arange(full_seq_len, device=device)[None, :],
        block_size=block_size,
        seq_len=L_padded
    )
    
    teacher_mask = dual_bd_attn_mask_generator(
        mask_flag_fn_teacher, 
        dtype=torch.float32, 
        device=device
    )
    
    print(f"Teacher Mask Shape: {teacher_mask.shape}")
    visualize_mask(teacher_mask, "Teacher Attention Mask", os.path.join(output_dir, "teacher_attn_mask.png"))
    
    print("\n" + "="*60)
    print("STEP 3: Loss Computation Test")
    print("="*60)
    
    losses = compute_loss(
        input_ids=input_ids,
        denoiser=denoiser,
        question_length=prompt_length,
        mask_id=mask_id,
        block_size=block_size,
        enable_shift=True, 
        share_steps=False, 
        self_align=True, # Enable to check teacher path
        feature_align=False, 
        self_step=0, 
        eos_id=eos_id
    )
    
    print(f"Loss computed successfully: {losses['loss'].item():.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

