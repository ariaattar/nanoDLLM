import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
from safetensors.torch import save_file, load_file
import random
import numpy as np
import os
import tiktoken

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextDataset(Dataset):
    def __init__(self, text, block_size, mask_token_id, tokenizer):
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab

        # Encode the text using tiktoken
        self.encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.encoded) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.block_size]
        return chunk

def load_dataset(block_size, batch_size, mask_token_id, tokenizer_name='gpt2'):
    # Initialize tiktoken tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    train_text = text[:int(0.9 * len(text))]
    val_text = text[int(0.9 * len(text)):]

    train_dataset = TextDataset(train_text, block_size, mask_token_id, tokenizer)
    val_dataset = TextDataset(val_text, block_size, mask_token_id, tokenizer)

    vocab_size = train_dataset.vocab_size

    # Use num_workers=0 to avoid multiprocessing issues that can cause hangs
    # Especially important on some systems/GPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Changed from 2 to 0 to avoid deadlocks
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 2 to 0 to avoid deadlocks
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, train_dataset, vocab_size, tokenizer

# ============================================================================
# RMSNorm - simpler than LayerNorm, no mean centering
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Keep in float32 for stability
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        # x: [batch, num_heads, seq_len, head_dim]
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, head_dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, head_dim]
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ============================================================================
# Gated MLP (SwiGLU-style) - same as Llama
# ============================================================================
class DreamMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: silu(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ============================================================================
# Bidirectional Attention with Grouped Query Attention
# ============================================================================
class DreamAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Q, K, V projections with GQA
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, num_heads * head_dim]
        k = self.k_proj(x)  # [B, T, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, T, num_kv_heads * head_dim]

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, T, head_dim]

        # Apply RoPE
        cos, sin = self.rotary_emb(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat K, V for grouped query attention
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention (BIDIRECTIONAL - no causal mask)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # CRITICAL: Bidirectional for diffusion
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)

# ============================================================================
# Dream Transformer Block
# ============================================================================
class DreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, intermediate_size, dropout=0.0, rms_norm_eps=1e-6):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DreamAttention(hidden_size, num_heads, num_key_value_heads, dropout)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DreamMLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        # Pre-norm + attention + residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # Pre-norm + MLP + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

# ============================================================================
# Dream Model - Masked Diffusion Language Model
# ============================================================================
class Dream(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8,
                 num_key_value_heads=8, intermediate_size=2048, dropout=0.0,
                 rms_norm_eps=1e-6, mask_token_id=None, pad_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DreamBlock(hidden_size, num_heads, num_key_value_heads, intermediate_size, dropout, rms_norm_eps)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        x = self.embed_tokens(input_ids)  # [B, T, hidden_size]

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm and project to vocab
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

# ============================================================================
# Masked Diffusion Training
# ============================================================================
def create_masking_schedule(batch_size, seq_len, device):
    """Sample masking ratio from uniform distribution."""
    # Sample t ~ Uniform(0, 1) for each sequence
    t = torch.rand(batch_size, device=device)

    # Number of tokens to mask per sequence
    num_masked = (t * seq_len).long()
    num_masked = torch.clamp(num_masked, min=1, max=seq_len)

    return t, num_masked

def create_random_mask(input_ids, num_masked, mask_token_id):
    """Randomly mask tokens in each sequence."""
    B, T = input_ids.shape
    device = input_ids.device

    # Store original tokens
    target = input_ids.clone()
    masked_input = input_ids.clone()
    mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)

    for i in range(B):
        # Randomly select positions to mask
        indices = torch.randperm(T, device=device)[:num_masked[i]]
        mask_indices[i, indices] = True
        masked_input[i, indices] = mask_token_id

    return masked_input, mask_indices, target

def compute_cart_weights(mask_indices, t, p=0.3):
    """
    Compute CART (Context-Aware Re-weighting for Training) weights.
    Gives higher weight to tokens near unmasked context.
    """
    B, L = mask_indices.shape
    device = mask_indices.device

    # Distance matrix between all positions
    idx = torch.arange(L, device=device)
    dist_matrix = (idx[None, :] - idx[:, None]).abs() - 1
    dist_matrix = torch.clamp(dist_matrix, min=0)  # [L, L]

    # Geometric decay based on distance
    log_p = torch.log(torch.tensor(p, device=device))
    log_1_minus_p = torch.log(torch.tensor(1 - p, device=device))
    geo_matrix = (log_p + (dist_matrix - 1).clamp(min=0) * log_1_minus_p).exp() * 0.5
    geo_matrix.masked_fill_(dist_matrix == 0, 0.0)  # Ignore distance = 0

    # Weight based on proximity to unmasked tokens
    valid_mask = (~mask_indices).float()  # [B, L], 1 = unmasked
    weights = valid_mask @ geo_matrix.T  # [B, L]
    weights = weights * mask_indices.float()  # Only weight masked positions

    return weights

def train_step(model, input_ids, mask_token_id, use_cart_weights=True):
    """Single training step for masked diffusion."""
    B, T = input_ids.shape
    device = input_ids.device

    # Sample masking ratio and create masks
    t, num_masked = create_masking_schedule(B, T, device)
    masked_input, mask_indices, target = create_random_mask(input_ids, num_masked, mask_token_id)

    # Forward pass
    logits = model(masked_input)

    # CRITICAL: AR shift for next-token prediction alignment
    # This ensures training matches the generation process where logits[:, i-1] predicts token[:, i]
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    # Compute loss only on masked positions
    if use_cart_weights:
        # Use CART weighting for better training
        weights = compute_cart_weights(mask_indices, t, p=0.3)
        weights = weights[mask_indices]

        # Weighted cross-entropy
        loss_per_token = F.cross_entropy(
            logits[mask_indices],
            target[mask_indices],
            reduction='none'
        )
        loss = (loss_per_token * weights).sum() / weights.sum()
    else:
        # Standard cross-entropy on masked tokens
        loss = F.cross_entropy(
            logits[mask_indices],
            target[mask_indices]
        )

    return loss

# ============================================================================
# Iterative Denoising Generation
# ============================================================================
def cosine_schedule(steps):
    """Cosine schedule for determining how many tokens to unmask per step."""
    # Alpha schedule from MaskGIT/Dream
    t = torch.linspace(0, 1, steps + 1)
    alpha = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha = alpha / alpha[0]

    # Number of masked tokens at each step
    num_masked_schedule = alpha * 1.0  # Will be scaled by actual mask count

    # Number to unmask at each step
    transfer_schedule = num_masked_schedule[:-1] - num_masked_schedule[1:]

    return transfer_schedule

@torch.no_grad()
def generate(model, prompts, mask_token_id, eos_token_id, max_new_tokens=50,
             steps=64, temperature=1.0, top_k=50, device='cuda', tokenizer=None, base_vocab_size=None):
    """
    Generate text using iterative denoising (Dream-style).

    Args:
        model: Dream model
        prompts: List of prompt tensors [batch_size, prompt_len]
        mask_token_id: Token ID for mask
        eos_token_id: Token ID for EOS/padding
        max_new_tokens: Number of tokens to generate
        steps: Number of denoising steps
        temperature: Sampling temperature
        top_k: Top-k sampling
        tokenizer: Optional tokenizer to decode token IDs to text
        base_vocab_size: Base vocabulary size (excluding special tokens)
    """
    model.eval()

    if not isinstance(prompts, list):
        prompts = [prompts]

    B = len(prompts)
    prompt_lens = [p.shape[0] for p in prompts]
    T = max(prompt_lens) + max_new_tokens

    # Initialize canvas with EOS padding
    x = torch.full((B, T), eos_token_id, dtype=torch.long, device=device)

    # Place prompts (right-aligned) and masks for generation
    for i, p in enumerate(prompts):
        total_len = prompt_lens[i] + max_new_tokens
        start = T - total_len
        x[i, start:start + prompt_lens[i]] = p
        x[i, start + prompt_lens[i]:] = mask_token_id

    # Attention mask (for left-padding)
    # Note: Don't convert to .long() - SDPA expects bool or float, not long
    attention_mask = None  # For bidirectional attention, we don't need masking

    # Get transfer schedule
    mask_index = x == mask_token_id
    total_masked = mask_index.sum(dim=1)  # Per-sequence mask count
    schedule = cosine_schedule(steps).to(device)

    # Compute number of tokens to unmask per step per sequence
    num_transfer_per_step = []
    for i in range(B):
        transfers = (schedule * total_masked[i]).round().long()
        num_transfer_per_step.append(transfers)

    # Iterative denoising
    if tokenizer is not None:
        print(f"\n\033[1m🌀 Denoising ({steps} steps)\033[0m")
        print("─" * 80)

    for step in range(steps):
        mask_index = x == mask_token_id

        if not mask_index.any():
            break

        # Forward pass
        logits = model(x, attention_mask)

        # CRITICAL: AR shift for next-token prediction alignment
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Get logits for masked positions only
        mask_logits = logits[mask_index]  # [num_masked, vocab]

        # Sample tokens with temperature and top-k
        if temperature > 0 and top_k > 0:
            # Apply temperature
            mask_logits = mask_logits / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(mask_logits, min(top_k, mask_logits.size(-1)))
            mask_logits = torch.full_like(mask_logits, float('-inf'))
            mask_logits.scatter_(1, top_k_indices, top_k_logits)

            # Sample
            probs = F.softmax(mask_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = torch.gather(probs, 1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            # Greedy (argmax)
            probs = F.softmax(mask_logits, dim=-1)
            confidence, sampled_tokens = probs.max(dim=-1)

        # Scatter confidence back to full sequence
        full_confidence = torch.full_like(x, float('-inf'), dtype=torch.float)
        full_confidence[mask_index] = confidence

        # Commit tokens for each sequence based on schedule
        for i in range(B):
            num_to_transfer = num_transfer_per_step[i][step].item()
            if num_to_transfer > 0 and mask_index[i].any():
                # Select top-k confident positions
                _, transfer_indices = torch.topk(full_confidence[i], min(num_to_transfer, mask_index[i].sum().item()))

                # Prepare sampled tokens
                x_new = torch.full_like(x[i], mask_token_id)
                mask_i = mask_index[i]
                x_new[mask_i] = sampled_tokens[mask_index[:i].sum():mask_index[:i+1].sum()]

                # Commit selected tokens
                x[i, transfer_indices] = x_new[transfer_indices]

        # Print progress - show actual denoised text
        if tokenizer is not None and base_vocab_size is not None:
            # Decode current state for first sequence
            current_ids = x[0].cpu().tolist()

            # Decode tokens in groups for proper BPE handling
            decoded_parts = []
            current_group = []

            for tid in current_ids:
                if tid == eos_token_id:
                    continue  # Skip padding
                elif tid == mask_token_id:
                    # Decode accumulated tokens before adding mask
                    if current_group:
                        decoded_parts.append(tokenizer.decode(current_group))
                        current_group = []
                    # Use dim gray color for masks
                    decoded_parts.append('\033[2m▒\033[0m')
                elif tid < base_vocab_size:
                    # Accumulate valid tokens
                    current_group.append(tid)
                else:
                    # Unknown token
                    if current_group:
                        decoded_parts.append(tokenizer.decode(current_group))
                        current_group = []
                    decoded_parts.append('?')

            # Decode any remaining tokens
            if current_group:
                decoded_parts.append(tokenizer.decode(current_group))

            decoded_text = ''.join(decoded_parts)
            remaining_masks = mask_index.sum().item()

            # Calculate progress percentage
            total_masks = (x[0] == mask_token_id).sum().item() if step == 0 else getattr(generate, '_initial_masks', remaining_masks)
            if step == 0:
                generate._initial_masks = remaining_masks
            progress = 100 * (1 - remaining_masks / max(1, generate._initial_masks))

            # Progress bar
            bar_width = 20
            filled = int(bar_width * progress / 100)
            bar = '█' * filled + '░' * (bar_width - filled)

            # Clear line and print with progress bar
            print(f"\r\033[K\033[36m{bar}\033[0m {progress:5.1f}% │ {decoded_text[:60]}",
                  end='', flush=True)

    # Print completion message
    if tokenizer is not None:
        # Clear the progress line and show completion
        print(f"\r\033[K\033[32m{'█' * 20}\033[0m 100.0% │ \033[1m✓ Complete\033[0m")
        print("─" * 80)

    return x

# ============================================================================
# Training Loop
# ============================================================================
def train(model, train_loader, val_loader, optimizer, mask_token_id, num_epochs=3,
          max_steps=None, eval_interval=100, use_cart_weights=True):
    best_val_loss = float('inf')
    training_time_ms = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    total_steps = max_steps if max_steps else num_epochs * len(train_loader)
    current_step = 0

    def evaluate(model, val_loader):
        print("  Computing validation loss...")
        model.eval()
        total_val_loss = 0
        val_steps = 0

        print(f"  Starting validation loop (total batches: {len(val_loader)})...")
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                print(f"    Processing validation batch {batch_idx + 1}/{len(val_loader)}", end='\r')
                data = data.to(device)
                loss = train_step(model, data, mask_token_id, use_cart_weights)
                total_val_loss += loss.item()
                val_steps += 1
        print(f"\n  Validation loss computed over {val_steps} batches")

        # Generate sample text during evaluation
        print("  Preparing sample generation...")
        with torch.no_grad():
            # Take first sequence from validation as prompt
            sample_data = next(iter(val_loader)).to(device)
            prompt_len = min(10, sample_data.shape[1] // 4)  # Use first 25% as prompt
            prompt_ids = sample_data[0, :prompt_len]
            print(f"  Running generation with {prompt_len} prompt tokens, 40 new tokens, 16 steps...")

            generated = generate(
                model,
                [prompt_ids],
                mask_token_id=mask_token_id,
                eos_token_id=pad_token_id,
                max_new_tokens=40,
                steps=16,  # Fewer steps for faster eval
                temperature=0.8,
                top_k=50,
                device=device,
                tokenizer=tokenizer,
                base_vocab_size=vocab_size - 2
            )

            print("  Decoding generated tokens...")
            # Decode
            generated_ids = generated[0].cpu().tolist()
            generated_ids = [id for id in generated_ids if id != pad_token_id]

            # Decode using tiktoken
            base_vocab = vocab_size - 2  # Remove mask and pad tokens
            # Filter out special tokens and decode
            prompt_ids_filtered = [id for id in prompt_ids.cpu().tolist() if id < base_vocab]
            generated_ids_filtered = [id for id in generated_ids if id < base_vocab]

            prompt_text = tokenizer.decode(prompt_ids_filtered) if prompt_ids_filtered else ""
            decoded = tokenizer.decode(generated_ids_filtered) if generated_ids_filtered else ""

            print(f"  Sample: '{decoded[:100]}{'...' if len(decoded) > 100 else ''}'")

        model.train()
        return total_val_loss / val_steps

    model.train()
    while current_step <= total_steps:
        last_step = (current_step == total_steps)

        # if last_step or (current_step > 0 and current_step % eval_interval == 0):
        #     if torch.cuda.is_available():
        #         torch.cuda.synchronize()
        #     training_time_ms += 1000 * (time.time() - t0)

        #     print("Evaluating on validation set...")
        #     val_loss = evaluate(model, val_loader)

        #     print(f'step:{current_step}/{total_steps} val_loss:{val_loss:.4f} '
        #           f'train_time:{training_time_ms:.0f}ms '
        #           f'step_avg:{training_time_ms/max(1,current_step-1):.2f}ms')

        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         save_file(model.state_dict(), 'best_dream_model.safetensors')
        #         torch.save({
        #             'step': current_step,
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'val_loss': val_loss,
        #         }, 'best_dream_optimizer.pt')
        #         print("Saved new best model!")

        #     if torch.cuda.is_available():
        #         torch.cuda.synchronize()
        #     t0 = time.time()

        if last_step:
            break

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            loss = train_step(model, data, mask_token_id, use_cart_weights)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{current_step+1}/{total_steps} train_loss:{train_loss:.4f} "
                  f"train_time:{approx_time:.0f}ms "
                  f"step_avg:{approx_time/max(1,current_step):.2f}ms")

            current_step += 1

            if current_step >= total_steps:
                break

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    if torch.cuda.is_available():
        print(f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Save final model
    print("\nSaving final model to 'final_dream_model.safetensors'...")
    save_file(model.state_dict(), 'final_dream_model.safetensors')
    torch.save({
        'step': current_step,
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'final_dream_optimizer.pt')
    print("✓ Final model saved successfully!")

    return model

def main(
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    num_key_value_heads=8,
    intermediate_size=2048,
    block_size=256,
    batch_size=16,
    learning_rate=3e-4,
    dropout=0.0,
    rms_norm_eps=1e-6,
    num_epochs=3,
    max_steps=1000,
    use_checkpoint=False,
    use_cart_weights=True,
    seed=42
):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(seed)

    # Reserve special token IDs
    # For tiktoken, we'll add mask and pad tokens at the end of the vocab
    global train_loader, val_loader, train_dataset, vocab_size, mask_token_id, pad_token_id, tokenizer

    # Load dataset with tiktoken tokenizer
    train_loader, val_loader, train_dataset, base_vocab_size, tokenizer = load_dataset(
        block_size, batch_size, mask_token_id=None  # Will be set after we know vocab size
    )

    # Add special tokens at the end of tiktoken vocab
    mask_token_id = base_vocab_size
    pad_token_id = base_vocab_size + 1
    vocab_size = base_vocab_size + 2

    # Update datasets with correct mask token
    train_dataset.mask_token_id = mask_token_id
    val_loader.dataset.mask_token_id = mask_token_id

    # Create model
    model = Dream(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        dropout=dropout,
        rms_norm_eps=rms_norm_eps,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    if use_checkpoint:
        if os.path.exists('best_dream_model.safetensors'):
            model.load_state_dict(load_file('best_dream_model.safetensors'))
            if os.path.exists('best_dream_optimizer.pt'):
                optimizer_checkpoint = torch.load('best_dream_optimizer.pt')
                optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
            print("Loaded previous checkpoint!")

    try:
        model = train(model, train_loader, val_loader, optimizer, mask_token_id,
                     num_epochs, max_steps, use_cart_weights=use_cart_weights)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save on interrupt
        print("Saving interrupted model...")
        save_file(model.state_dict(), 'interrupted_dream_model.safetensors')
        print("Interrupted model saved!")

    # Test generation
    print("\n" + "="*80)
    print("Testing generation with prompt...")
    print("="*80)

    # Create a simple prompt
    prompt_text = "The "
    print(f"\nGenerating text with prompt: '{prompt_text}'")
    prompt_ids = torch.tensor(tokenizer.encode(prompt_text), device=device)

    generated = generate(
        model,
        [prompt_ids],
        mask_token_id=mask_token_id,
        eos_token_id=pad_token_id,
        max_new_tokens=50,
        steps=64,
        temperature=1.0,
        top_k=50,
        device=device,
        tokenizer=tokenizer,
        base_vocab_size=base_vocab_size
    )

    # Decode and print final result
    generated_ids = generated[0].cpu().tolist()
    # Remove padding and special tokens
    generated_ids = [id for id in generated_ids if id != pad_token_id and id < base_vocab_size]
    # Decode using tiktoken
    decoded = tokenizer.decode(generated_ids) if generated_ids else ""
    print(f"\n\033[1m📝 Final Result\033[0m")
    print(f"{'─'*80}")
    print(f"\033[36mPrompt:\033[0m {prompt_text}")
    print(f"\033[32mGenerated:\033[0m {decoded}")
    print(f"{'─'*80}\n")

    # Save config for inference
    print("Saving model configuration for inference...")
    config = {
        'vocab_size': vocab_size,
        'base_vocab_size': base_vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_key_value_heads': num_key_value_heads,
        'intermediate_size': intermediate_size,
        'dropout': dropout,
        'rms_norm_eps': rms_norm_eps,
        'mask_token_id': mask_token_id,
        'pad_token_id': pad_token_id,
        'tokenizer_name': 'gpt2',  # Save tokenizer name instead of vocab dicts
    }
    torch.save(config, 'dream_config.pt')
    print("✓ Model config saved to 'dream_config.pt'")

    print("\n" + "="*80)
    print("All done! You can now run inference with:")
    print("  python dream_inference.py")
    print("="*80 + "\n")

    return model

if __name__ == "__main__":
    main()
