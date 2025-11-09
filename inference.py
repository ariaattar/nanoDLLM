"""
Dream Inference Script
Load a trained Dream model and generate text with in-place denoising visualization.
"""

import torch
import math
import argparse
import tiktoken
from safetensors.torch import load_file
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Import all model classes from dllm.py
from dllm import (
    Dream,
    RMSNorm,
    RotaryEmbedding,
    DreamMLP,
    DreamAttention,
    DreamBlock,
    apply_rotary_pos_emb,
    rotate_half
)

console = Console()

# ============================================================================
# Generation utilities
# ============================================================================

def cosine_schedule(steps):
    """Cosine schedule for determining how many tokens to unmask per step."""
    t = torch.linspace(0, 1, steps + 1)
    alpha = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha = alpha / alpha[0]
    num_masked_schedule = alpha * 1.0
    transfer_schedule = num_masked_schedule[:-1] - num_masked_schedule[1:]
    return transfer_schedule

@torch.no_grad()
def generate(model, prompt_text, tokenizer, mask_token_id, pad_token_id,
             max_new_tokens=50, steps=64, temperature=1.0, top_k=50, device='cpu', base_vocab_size=None):
    """
    Generate text using iterative denoising with in-place visualization.

    Args:
        model: Trained Dream model
        prompt_text: String prompt
        tokenizer: tiktoken tokenizer
        mask_token_id: Mask token ID
        pad_token_id: Padding token ID
        max_new_tokens: Number of tokens to generate
        steps: Number of denoising steps
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device to run on
        base_vocab_size: Base vocabulary size (excluding special tokens)
    """
    model.eval()

    # Encode prompt using tiktoken
    prompt_ids = torch.tensor(tokenizer.encode(prompt_text), device=device)

    B = 1
    prompt_len = len(prompt_ids)
    T = prompt_len + max_new_tokens

    # Initialize canvas
    x = torch.full((B, T), pad_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt_ids
    x[0, prompt_len:] = mask_token_id

    # Attention mask (None for bidirectional attention)
    attention_mask = None

    # Get transfer schedule
    mask_index = x == mask_token_id
    total_masked = mask_index.sum(dim=1)
    schedule = cosine_schedule(steps).to(device)

    num_transfer_per_step = []
    for i in range(B):
        transfers = (schedule * total_masked[i]).round().long()
        num_transfer_per_step.append(transfers)

    # Iterative denoising with rich TUI visualization
    if base_vocab_size is None:
        base_vocab_size = tokenizer.n_vocab

    console.print()  # Spacing

    # Track initial masks for progress calculation
    initial_masks = (x == mask_token_id).sum().item()

    with Live(console=console, refresh_per_second=10) as live:
        for step in range(steps):
            mask_index = x == mask_token_id

            if not mask_index.any():
                break

            # Forward pass
            logits = model(x, attention_mask)

            # AR-shift for next-token prediction alignment
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Get logits for masked positions
            mask_logits = logits[mask_index]

            # Sample tokens with temperature and top-k
            if temperature > 0 and top_k > 0:
                mask_logits = mask_logits / temperature
                top_k_logits, top_k_indices = torch.topk(mask_logits, min(top_k, mask_logits.size(-1)))
                mask_logits = torch.full_like(mask_logits, float('-inf'))
                mask_logits.scatter_(1, top_k_indices, top_k_logits)

                probs = torch.nn.functional.softmax(mask_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                confidence = torch.gather(probs, 1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            else:
                probs = torch.nn.functional.softmax(mask_logits, dim=-1)
                confidence, sampled_tokens = probs.max(dim=-1)

            # Scatter confidence back to full sequence
            full_confidence = torch.full_like(x, float('-inf'), dtype=torch.float)
            full_confidence[mask_index] = confidence

            # Commit tokens for each sequence based on schedule
            for i in range(B):
                num_to_transfer = num_transfer_per_step[i][step].item()
                if num_to_transfer > 0 and mask_index[i].any():
                    _, transfer_indices = torch.topk(full_confidence[i], min(num_to_transfer, mask_index[i].sum().item()))

                    x_new = torch.full_like(x[i], mask_token_id)
                    mask_i = mask_index[i]
                    x_new[mask_i] = sampled_tokens[mask_index[:i].sum():mask_index[:i+1].sum()]

                    x[i, transfer_indices] = x_new[transfer_indices]

            # Build rich Text object with color coding
            current_ids = x[0].cpu().tolist()
            rich_text = Text()

            # Decode tokens and build colored text
            current_group = []

            for tid in current_ids:
                if tid == pad_token_id:
                    continue  # Skip padding
                elif tid == mask_token_id:
                    # Decode accumulated tokens before adding mask
                    if current_group:
                        decoded = tokenizer.decode(current_group)
                        rich_text.append(decoded, style="green")
                        current_group = []
                    # Add masked token in dim style
                    rich_text.append("█", style="dim white")
                elif tid < base_vocab_size:
                    # Accumulate valid tokens
                    current_group.append(tid)
                else:
                    # Unknown token
                    if current_group:
                        decoded = tokenizer.decode(current_group)
                        rich_text.append(decoded, style="green")
                        current_group = []
                    rich_text.append("?", style="red")

            # Decode any remaining tokens
            if current_group:
                decoded = tokenizer.decode(current_group)
                rich_text.append(decoded, style="green")

            # Calculate progress
            remaining_masks = mask_index.sum().item()
            progress_pct = 100 * (1 - remaining_masks / max(1, initial_masks))

            # Create display panel
            panel = Panel(
                rich_text,
                title=f"[cyan]Step {step+1}/{steps}[/cyan]",
                subtitle=f"[yellow]{progress_pct:.0f}% complete[/yellow]",
                border_style="blue",
                expand=False
            )

            live.update(panel)

    console.print()

    return x

# ============================================================================
# Model loading and inference
# ============================================================================

def load_model(model_path='final_dream_model.safetensors', config_path='dream_config.pt', device='cpu'):
    """Load trained model and config."""
    console.print(f"[dim]Loading config from {config_path}...[/dim]")

    try:
        config = torch.load(config_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            "Please train the model first using dllm.py"
        )

    # Check if this is an old config format
    if 'stoi' in config or 'itos' in config:
        raise ValueError(
            "This config file uses the old character-level encoding format.\n"
            "Please retrain your model using the updated dllm.py with tiktoken support.\n"
            "Run: python dllm.py"
        )

    # Validate required fields
    required_fields = ['vocab_size', 'base_vocab_size', 'hidden_size', 'num_layers',
                       'num_heads', 'mask_token_id', 'pad_token_id']
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(
            f"Config file is missing required fields: {missing_fields}\n"
            "Please retrain your model using the updated dllm.py"
        )

    # Load tiktoken tokenizer
    tokenizer_name = config.get('tokenizer_name', 'gpt2')
    console.print(f"[dim]Loading tokenizer: {tokenizer_name}...[/dim]")
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    console.print(f"[dim]Loading model from {model_path}...[/dim]")
    model = Dream(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_key_value_heads=config.get('num_key_value_heads', config['num_heads']),
        intermediate_size=config.get('intermediate_size', config['hidden_size'] * 4),
        dropout=config.get('dropout', 0.0),
        rms_norm_eps=config.get('rms_norm_eps', 1e-6),
        mask_token_id=config['mask_token_id'],
        pad_token_id=config['pad_token_id']
    ).to(device)

    try:
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Please train the model first using dllm.py"
        )

    model.eval()

    console.print(f"[green]✓ Model loaded[/green] [dim]({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)[/dim]")

    return model, config, tokenizer

def interactive_mode(model, config, tokenizer, device='cpu'):
    """Interactive generation mode."""
    console.print("\n[bold cyan]Dream Interactive Generation[/bold cyan]")
    console.print("[dim]Type 'quit' or 'exit' to stop[/dim]\n")
    console.print("[bold]Commands:[/bold]")
    console.print("  [cyan]/temp <value>[/cyan]   - Set temperature (default: 1.0)")
    console.print("  [cyan]/steps <value>[/cyan]  - Set denoising steps (default: 64)")
    console.print("  [cyan]/tokens <value>[/cyan] - Set max new tokens (default: 50)")
    console.print("  [cyan]/topk <value>[/cyan]   - Set top-k sampling (default: 50)\n")

    temperature = 1.0
    steps = 64
    max_new_tokens = 50
    top_k = 50

    while True:
        try:
            prompt = input("\n> ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                console.print("[dim]Goodbye![/dim]")
                break

            # Handle commands
            if prompt.startswith('/temp '):
                temperature = float(prompt.split()[1])
                console.print(f"[green]Temperature set to {temperature}[/green]")
                continue
            elif prompt.startswith('/steps '):
                steps = int(prompt.split()[1])
                console.print(f"[green]Steps set to {steps}[/green]")
                continue
            elif prompt.startswith('/tokens '):
                max_new_tokens = int(prompt.split()[1])
                console.print(f"[green]Max new tokens set to {max_new_tokens}[/green]")
                continue
            elif prompt.startswith('/topk '):
                top_k = int(prompt.split()[1])
                console.print(f"[green]Top-k set to {top_k}[/green]")
                continue

            if not prompt:
                continue

            generated = generate(
                model,
                prompt,
                tokenizer,
                config['mask_token_id'],
                config['pad_token_id'],
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                device=device,
                base_vocab_size=config['base_vocab_size']
            )

            # Decode final result using tiktoken
            generated_ids = generated[0].cpu().tolist()
            base_vocab_size = config['base_vocab_size']
            # Filter out special tokens
            generated_ids = [id for id in generated_ids if id != config['pad_token_id'] and id < base_vocab_size]
            decoded = tokenizer.decode(generated_ids) if generated_ids else ""

            console.print(f"\n[bold]{decoded}[/bold]\n")

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit or continue with a new prompt.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description='Dream Model Inference')
    parser.add_argument('--model', type=str, default='final_dream_model.safetensors',
                        help='Path to model weights')
    parser.add_argument('--config', type=str, default='dream_config.pt',
                        help='Path to model config')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt text (if not provided, enters interactive mode)')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--steps', type=int, default=64,
                        help='Number of denoising steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu/mps)')

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    console.print(f"[dim]Using device: {device}[/dim]")

    # Load model
    model, config, tokenizer = load_model(args.model, args.config, device)

    # Generate
    if args.prompt:
        # Single generation
        console.print(f"\n[cyan]>[/cyan] {args.prompt}")

        generated = generate(
            model,
            args.prompt,
            tokenizer,
            config['mask_token_id'],
            config['pad_token_id'],
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            base_vocab_size=config['base_vocab_size']
        )

        # Decode using tiktoken
        generated_ids = generated[0].cpu().tolist()
        base_vocab_size = config['base_vocab_size']
        generated_ids = [id for id in generated_ids if id != config['pad_token_id'] and id < base_vocab_size]
        decoded = tokenizer.decode(generated_ids) if generated_ids else ""

        console.print(f"[bold]{decoded}[/bold]\n")
    else:
        # Interactive mode
        interactive_mode(model, config, tokenizer, device)

if __name__ == "__main__":
    main()
