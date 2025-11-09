# nanoDLLM

Minimal PyTorch implementation of a Diffusion Language Model based on the ["Dream" paper](https://arxiv.org/html/2508.15487v1).

![Demo](assets/demo.gif)

## Setup

```bash
pip install -r requirements.txt
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Training

```bash
python dllm.py
```

Default configuration: 371.45M parameters. Edit hyperparameters in `dllm.py:main()`.

## Inference

```bash
# Single generation
python inference.py --prompt "To be or not to be"

# Interactive mode
python inference.py
```

**Interactive commands:**
- `/temp <value>` - Set temperature
- `/steps <value>` - Set denoising steps
- `/tokens <value>` - Set max tokens
- `/topk <value>` - Set top-k sampling

## Architecture

- Bidirectional transformer with RoPE, RMSNorm, SwiGLU, GQA
- Masked diffusion training with cosine schedule
- CART weighting for context-aware denoising

## License

MIT
