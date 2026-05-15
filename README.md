# Transformer

Encoder-decoder Transformer for sequence-to-sequence tasks.
Built from scratch in PyTorch.

## Design choices

- **Pre-LN** over Post-LN: more stable gradients, no warmup tuning needed
- **Fused W_qkv**: single matmul for Q/K/V projection in self-attention
- **KV cache**: incremental decoding in `generate`, avoids recomputing past keys/values
- **Batch-aware generation**: `is_finished` mask handles variable-length sequences without stopping early

## Usage

```python
import torch
from transformer import Transformer

model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

src = torch.randint(0, 32000, (2, 20))
tgt = torch.randint(0, 32000, (2, 15))

out = model(src, tgt)        # [2, 15, 32000]
gen = model.generate(src)    # [2, <=50]
```

## Mask format

`src_mask` and `tgt_mask` should be shaped `[B, 1, 1, seq_len]`, with 1 = attend, 0 = ignore.

## Run smoke test

```bash
python smoke_test.py
```
