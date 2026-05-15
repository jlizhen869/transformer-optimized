import time
import torch
from transformer import Transformer

model = Transformer(1000, 1000, 128, 4, 2, 512, 64, 0.1)
model.eval()

src = torch.randint(0, 1000, (1, 20))

# with KV cache (current generate)
times = []
for _ in range(50):
    t0 = time.time()
    model.generate(src, max_len=30)
    times.append(time.time() - t0)
avg_cache = sum(times) / len(times) * 1000
print(f"with KV cache:    {avg_cache:.1f}ms avg")

# without KV cache (full decode every step)
def generate_no_cache(model, src, max_len=30, start_token=2, end_token=3):
    device = src.device
    batch_size = src.size(0)
    memory = model.encode(src, src_mask=None)
    generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    for _ in range(max_len):
        causal_mask = model._make_causal_mask(generated)
        out = model.decode(generated, memory, causal_mask, tgt_mask=None)
        logits = model.final_linear(out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == end_token).all():
            break
    return generated[:, 1:]

times = []
for _ in range(50):
    t0 = time.time()
    generate_no_cache(model, src, max_len=30)
    times.append(time.time() - t0)
avg_no_cache = sum(times) / len(times) * 1000
print(f"without KV cache: {avg_no_cache:.1f}ms avg")

speedup = avg_no_cache / avg_cache
print(f"speedup:          {speedup:.2f}x")
