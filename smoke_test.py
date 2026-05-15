import torch
from transformer import Transformer

model = Transformer(1000, 1000, 128, 4, 2, 512, 64, 0.1)
model.eval()

src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))

# forward
out = model(src, tgt)
print("forward:", out.shape)
assert out.shape == (2, 8, 1000)

# generate
gen = model.generate(src, max_len=15)
print("generate:", gen.shape)
assert gen.shape[0] == 2

print("ok")