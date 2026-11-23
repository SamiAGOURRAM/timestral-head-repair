"""Free: print the config and the parameter shapes / norms. No model run."""
import torch
from harness import Bench, passes_used

b = Bench()
m = b.base
print(m.config)
for n, p in m.named_parameters():
    print(f"{n:60s} {tuple(p.shape)!s:18s} norm {p.norm():8.3f}")
print("rope inv_freq", m.transformer.rope.inv_freq)
print("passes used", passes_used())
