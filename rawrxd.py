import torch

print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x

print("ok, y mean:", y.mean().item())