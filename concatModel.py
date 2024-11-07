import torch

model = torch.jit.load('ckpt/pretrained/128_model.pt')
model.eval()

input = torch.randn(1,3,128,96)
output = model(input)

print(model)
print(f"Input shape: {input.shape}")

for i, out in enumerate(output):
    print(f"Output {i} shape: {out.shape}")

out = torch.cat((output[0], output[1]), dim=-1)
print(f"Output shape: {out.shape}")