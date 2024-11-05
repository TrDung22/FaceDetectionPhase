import torch

# Load the scripted model
scripted_model = torch.jit.load('ckpt/pretrained/128_model.pt')
scripted_model.eval()

model_path = f"ckpt/onnx/128_model.onnx"
dummy_input = torch.randn(1, 3, 96, 128).to("cpu")
# dummy_input = torch.randn(1, 3, 480, 640).to("cuda") #if input size is 640*480
torch.onnx.export(scripted_model, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
