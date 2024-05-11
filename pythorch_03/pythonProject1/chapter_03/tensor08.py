import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.tensor([1, 2, 3], dtype=torch.float, device=device)
ndarray = tensor.detach().cpu().numpy()
print(ndarray)
print(type(ndarray))