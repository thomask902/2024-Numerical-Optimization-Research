import torch

n = 10 # number of features
m = 2 # number of samples


features = torch.rand(m, n)
labels = torch.randint(0, 2, (m,))


print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")

torch.save({'features': features, 'labels': labels}, 'test_dataset.pt')