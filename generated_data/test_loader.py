import torch
from torch.utils.data import DataLoader, TensorDataset

data = torch.load('generated_data/test_dataset.pt')
features = data['features']
labels = data['labels']

dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for idx, (features, label) in enumerate(loader):
    if idx == 3:
        break
    print(f'features: {features}')
    print(f'label: {label}')
