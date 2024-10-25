import torch

n = 2000  # number of features
m = 10000  # number of training samples
k = int(0.05 * n)  # number of non-zero components

# Load the saved x_bar
x_bar = torch.load('generated_data/x_bar.pt')

features = []
labels = []

for _ in range(m):
    # sparse vector
    indices = torch.randperm(n)[:k]
    values = torch.rand(k)
    u_i = torch.zeros(n)
    u_i[indices] = values

    # s_i and label v_i
    s_i = torch.dot(x_bar, u_i)

    # assign label based on sign
    v_i = 1 if s_i > 0 else -1  

    features.append(u_i)
    labels.append(v_i)

features = torch.stack(features)
labels = torch.tensor(labels)

print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")

# train data
#torch.save({'features': features, 'labels': labels}, f'generated_data/n_{n}_m_{m}.pt')

# test data
torch.save({'features': features, 'labels': labels}, f'generated_data/n_{n}_test.pt')