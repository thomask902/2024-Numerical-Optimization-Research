import torch

n = 8000 # number of features
m = 1000 # number of samples

k = int(0.05 * n)  # number of non-zero components
a = 1.0  # norm constraint for x

# generate x with norm ≤ a
x_bar = torch.randn(n)
x_bar = x_bar / x_bar.norm() * a

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
    v_i = 1 if s_i > 0 else 0  

    features.append(u_i)
    labels.append(v_i)

features = torch.stack(features)
labels = torch.tensor(labels)

print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")

torch.save({'features': features, 'labels': labels}, f'generated_data/n_{n}_m_{m}.pt')