import torch

n = 2000  # number of features
a = 1.0   # norm constraint for x

# Generate x_bar with norm ≤ a
x_bar = torch.randn(n)
x_bar = (x_bar / x_bar.norm()) * a

# Save x_bar
torch.save(x_bar, 'generated_data/x_bar.pt')

print(f"x_bar generated with shape: {x_bar.shape}")