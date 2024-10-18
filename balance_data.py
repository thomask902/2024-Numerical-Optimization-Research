import torch
import os

# Load the original training dataset
train_data = torch.load('generated_data/n_2000_test.pt')
train_features = train_data['features']
train_labels = train_data['labels'].unsqueeze(1).float()

# Convert labels to a 1D tensor for easier indexing
train_labels_flat = train_labels.squeeze()

# Get the indices of samples for each class
indices_class_0 = torch.nonzero(train_labels_flat == 0, as_tuple=False).squeeze()
indices_class_1 = torch.nonzero(train_labels_flat == 1, as_tuple=False).squeeze()

# Determine the minority and majority classes
num_class_0 = len(indices_class_0)
num_class_1 = len(indices_class_1)

if num_class_0 < num_class_1:
    minority_class_indices = indices_class_0
    majority_class_indices = indices_class_1
    minority_class_size = num_class_0
else:
    minority_class_indices = indices_class_1
    majority_class_indices = indices_class_0
    minority_class_size = num_class_1

print(f"Number of samples before balancing:")
print(f"Class 0: {num_class_0}")
print(f"Class 1: {num_class_1}")

# Randomly select samples from the majority class to match the minority class size
# Use torch.randperm to shuffle the indices
shuffled_majority_indices = majority_class_indices[torch.randperm(len(majority_class_indices))]
selected_majority_indices = shuffled_majority_indices[:minority_class_size]

# Combine the indices from both classes
balanced_indices = torch.cat((minority_class_indices, selected_majority_indices))

# Shuffle the combined indices to mix the classes
balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

# Select the features and labels for the balanced dataset
balanced_features = train_features[balanced_indices]
balanced_labels = train_labels[balanced_indices]

# Verify that the classes are now balanced
balanced_labels_flat = balanced_labels.squeeze()
num_class_0_balanced = (balanced_labels_flat == 0).sum().item()
num_class_1_balanced = (balanced_labels_flat == 1).sum().item()
print(f"\nNumber of samples after balancing:")
print(f"Class 0: {num_class_0_balanced}")
print(f"Class 1: {num_class_1_balanced}")

# Prepare the balanced dataset dictionary
balanced_data = {
    'features': balanced_features,
    'labels': balanced_labels.squeeze()
}

# Ensure the output directory exists
os.makedirs('generated_data', exist_ok=True)

# Save the balanced dataset to a new file
output_path = 'generated_data/n_2000_test_balanced.pt'
torch.save(balanced_data, output_path)

print(f"\nBalanced dataset saved to {output_path}")
