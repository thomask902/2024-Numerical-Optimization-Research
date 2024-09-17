import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.gnom import GNOM

# defining the model
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

# importing the dataset
wine_quality = fetch_ucirepo(id=186) 
  
X = wine_quality.data.features 
y = wine_quality.data.targets 

#print(wine_quality.metadata) 
#print(wine_quality.variables) 
#print(X.head())
#print(y.head())

# normalize features and test train split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split the dataset into train and test sets (80% train, 20% test), random state fixes seed and gives same split each time
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

inputDim = X_train.shape[1]
outputDim = y_train.shape[1]
print("Input Parameters:", inputDim)
print("Output Parameters:", outputDim)

# convert data into PyTorch Tensors
X_train_vals = torch.tensor(X_train.values, dtype=torch.float32)
X_test_vals = torch.tensor(X_test.values, dtype=torch.float32)
y_train_vals = torch.tensor(y_train.values, dtype=torch.float32)
y_test_vals = torch.tensor(y_test.values, dtype=torch.float32)

# create datasets and loaders
train_torch = torch.utils.data.TensorDataset(X_train_vals, y_train_vals)
test_torch = torch.utils.data.TensorDataset(X_test_vals, y_test_vals)
train_loader = torch.utils.data.DataLoader(dataset=train_torch, batch_size=len(train_torch), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_torch, batch_size=len(test_torch), shuffle=False)

# set up model
learningRate = 0.01 
epochs = 200
model = linearRegression(inputDim, outputDim)
model2 = linearRegression(inputDim, outputDim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# initialize loss and optimzer
criterion = torch.nn.MSELoss() 
base_optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
optimizer = GNOM(params=model.parameters(), base_optimizer=base_optimizer, model=model)

# training
print("Training with GNOM")

test_accuracy = []

for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # calls set_closure from optimizer to set up GNOM with information
        optimizer.set_closure(criterion, inputs, labels)

        # calls step() from GNOM to use info from closure to run steps
        predictions, loss = optimizer.step()
        
        total_loss += loss.item()

    # test set eval
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_loss = 0
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs, test_labels).item()

        mse = test_loss / len(test_loader)
        test_accuracy.append(mse)
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader)}, Test Loss: {mse}')

print("Training with SGD")
model2.to(device)
sgd_optimizer = torch.optim.SGD(model2.parameters(), lr=learningRate)
test_accuracy2 = []

for epoch in range(epochs):
    model2.train()  # Set model to training mode
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        sgd_optimizer.zero_grad()

        # Forward pass
        outputs = model2(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        sgd_optimizer.step()

        total_loss += loss.item()

    # test set eval
    model2.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_loss = 0
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model2(test_inputs)
            test_loss += criterion(test_outputs, test_labels).item()

        mse = test_loss / len(test_loader)
        test_accuracy2.append(mse)
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader)}, Test Loss: {mse}')

# plotting the training 
plt.plot(range(1, epochs + 1), test_accuracy, label='GNOM')
plt.plot(range(1, epochs + 1), test_accuracy2, label='SGD')
plt.xlabel('Epoch')
plt.ylabel('Test Loss (MSE)')
plt.title(f'Test Loss over {epochs} Epochs')
plt.legend()
plt.show()