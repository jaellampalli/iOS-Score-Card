import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# code derived from YouTube creator Aladdin Persson's PyTorch tutorial series

# neural network class. methods for initialization and forward propagation. backward propagation is done automatically
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # activation function
        x = self.fc2(x)
        return x

model = NN(784, 10)
x = torch.randn(64, 784)

# determines hardware that is used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# specifications for model
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# gather MNIST datasets and create loaders
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# create model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# backpropagation specifications
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # backward propagation
        optimizer.zero_grad() # reset grads
        loss.backward() # update grads

        optimizer.step() # update weights, i.e. learn

# prints accuracy of model for given loader input (either training or test data)
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval() # sets model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1) # onehot max, i.e. model's prediction
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train() # sets model back to train mode

# calls accuracy function with training and testing data
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

