import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepshit
import time


def cuda_if_available():
    if torch.cuda.is_available():
        print("Running Cuda!")
        return torch.device('cuda')
    print("Running on cpu")
    return torch.device("cpu")


def load_model():
    ans = ""
    while ans not in ("y", "Y", "N", "n"):
        ans = input("Load previous model? (y/n): ")
    if ans == "y" or ans == "Y":
        path = input("enter path: ")
        model.load_state_dict(torch.load(path))


def save_model():
    pass
    ans = ""
    while ans not in ("y", "Y", "N", "n"):
        ans = input("Save model? (y/n): ")
    if ans == "y" or ans == "Y":
        path = "Models/" + input("Enter a name: ")
        torch.save(model.state_dict(), path)


def dataset_from_json():
    pass
    train_dataset = deepshit.CosmoDataset(path='train_dataset.json')
    test_dataset = deepshit.CosmoDataset(path='test_dataset.json')
    return (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False))


def train_network(loader, model_, lossfunc_, optimizer_):
    timer = time.time()
    n_total_steps = len(loader)
    for epoch in range(num_epochs):
        for i, (images, params) in enumerate(loader):
            images = images.to(device)
            params = params.to(device)
            # Forward pass
            output = model_(images)
            loss_ = lossfunc_(output, params)
            # Backward pass
            optimizer_.zero_grad()
            loss_.backward()
            optimizer_.step()
            print(f'Epoch: {epoch+1} / {num_epochs}\tstep: {i+1} / {n_total_steps}\tloss: {loss_.item():.10f}\ttime: {(time.time() - timer)}')


def test_network(loader, model_, lossfunc_):
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            params = params.to(device)
            output = model_(images)
            loss_ = lossfunc_(output, params)
            total_loss += loss_
            n_batches += 1
            for i, param in enumerate(params):
                nice = [round(n) for n in output[i].tolist()]
                print(f'Correct: {param.tolist()}, \tOutput: {nice}')
        return total_loss / n_batches


num_epochs = 1000
batch_size = 100
# learning_rate = 0.001

device = cuda_if_available()

# (train_loader, test_loader) = dataset_from_json()
# torch.save(train_loader, "test_loader.pt")

train_loader = torch.load("train_loader.pt")
test_loader = torch.load("test_loader.pt")
model = deepshit.ConvNet().to(device)

# load_model()

for lr in range(1, 11):
    learning_rate = lr/10000
    lossfunc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_before = test_network(test_loader, model, lossfunc)
    print(f'\nAverage loss over test data before training: {loss_before}\n')

    try:
        train_network(train_loader, model, lossfunc, optimizer)
    except KeyboardInterrupt:
        print("Keyboard interrupt, exit training loop")

    loss_train = test_network(train_loader, model, lossfunc)
    print(f'\nAverage loss over train data after training: {loss_train}\n')

    loss_test = test_network(test_loader, model, lossfunc)
    print(f'\nAverage loss over test data after training: {loss_test}\n')

    # save_model()

    with open("training_params.txt", "a") as outfile:
        outfile.write(f'Epochs: {num_epochs}\tbatch size: {batch_size}\tlearning rate: {learning_rate}\taverage loss train data: {loss_train:.1f}\taverage loss test data: {loss_test:.1f}\n')

