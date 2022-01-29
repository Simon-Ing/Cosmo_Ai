import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepshit
import time
import torch.optim.lr_scheduler as lr_scheduler

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
 

def test_network(loader, model_, lossfunc_, print_results=False):
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
            if print_results:
                for i, param in enumerate(params):
                    nice = [round(n) for n in output[i].tolist()]
                    print(f'Correct: {param.tolist()}, \tOutput: {nice}')
        return total_loss / n_batches


num_epochs = 1000
batch_size = 300
learning_rate = 0.01

device = cuda_if_available()

# (train_loader, test_loader) = dataset_from_json()
# torch.save(train_loader, "test_loader.pt")

train_loader = torch.load("train_loader.pt")
test_loader = torch.load("test_loader.pt")

# load_model()

model = deepshit.ConvNet().to(device)
lossfunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)    

loss_before = test_network(test_loader, model, lossfunc)
print(f'\nAverage loss over test data before training: {loss_before}\n')

timer = time.time()
n_total_steps = len(train_loader)
prev_loss = 100000000000
for epoch in range(num_epochs):
    for i, (images, params) in enumerate(train_loader):
        images = images.to(device)
        params = params.to(device)
        # Forward pass
        output = model(images)
        loss = lossfunc(output, params)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1} / {num_epochs}\tstep: {i+1} / {n_total_steps}\tloss: {loss.item():.10f}\ttime: {(time.time() - timer)}')
        
    scheduler.step(loss)
    if (epoch+1) % 1 == 0:
        loss = test_network(test_loader, model, lossfunc, False)
        print(f"\nLoss test data: {loss} lr: {optimizer.state_dict()['param_groups'][0]['lr']}\n")


loss_train = test_network(train_loader, model, lossfunc)
print(f'\nAverage loss over train data after training: {loss_train}\n')

loss_test = test_network(test_loader, model, lossfunc)
print(f'\nAverage loss over test data after training: {loss_test}\n')

# save_model()

with open("training_params.txt", "a") as outfile:
    outfile.write(f'Epochs: {num_epochs}\tbatch size: {batch_size}\tlearning rate: {learning_rate}\taverage loss train data: {loss_train:.1f}\taverage loss test data: {loss_test:.1f}\n')

