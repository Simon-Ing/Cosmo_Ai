from torch.utils.data import DataLoader
from deepshit import *
import time
import torch.optim.lr_scheduler as lr_scheduler

# Training parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.01

n_train_samples = 1000
n_test_samples = 100
img_size = 400

# set to true when you want new data points
gen_new_train = True
gen_new_test = True

device = cuda_if_available()  # Use cuda if available

# Initialize your network, loss function, optimizer and scheduler
model = ConvNet().to(device)
load_model(model)  # Load a saved model if you want to
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# Load a dataset for training and one for verification
train_dataset = dataset_from_png(n_samples=n_train_samples, size=img_size, folder="train", gen_new=gen_new_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = dataset_from_png(n_samples=n_test_samples, size=img_size, folder="test", gen_new=gen_new_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
        
# Test network prior to training
loss = test_network(test_loader, model, criterion, device)
print(f'\nAverage loss over test data before training: {loss}\n')

timer = time.time()
print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, \
        train samples: {n_train_samples} test samples: {n_test_samples} img size: {img_size}')

# Training loop
for epoch in range(num_epochs):
    for i, (images, params) in enumerate(train_loader):
        images = images.to(device)
        params = params.to(device)

        # print_images(images, params)  # print the images with parameters to make sure data is correct

        # Forward pass
        output = model(images)
        loss = criterion(output, params)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # scheduler.step(loss)
    # Test network for each epoch
    loss = test_network(test_loader, model, criterion, device, print_results=False)
    print(f"\nEpoch: {epoch+1}, Loss test data: {loss} lr: {optimizer.state_dict()['param_groups'][0]['lr']}\n")

# Test network after training
loss_test = test_network(test_loader, model, criterion, device, True)
print(f'\nAverage loss over test data after training: {loss_test}\n')

# Save your model if you want to
# save_model(model)
