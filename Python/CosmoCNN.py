from torch.utils.data import DataLoader
from deepshit import *
import time
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

# Training parameters
num_epochs = 100
batch_size = 400
learning_rate = 0.001

n_train_samples = 100000
n_test_samples = 1000
img_size = 400

# set to true when you want new data points
gen_new_train = False
gen_new_test = False

device = cuda_if_available()  # Use cuda if available

# Initialize your network, loss function, optimizer and scheduler
model = ConvNet().to(device)
load_model(model)  # Load a saved model if you want to
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=1/2)

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
try:
    for epoch in tqdm(range(num_epochs), desc="Total"):
        for i, (images, params) in enumerate(tqdm(train_loader, desc='Epoch')):
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

        scheduler.step(loss)
        # Test network for each epoch
        loss = test_network(test_loader, model, criterion, device, print_results=False)
        print(f"\nEpoch: {epoch+1}, Loss: {loss} lr: {optimizer.state_dict()['param_groups'][0]['lr']}, time: {time.time() - timer}\n")

except KeyboardInterrupt:
    print("Training aborted by keyboard interrupt.")
except TypeError:
    print("Training aborted by keyboard interrupt.")


# Test network after training
loss = test_network(test_loader, model, criterion, device, print_results=True)
message = f'Loss: {loss}\nEpochs: {num_epochs}\nbatch_size: {batch_size}\nlearning_rate: {learning_rate}\nn_train_samples: {n_train_samples}\nimg_size: {img_size}'

print(message)


# send_email('simon.ing.89@gmail.com', message)


# Save your model if you want to
save_model(model)
