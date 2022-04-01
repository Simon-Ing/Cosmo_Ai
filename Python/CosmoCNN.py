from torch.utils.data import DataLoader
from deepshit import *
import time
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast

# Training parameters
num_epochs = 100
batch_size = 128
learning_rate = 0.001

n_train_samples = 100
n_test_samples = 100
img_size = 512

# set to true when you want new data points
gen_new_train = True
gen_new_test = True
load_checkpoint = True
checkpoint_path = "Models/autosave/autosave_epoch40"

device = cuda_if_available()  # Use cuda if available

# Initialize your network, loss function, optimizer and scheduler
model = AlexNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)


# Load a dataset for training and one for verification
train_dataset = dataset_from_png(n_samples=n_train_samples, size=img_size, folder="train", gen_new=gen_new_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = dataset_from_png(n_samples=n_test_samples, size=img_size, folder="test", gen_new=gen_new_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# Load from checkpoint if desired:
if (load_checkpoint):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
        
# Test network prior to training
loss = test_network(test_loader, model, criterion, device)
print(f'\nAverage loss over test data before training: {loss}\n')

timer = time.time()
print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, \
        train samples: {n_train_samples} test samples: {n_test_samples} img size: {img_size}')

scaler = torch.cuda.amp.GradScaler()

# Training loop
try:
    for epoch in tqdm(range(num_epochs), desc="Total"):
        for i, (images, params) in enumerate(tqdm(train_loader, desc='Epoch')):
            images = images.to(device)
            params = params.to(device)

            optimizer.zero_grad()
            # print_images(images, params)  # print the images with parameters to make sure data is correct

            # Forward pass
            
            with autocast():
                output = model(images)
                loss = criterion(output, params)
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step(loss)
        # Test network for each epoch
        loss = test_network(test_loader, model, criterion, device, print_results=False)
        print(f"\nEpoch: {epoch+1}, Loss: {loss} lr: {optimizer.state_dict()['param_groups'][0]['lr']}, time: {time.time() - timer}\n")

        # Save checpoint
        if (epoch % 20 == 0) and (epoch > 0):
            autosave_path = "Models/autosave/" + "autosave_epoch" + str(epoch)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, autosave_path)

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
