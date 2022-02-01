import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import deepshit
import time
import torch.optim.lr_scheduler as lr_scheduler
import os


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
        path = "Models/" + input("enter path: ")
        model.load_state_dict(torch.load(path))


def save_model():
    pass
    ans = ""
    while ans not in ("y", "Y", "N", "n"):
        ans = input("Save model? (y/n): ")
    if ans == "y" or ans == "Y":
        path = "Models/" + input("Enter a name: ")
        torch.save(model.state_dict(), path)


def dataset_from_png(n_samples, size, folder, gen_new):
    if gen_new:
        print(f"Started generating {folder} data")
        os.system(f'rm -r {folder}')
        os.system(f'mkdir {folder}')
        os.system(f'mkdir {folder}/images')
        os.system('./Data ' + str(n_samples) + " " + str(size) + " " + str(folder))
        print("Done generating, start loading")
    dataset = deepshit.CosmoDatasetPng(str(folder))
    print("Done loading")
    return dataset


def test_network(loader, model_, lossfunc_, print_results=False):
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            params = params.to(device)
            output = model_(images)
            loss_ = lossfunc_(output, params)
            if print_results:
                for i, param in enumerate(params):
                    niceoutput = [round(n, 4) for n in output[i].tolist()]
                    niceparam = [round(n, 4) for n in param.tolist()]
                    print(f'Correct: {niceparam},\t\tOutput: {niceoutput}')
        return loss_


def print_images(images, params):
    for i, img in enumerate(images):
        image = images[i].numpy().reshape(28, 28, 1)
        image = cv2.resize(image, (400,400))
        image = cv2.putText(image, str(params[i]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("img", image)
        cv2.waitKey(0)


num_epochs = 200
batch_size = 50
learning_rate = 0.001

n_train_samples = 1000
n_test_samples = 100
img_size = 400

gen_new_train = False
gen_new_test = False

device = cuda_if_available()

# load_model()

# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=1/2)

# for batch_size in range(10, 101, 10):

train_dataset = dataset_from_png(n_samples=n_train_samples, size=img_size, folder="train", gen_new=gen_new_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset_from_png(n_samples=n_test_samples, size=img_size, folder="test", gen_new=gen_new_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=n_test_samples)

model = deepshit.ConvNet().to(device)
lossfunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_before = test_network(test_loader, model, lossfunc)
print(f'\nAverage loss over test data before training: {loss_before}\n')

print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, train samples: \
    {n_train_samples} test samples: {n_test_samples} img size: {img_size}\n')

timer = time.time()
for epoch in range(num_epochs):
    # if (epoch+1) % 20 == 0:
    #     print("New data!")
    #     train_dataset = dataset_from_png(n_samples=n_train_samples, size=img_size, folder="train", gen_new=gen_new_train)
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for i, (images, params) in enumerate(train_loader):
        images = images.to(device)
        params = params.to(device)

        # print_images(images, params)  # print the images with parameters to make sure data is correct

        # Forward pass
        output = model(images)
        loss = lossfunc(output, params)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(model.state_dict())
    # scheduler.step(loss)
    if (epoch+1) % 1 == 0:
        loss = test_network(test_loader, model, lossfunc, print_results=False)
        print(f"Epoch: {epoch+1}\tLoss test data: {loss:.4f}\tlr: {optimizer.state_dict()['param_groups'][0]['lr']:.8f}\
                \ttime: {(time.time() - timer):.2f}\tbatch_size: {batch_size}\n")


loss_train = test_network(train_loader, model, lossfunc, print_results=False)
print(f'\nAverage loss over train data after training: {loss_train}\n')

loss_test = test_network(test_loader, model, lossfunc, print_results=False)
print(f'\nAverage loss over test data after training: {loss_test}\tbatch_size: {batch_size}\n')

with open("records.txt", "a") as outfile:
    outfile.write(f"epochs: {num_epochs}\tbatch_size: {batch_size}\tlr: {learning_rate}\ttrain samples: {n_train_samples}\tLoss test data: {loss_test}\n")

# save_model()


