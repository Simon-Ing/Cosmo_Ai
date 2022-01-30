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
import pandas as pd

sh = cv2.imread("Starship_SN9_Launch_Pad.jpg")


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
    train_dataset = deepshit.CosmoDataset(path='train_dataset.json')
    test_dataset = deepshit.CosmoDataset(path='test_dataset.json')
    return (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False))


def dataset_from_png(n_samples, size, folder):
    print("Started generating")
    os.system('./Data ' + str(n_samples) + " " + str(size) + " " + str(folder))
    print("Done generating, start loading")
    dataset = deepshit.CosmoDatasetPng(str(folder), transform=transforms.ToTensor())
    for i, img in enumerate(dataset.imgs):
        name = img[0].lstrip(folder + "/images/").rstrip(".png")
        params = name.split(",")
        dataset.targets[i] = torch.tensor([int(n) for n in params], dtype=torch.float)
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
            total_loss += loss_
            n_batches += 1
            if print_results:
                for i, param in enumerate(params):
                    niceoutput = [round(n, 2) for n in output[i].tolist()]
                    niceparam = [round(n, 2) for n in param.tolist()]
                    print(f'Correct: {niceparam},\t\tOutput: {niceoutput}')
        return total_loss / n_batches


def print_images(images, params):
    for i, img in enumerate(images):
        image = images[i].numpy().reshape(28, 28, 1)
        image = cv2.resize(image, (400,400))
        image = cv2.putText(image, str(params[i]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("img", image)
        cv2.waitKey(0)

num_epochs = 40
batch_size = 1000
learning_rate = 0.001

n_train_samples = 10000
n_test_samples = 1000
img_size = 100

device = cuda_if_available()

# train_dataset = dataset_from_png(n_samples=10, size=600, folder="train")
test_dataset = dataset_from_png(n_samples=n_test_samples, size=img_size, folder="test")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

model = deepshit.ConvNet().to(device)
lossfunc = nn.MSELoss()


loss_before = test_network(test_loader, model, lossfunc)
print(f'\nAverage loss over test data before training: {loss_before}\n')

timer = time.time()
try:
    while True:
        train_dataset = dataset_from_png(n_samples=n_train_samples, size=img_size, folder="train")
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        for epoch in range(num_epochs):
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
                # print(f'Epoch: {epoch+1} / {num_epochs}\tstep: {i+1} / {n_train_samples/batch_size}\tloss: {loss.item():.10f}\ttime: {(time.time() - timer)}')

            scheduler.step(loss)
            if (epoch+1) % 1 == 0:
                loss = test_network(test_loader, model, lossfunc, print_results=False)
                print(f"\nEpoch: {epoch+1}, Loss test data: {loss} lr: {optimizer.state_dict()['param_groups'][0]['lr']}\n")
                
except(KeyboardInterrupt):
    print("Keyboard interrupt Broke training loop")

loss_train = test_network(train_loader, model, lossfunc, print_results=False)
print(f'\nAverage loss over train data after training: {loss_train}\n')

loss_test = test_network(test_loader, model, lossfunc, print_results=True)
print(f'\nAverage loss over test data after training: {loss_test}\n')

save_model()

#with open("training_params.txt", "a") as outfile:
#    outfile.write(f'Epochs: {num_epochs}\tbatch size: {batch_size}\tlearning rate: {learning_rate}\taverage loss train data: {loss_train:.1f}\taverage loss test data: {loss_test:.1f}\n')
