import os
import platform
import shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import smtplib
import ssl


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 7, (5, 5))
        self.conv3 = nn.Conv2d(7, 10, (5, 5))
        self.conv4 = nn.Conv2d(10, 10, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, 5)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv4(x)))
        x = self.pool(func.relu(self.conv4(x)))
        x = self.pool(func.relu(self.conv4(x))).view(-1, 40)
        x = self.fc2(func.relu((self.fc1(x))))
        return x


class ConvNetNew(nn.Module):
    def __init__(self):
        super(ConvNetNew, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (10, 10))
        self.conv2 = nn.Conv2d(4, 8, (10, 10))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 512)
        x = self.fc2(func.relu((self.fc1(x))))
        return x


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 32)
        # print(x.shape)
        x = self.fc1(x)
        return x

class ProConvNet(nn.Module):
    def __init__(self):
        super(ProConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 2)
        self.conv4 = nn.Conv2d(16, 16, 2)
        
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        
        x = func.max_pool2d(func.relu(self.conv1(x)), 2)
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = func.max_pool2d(func.relu(self.conv3(x)), 2)
        x = func.max_pool2d(func.relu(self.conv4(x)), 2)
        x = func.max_pool2d(func.relu(self.conv4(x)), 2)
        
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.classifier(x)
        return x

def cuda_if_available():
    if torch.cuda.is_available():
        print("Running Cuda!")
        return torch.device('cuda')
    print("Running on cpu")
    return torch.device("cpu")


def load_model(model):
    ans = ""
    while True:
        while ans not in ("y", "Y", "N", "n"):
            ans = input("Load previous model? (y/n): ")
        if ans == "n" or ans == "N":
            break
        if ans == "y" or ans == "Y":
            try:
                path = "Models/" + input("enter path: ")
                model.load_state_dict(torch.load(path))
                break
            except FileNotFoundError as e:
                print(e)
                ans = ""
            except IsADirectoryError:
                print("You must enter a file name goddamnit!")
                ans = ""


def save_model(model):
    pass
    ans = ""
    while ans not in ("y", "Y", "N", "n"):
        ans = input("Save model? (y/n): ")
    if ans == "y" or ans == "Y":
        path = "Models/" + input("Enter a name: ")
        torch.save(model.state_dict(), path)


def dataset_from_png(n_samples, size, folder, gen_new):
    if platform.system() == 'Windows':
        if gen_new:
            print(f"Started generating {folder} data")
            _, current_folder = os.path.split(os.getcwd())
            if current_folder != "python":
                os.chdir('python')
            shutil.rmtree(folder)
            os.makedirs(f'{folder}/images')
            os.system('new.exe ' + str(n_samples) + " " + str(size) + " " + str(folder))
            print("Done generating, start loading")
    else:
        if gen_new:
            print(f"Started generating {folder} data")
            os.system(f'rm -r {folder}')
            os.system(f'mkdir {folder}')
            os.system(f'mkdir {folder}/images')
            os.system('./new ' + str(n_samples) + " " + str(size) + " " + str(folder))
            print("Done generating, start loading")

    dataset = CosmoDatasetPng(str(folder))
    print("Done loading")
    return dataset


def test_network(loader, model_, criterion_, device, print_results=False):
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            params = params.to(device)
            output = model_(images)
            loss_ = criterion_(output, params)
            total_loss += loss_
            n_batches += 1
            if print_results:
                for i, param in enumerate(params):
                    niceoutput = [round(n, 4) for n in output[i].tolist()]
                    niceparam = [round(n, 4) for n in param.tolist()]
                    print(f"{f'Correct: {niceparam}' : <50} {f'Output: {niceoutput}' : ^50}")
        return total_loss / n_batches


def print_images(images, params):
    for i, img in enumerate(images):
        image = images[i].numpy().reshape(images[i].shape[1], images[i].shape[2], 1)
        image = cv2.resize(image, (400, 400))
        image = cv2.putText(image, f' dist: {str(params[i][0].item())} ein: {str(params[i][1].item())} sigma: {str(params[i][2].item())} x: {str(params[i][3].item())} y: {str(params[i][4].item())}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("img", image)
        cv2.waitKey(0)


class CosmoDatasetPng(ImageFolder):
    def __init__(self, root):
        super(CosmoDatasetPng, self).__init__(root, transform=transforms.ToTensor())
        if platform.system() == 'Windows':
            self.targets = torch.tensor([[int(a), int(b), int(c), int(d)] for (a, b, c, d) in [t[0].lstrip(
                root + "\\images\\").rstrip(".png").split(",") for t in self.imgs]], dtype=torch.float)

        else:
            self.targets = torch.tensor([[int(a), int(b), int(c), int(d)] for (a, b, c, d) in [t[0].lstrip(
                root + "/images/").rstrip(".png").split(",") for t in self.imgs]], dtype=torch.float)

    def __getitem__(self, item):
        path, _ = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample[0].view(1, sample.shape[1], sample.shape[2]), self.targets[item]


def send_email(receiver_email, message):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "simon.ing.dev@gmail.com"  # Enter your address
    password = "developer69"
    message = "Subject: Training finished\n\n" + message
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


# class CosmoDatasetJson(Dataset):
#     def __init__(self, path):
#         n_samples = 100
#         x = np.zeros((n_samples, 600*600))
#         y = np.zeros((n_samples, 3))
#         for i in range(n_samples):
#             path = "data/cosmo_data/datapoint" + str(i + 900) + ".json"
#             with open(path, "r") as infile:
#                 indata = json.load(infile)
#             x[i] = (indata["image"])
#             y[i, 0] = indata["actualPos"]
#             y[i, 1] = indata["einsteinR"]
#             y[i, 2] = indata["source_size"]
#         self.x = torch.tensor(x, dtype=torch.float).view(-1, 1, 600, 600)
#         self.y = torch.tensor(y, dtype=torch.float)
#         self.n_samples = self.x.shape[0]
#
#     def __getitem__(self, index):
#         return self.x[index], self.y[index]
#
#     def __len__(self):
#         return self.n_samples
