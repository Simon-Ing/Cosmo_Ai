from deepshit import *
from torch.utils.data import DataLoader


#load model
device = cuda_if_available()  # Use cuda if available
model = AlexNet().to(device)
#model = torch.nn.DataParallel(model)
PATH = "Python/Models/73loss_distratio_200k"
#checkpoint = torch.load(PATH)

#model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(torch.load(PATH))
model.eval()


#load test data 
batch_size = 1
test_dataset = CosmoDatasetPng("test")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


# Test network 
criterion = nn.MSELoss()
loss = test_network(test_loader, model, criterion, device, print_results=True)
message = f'Loss: {loss}\n  batch_size: {batch_size}\n'
print(message)
