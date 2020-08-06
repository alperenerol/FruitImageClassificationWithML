# ---------------------------
# PACKETLER, MODULLER
# -------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from PIL import Image 
import os 
import shutil
import random

# -------------------------
# Fotoğraflar 100x100 olacak şekilde resize olur. Datasetteki tüm fotoğrafları crop ve resize ile cevirir. Training ve Testing datasetleri olusturur. Traning icin tumu, Testing icin ise datasetin ucte birini kullanir.
# -------------------------

num_classes = 7 # Sinif sayisi burada belirtilmeli.
directory_in_str = './hackathon_dataset'  # Dataset pathi burada belirtilmeli

tempDirectory_in_str = './tempDataset/Testing'
tempDirectory_in_strTrain = './tempDataset/Training'

print('Imajlar training processi icin hazirlaniyor..')

directory = os.fsencode(directory_in_str)
if (os.path.isdir('./tempDataset')) == True:
    shutil.rmtree('./tempDataset' , ignore_errors=True)
    os.makedirs(tempDirectory_in_str)
    os.makedirs(tempDirectory_in_strTrain)
else:
    os.makedirs(tempDirectory_in_str)
    os.makedirs(tempDirectory_in_strTrain)

for file in os.listdir(directory):
     
    className = os.fsdecode(file)

    directoryClass = os.fsencode(directory_in_str + '/' + className)
    os.makedirs(tempDirectory_in_str + '/' + className)
    os.makedirs(tempDirectory_in_strTrain + '/' + className)
    for file in os.listdir(directoryClass):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):          
            im = Image.open(directory_in_str + '/' + className + '/' + filename)

            width, height = im.size   # Get dimensions
            minxy = min(width, height)          
     
            # Crop the center of the image (minxy)
            left = (width - minxy)//2
            top = (height - minxy)//2
            right = (width + minxy)//2
            bottom = (height + minxy)//2
            im = im.crop((left, top, right, bottom))
            newsize = (100, 100) 
            im = im.resize(newsize) 

            if (random.random() < 0.33):
                im.save(tempDirectory_in_str + '/' + className + '/' + filename)
                im.save(tempDirectory_in_strTrain + '/' + className + '/' + filename)
            else:
                im.save(tempDirectory_in_strTrain + '/' + className + '/' + filename)              

        else:
            print('Klasorde yanlis formatta bir dosya var, gecildi')
            continue

print('Hazirlik tamamlandi..')


# -------------------------
# DATASET'I YUKLER
# -------------------------

# training işlemi süresi için sayaç başlar.
start_time = time.time()

# training ve testing için dosya pathleri
train_dict = "./tempDataset/Training"
test_dict = "./tempDataset/Testing"

# Imajlari FloatTensor e cevirir (color x height x weight) şeklinde ve normalize eder.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(train_dict, transform= transform)
test_data = datasets.ImageFolder(test_dict, transform= transform)

# batch size belirlenir (optimal 48)
batch_size = 48

# DataLoader ile datalar batchler halinde alınır.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = True)
print()
print('Size of Training Set: ' , (len(train_loader.dataset)))
print('Size of Testing Set: ' , (len(test_loader.dataset)))
print()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images[0:8]))
print()
print('Ornek bir training imaj batchi:')
print(', '.join('%5s' % train_data.classes[labels[j]] for j in range(8)))
print()
plt.show()
print('Devam etmek için grafiği kapatınız.')

# -------------------------
# HYPER-PARAMETERS
# -------------------------

num_epochs = 6
alpha = 0.0001

# ------------------------------------------
# CONVOLUTIONAL NEURAL NETWORK (CNN) MODELI
# ------------------------------------------

# 3-layer convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),            
            nn.BatchNorm2d(num_features=32),            
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=18432, out_features=num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = CNN()

print()
print('CNN Mimarisi: ' , (cnn))
print()

# -------------------------
# LOSS & OPTIMIZER
# -------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=alpha)

# -------------------------
# TRAIN MODEL
# -------------------------

# Cuda'li bir GPU varsa onunla training yapar
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print()
cnn.to(device)

loss_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # forward pass, backpropagate, optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))

# print training run time
print()
print('Run time: %.3f seconds' % (time.time() - start_time))
print()

# # training loss
# plt.plot(loss_list)
# plt.title('Training Loss')
# plt.xlabel("Epochs")

# plt.xticks(np.arange(0, (len(train_data) // batch_size)*num_epochs, step=len(train_data) // batch_size), ('0','1', '2', '3', '4', '5','6'))
# plt.ylabel("Loss")

# plt.yscale('log')
# plt.show()

# Model parametrelerini inference icin kaydeder.
torch.save(cnn.state_dict(), 'trainedModelParams.pth')

# --------------------------
# TEST MODEL
# -------------------------

print()
print('Prediction processi basladi..')
print()

cnn.cpu()

correct = 0.0
total = 0.0

accuracy_list = []

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(8):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    # model accuracy'si hesaplanir
    accuracy = 100 * correct // total
    accuracy_list.append(accuracy.data)

# class accuracy hesaplanir
for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        train_data.classes[i], 100 * class_correct[i] // class_total[i]))

# # plot test accuracy
# plt.plot(accuracy_list)
# plt.title('Test Accuracy')
# plt.xlabel("Epochs")

# plt.xticks(np.arange(0, 130, step=22), ('0','1', '2', '3', '4', '5','6'))
# plt.ylabel("Accuracy (%)")
# plt.show()

# print test accuracy
print()
print('Accuracy: %.3f %%' % (100 * correct / total))
print()

# En basta olusturulan imaj klasorunu siler 
# shutil.rmtree('./tempPrepare' , ignore_errors=True)

a = input('Press enter to exit')
if a:
    exit(0)