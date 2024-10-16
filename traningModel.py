import torch
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

class model(nn.Module):
    #define neural network model

    def __init__(self, numClass):


        super(model, self).__init__()

        #load pretrained resnet18 model
        self.resnet = models.resnet18(pretrained=True)
        #resnet18 is a 18 layered CNN

        #freeze resner layers
        for i in self.resnet.parameters():
            i.requires_grad = False
        #layers to corospond to the five classes (5 animals)
        features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(features, numClass)

    # forward pass thro resnet

    def forward(self, x):
        output = self.resnet(x)
        return output

class imageDs(Dataset):
    #define custom dataset
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
    def __len__(self):
        #retturn numbre of samples
        return sum(len(files) for _, _, files in os.walk(self.root_dir))

    def __getitem__(self, index):
        #add logic to get and process the sample
        classIdx = index % len(self.classes)
        classFolder = os.path.join(self.root_dir, self.classes[classIdx])

        imgFile = os.listdir(classFolder)
        fileIdx = index % len(imgFile)
        imgPatch = os.path.join(classFolder, imgFile[fileIdx])
        image = Image.open(imgPatch)

        if self.transform:
            image = self.transform(image)
        return image, classIdx

def trainModel():
    #trains model
    numClass = 5
    batchSize = 128
    epochNum = 15

    transform = transforms.Compose([
        #add neccaset=ry transformations
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])


    #create dataset object
