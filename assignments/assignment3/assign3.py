#This code follows the tutorial found at http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html, as instructed in the homework
from __future__ import print_function, division


import time, os, copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms



dataDir = 'tiny-imagenet-5'


def defineTranforms():
    dataTransforms = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    }
    
    return dataTransforms

def createDataSet(dataDir,dataTransforms):
    imgDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms[x]) for x in ['train','val']}
    return imgDatasets

def createDataLoaders(imgDatasets):
    dataloaders = {x: torch.utils.data.DataLoader(imgDatasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train','val']}
    return dataloaders


def tensorImageView(image, title = ''):
    # This shows the image with associated title
    print("Showing image")
    image = image.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image  + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)

def trainModel(model, criteria, optimizer, scheduler, numEpochs = 50):
    startTime = time.time()
    
    bestWeights = copy.deepcopy(model.state_dict())

    bestAccuracy = 0.0

    for epoch in range(numEpochs):
        print('Startin Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-'*10)

        # first do a training phase then a validation 
        for phase in ['train','val']:
            if phase == 'train':
                #during the training phase, we have a model in training mode
                scheduler.step()
                model.train(True)
            else:
                # during validation phase we aren't in training mode
                model.train(False)
            
            runningLoss = 0.0
            runningCorrect = 0

            for data in dataLoader[phase]:

                inputs, labels = data

                if useGpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()
                
                #do forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data,1)
                loss = criterion(outputs, labels)

                #go backward only if training
                if  phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                #compute stats
                runningLoss += loss.data[0] * inputs.size(0)
                runningCorrect += torch.sum(preds == labels.data)
            
            epochLoss = runningLoss / datasetSize[phase]
            epochAccuracy = runningCorrect / datasetSize[phase]

            print('{} Loss: {:.4f} Acc:{:.4f}'.format(phase, epochLoss, epochAccuracy))


            if phase == 'val' and epochAccuracy > bestAccuracy:
                bestAccuracy = epochAccuracy
                bestWeights = copy.deepcopy(model.state_dict()) 
        
        print()

    timeElapsed = time.time() - startTime

    print('Training complete in {:.0f}m {:.0f}s'.format(
        timeElapsed // 60, timeElapsed % 60))
    print('Best val Acc: {:4f}'.format(bestAccuracy))

    model.load_state_dict(bestWeights)
    return model



if __name__ == '__main__':

    transf = defineTranforms()
    dataSet = createDataSet(dataDir,transf)
    dataLoader = createDataLoaders(dataSet)

    datasetSize = {x: len(dataSet[x]) for x in ['train', 'val']}
    classNames = dataSet['train'].classes
    useGpu = torch.cuda.is_available()
    if (useGpu):
        print("GPU is Available :) ")

    # inputs, classes = next(iter(dataLoader['train']))

    # out = torchvision.utils.make_grid(inputs)

    # tensorImageView(out, title=[classNames[x] for x in classes])


    testedModels = ['resnet50', 'vgg16', 'resnet34']
    for curModel in testedModels:
        print("Testing " + curModel + " model")

        if curModel == 'resnet50':
            modelFT = models.resnet50(pretrained=True)
        elif curModel == 'vgg16':
            modelFT = models.vgg16(pretrained=True)
        elif curModel == 'resnet34':
            modelFT = models.resnet34(pretrained=True)
        else:
            print("tested Model not implemented, aborting")
            exit(1)
        numFeatures = modelFT.fc.in_features
        modelFT.fc = nn.Linear(numFeatures, 5) # 5 outputs

        if useGpu:
            modelFT = modelFT.cuda()

        criterion = nn.CrossEntropyLoss()

        optimizerFT = optim.SGD(modelFT.parameters(), lr=0.001, momentum=0.9)
        
        expScheduler = lr_scheduler.StepLR(optimizerFT, step_size = 7, gamma=0.1)


        modelFT = trainModel(modelFT, criterion, optimizerFT, expScheduler, numEpochs=50)
        torch.save(modelFT.state_dict(),'modelFT' + curModel + '.pt') #save the model
    

        if curModel == 'resnet50':
            modelConv = torchvision.models.resnet50(pretrained=True)
        elif curModel == 'vgg16':
            modelConv = torchvision.models.vgg16(pretrained=True)
        elif curModel == 'resnet34':
            modelConv = torchvision.models.resnet34(pretrained=True)
        else:
            print("tested Model not implemented, aborting")
            exit(1)

        for param in modelConv.parameters():
            param.requires_grad = False
        
        numFeatures = modelConv.fc.in_features
        modelConv.fc = nn.Linear(numFeatures, 5)

        if useGpu:
            modelConv = modelConv.cuda()

        criterion = nn.CrossEntropyLoss()

        optimizerConv = optim.SGD(modelConv.fc.parameters(),lr = 0.001, momentum = 0.9)
        expConvScheduler = lr_scheduler.StepLR(optimizerConv,step_size=7, gamma=0.1)

        modelConv = trainModel(modelConv,criterion,optimizerConv,expConvScheduler,numEpochs=50)
        torch.save(modelFT.state_dict(),'modelConv' + curModel + '.pt') #save the model
