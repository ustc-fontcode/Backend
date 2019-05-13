import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets

EPOCH = 10
BATCH_SIZE = 8
CLASS_NUM = 2
NET_WORKERS = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

# get data set
full_dataset = datasets.ImageFolder(root='../results', transform = transform)
print(full_dataset.class_to_idx)



train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = data.DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NET_WORKERS)
test_loader = data.DataLoader(dataset= test_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=NET_WORKERS)


# get model
model = models.resnet18(pretrained = True)
resnet_features = model.fc.in_features
model.fc = nn.Linear(resnet_features, CLASS_NUM)
model = model.to(device)

# define optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_func = torch.nn.CrossEntropyLoss()

# train model function
def train_model(loader):
    running_loss = 0
    for epoch in range(EPOCH):
        scheduler.step()
        model.train(True)
        for step, (features, target) in enumerate(loader):
            # print(step)
            # print(input[0])
            # target = torch.zeros(BATCH_SIZE, CLASS_NUM)
            # target = target.scatter_(1, target.long(), 1).long()
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step() 
        
        
            running_loss += loss.item()
            if step % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 200))
                running_loss = 0.0

# use train_dataset to train
train_model(train_loader)

print('train finished')

# val model


# start measure accuracy
model.eval()
classes = ['HuaWenSun', 'MicroSun']
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        if c.dim() < 1:
            break
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

print('measure finished')

# use last test_loader to train
train_model(test_loader)
print('all finished')