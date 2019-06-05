import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets
import argparse

parser = argparse.ArgumentParser()
pretrain_parser = parser.add_mutually_exclusive_group(required=False)
pretrain_parser.add_argument("--pretrained", dest='pretrain', action='store_true', help="choose to use prtrained model.")
pretrain_parser.add_argument("--no-pretrained", dest='pretrain', action='store_false', help="choose not to use pretrained model.")
parser.set_defaults(pretrain=True)
parser.add_argument("--file", action="store", help="if use pretrained model choose pkl file")
args = parser.parse_args()

EPOCH = 12 
BATCH_SIZE = 64
CLASS_NUM = 2
NET_WORKERS = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                     std=[0.5,0.5,0.5])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

# get data set
full_dataset = datasets.ImageFolder(root='../TrainData', transform = transform)
print(full_dataset.class_to_idx)



train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = data.DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NET_WORKERS)
test_loader = data.DataLoader(dataset= test_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=NET_WORKERS)


# get model
if args.pretrain:
    model = torch.load(args.file).to(device)
else:
    model = models.resnet18(pretrained = True)
    resnet_features = model.fc.in_features
    model.fc = nn.Linear(resnet_features, CLASS_NUM)
    model = model.to(device)

# define optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_func = torch.nn.CrossEntropyLoss()

# train model function
def train_model(train_loader, test_loader, need_train=True):
    classes = ['HuaWenSun', 'MicroSun']
    running_loss = 0
    running_class_correct = list(0. for i in range(2))
    running_class_total = list(0. for i in range(2))
    running_all_correct = 0
    running_all_total = 0

    for epoch in range(EPOCH):
        scheduler.step()
        model.train(True)
        for step, (features, target) in enumerate(train_loader):
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

            # cal loss    
            running_loss += loss.item()

            # cal acc
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(len(c)):
                label = target[i]
                running_class_correct[label] += c[i].item()
                running_class_total[label] += 1
                running_all_correct += c[i].item() 
                running_all_total += 1

            if step % 200 == 199:    # print every 200 mini-batches
                running_loss = running_loss / 200
                running_acc_class0 = running_class_correct[0] / running_class_total[0]
                running_acc_class1 = running_class_correct[1] / running_class_total[1]
                running_acc_total = running_all_correct / running_all_total
                print('[%d,%5d] loss: %.7f acc[%5s]: %.7f acc[%5s]: %.7f acc[total]: %7f' % (
                    epoch + 1, step + 1, running_loss, 
                    classes[0], running_acc_class0, classes[1], 
                    running_acc_class1, running_acc_total))
                running_class_correct = list(0. for i in range(2))
                running_class_total = list(0. for i in range(2))
                running_all_correct = 0
                running_all_total = 0
                running_loss = 0.0

        if need_train:
            model.eval()
            validate_class_correct = list(0. for i in range(2))
            validate_class_total = list(0. for i in range(2))
            validate_all_correct = 0
            validate_all_total = 0
            validate_loss = 0
            validate_loss_cnt = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    val_loss = loss_func(outputs, labels)
                    validate_loss += val_loss.item()
                    validate_loss_cnt += 1
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    if c.dim() < 1:
                        break
                    for i in range(len(c)):
                        label = labels[i]
                        validate_class_correct[label] += c[i].item()
                        validate_class_total[label] += 1
                        validate_all_correct += c[i].item()
                        validate_all_total += 1

                print('validation: loss: %.7f acc[%5s]: %.7f acc[%5s]: %.7f acc[total]: %7f' % (
                    validate_loss / validate_loss_cnt,
                    classes[0], validate_class_correct[0] / validate_class_total[0],
                    classes[1], validate_class_correct[1] / validate_class_total[1],
                    validate_all_correct / validate_all_total
                    ))
                
                validate_class_correct = list(0. for i in range(2))
                validate_class_total = list(0. for i in range(2))
                validate_all_correct = 0
                validate_all_total = 0
                validate_loss = 0
                validate_loss_cnt = 0
        
        # save model
        if epoch % 4 == 0:
            PATH = '../Model/ResnetModel_'+str(epoch)+'.pkl'
            torch.save(model, PATH) 
        

# use train_dataset to train
train_model(train_loader, test_loader, need_train=True)

print('train finished')

# use last test_loader to train
train_model(test_loader, test_loader, need_train=False)
print('all finished')
torch.save(model, '../Model/ResnetModel_last.pkl')  # save model
