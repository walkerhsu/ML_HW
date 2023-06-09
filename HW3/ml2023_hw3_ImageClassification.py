# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.593407Z","iopub.execute_input":"2023-03-20T03:01:21.594164Z","iopub.status.idle":"2023-03-20T03:01:21.606912Z","shell.execute_reply.started":"2023-03-20T03:01:21.594066Z","shell.execute_reply":"2023-03-20T03:01:21.605704Z"}}
# -*- coding: utf-8 -*-
"""「ML2023-HW3-ImageClassification」的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14paOscBWZkulBqDVkYxxVD1HznFjVdKk

# HW3 Image Classification
## We strongly recommend that you run with [Kaggle](https://www.kaggle.com/t/86ca241732c04da99aca6490080bae73) for this homework

If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# Check GPU Type
"""


"""# Get Data
Notes: if the links are dead, you can download the data directly from Kaggle and upload it to the workspace, or you can use the Kaggle API to directly download the data into colab.

"""

# Download Link
# Link 1 (Dropbox): https://www.dropbox.com/s/up5q1gthsz3v0dq/food-11.zip?dl=0
# Link 2 (Google Drive): https://drive.google.com/file/d/1tbGNwk1yGoCBdu4Gi_Cia7EJ9OhubYD9/view?usp=share_link
# Link 3: Kaggle Competition.

# (1) dropbox link
# !wget -O food11.zip https://www.dropbox.com/s/up5q1gthsz3v0dq/food-11.zip?dl=0

# (2) google drive link

# !pip install gdown --upgrade
# !gdown --id '1tbGNwk1yGoCBdu4Gi_Cia7EJ9OhubYD9' --output food11.zip

# ! unzip food11.zip

# useful links
# 1) https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation (gradient_accumulation)
# 2) https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383 (custom ensemble)
# 3) https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html (nn.ModuleList)
# 4) https://stackoverflow.com/questions/55810665/changing-input-dimension-for-alexnet (change the model's layer)
# 5) https://blog.paperspace.com/popular-deep-learning-architectures-alexnet-vgg-googlenet/ (hyperparameters for each model)
# 6) https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau(scheduler)
# 7) https://zhuanlan.zhihu.com/p/250253050 (KFold)
# 8) https://chih-sheng-huang821.medium.com/03-pytorch-dataaug-a712a7a7f55e (data augmentation)
"""# Import Packages"""

# %% [markdown]
# ## HW3 Image Classification
# #### Solve image classification with convolutional neural networks(CNN).
# #### If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# %% [markdown]
# ### Import Packages

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.609337Z","iopub.execute_input":"2023-03-20T03:01:21.610227Z","iopub.status.idle":"2023-03-20T03:01:21.618891Z","shell.execute_reply.started":"2023-03-20T03:01:21.610180Z","shell.execute_reply":"2023-03-20T03:01:21.617836Z"}}
_exp_name = "CNN"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.620915Z","iopub.execute_input":"2023-03-20T03:01:21.621831Z","iopub.status.idle":"2023-03-20T03:01:21.633641Z","shell.execute_reply.started":"2023-03-20T03:01:21.621787Z","shell.execute_reply":"2023-03-20T03:01:21.632544Z"}}
# Import necessary packages.
import math
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision import models
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from accelerate import Accelerator

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.637983Z","iopub.execute_input":"2023-03-20T03:01:21.639713Z","iopub.status.idle":"2023-03-20T03:01:21.645937Z","shell.execute_reply.started":"2023-03-20T03:01:21.639668Z","shell.execute_reply":"2023-03-20T03:01:21.644841Z"}}
myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# %% [markdown]
# ### Transforms

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.647815Z","iopub.execute_input":"2023-03-20T03:01:21.648755Z","iopub.status.idle":"2023-03-20T03:01:21.658680Z","shell.execute_reply.started":"2023-03-20T03:01:21.648718Z","shell.execute_reply":"2023-03-20T03:01:21.657567Z"}}
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
transform_set_direction = [ 
    # transforms.CenterCrop(128), 
    # transforms.RandomResizedCrop((128, 128)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomRotation(degrees=(0, 180)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),
]
padding = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
transform_set_pad = [
    transforms.Pad(padding, padding_mode="edge"),
    transforms.Pad(padding, fill=fill, padding_mode="constant"),
    transforms.Pad(padding, padding_mode="symmetric"),
]
transform_set_color = [
    transforms.ColorJitter(brightness=(0, 1), contrast=(0, 1), hue=[-0.5, 0.5], saturation=(0, 1)),
    # transforms.Grayscale(num_output_channels=3),
]
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomRotation(degrees=(0, 180)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
    # transforms.RandomChoice(transform_set_color),
    # transforms.RandomChoice(transform_set_pad),
    # transforms.Resize((128, 128)),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# %% [markdown]
# ### Datasets

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.660644Z","iopub.execute_input":"2023-03-20T03:01:21.661428Z","iopub.status.idle":"2023-03-20T03:01:21.670614Z","shell.execute_reply.started":"2023-03-20T03:01:21.661390Z","shell.execute_reply":"2023-03-20T03:01:21.669451Z"}}
class FoodDataset(Dataset):

    def __init__(self,files,tfm=test_tfm):
        super(FoodDataset).__init__()
        self.files = files

        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        if idx==9995:
            save_image(im, f"tmp{idx}.jpg")
#         print(im.shape)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label

# %% [markdown]
# ### Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.723331Z","iopub.execute_input":"2023-03-20T03:01:21.724185Z","iopub.status.idle":"2023-03-20T03:01:21.741302Z","shell.execute_reply.started":"2023-03-20T03:01:21.724138Z","shell.execute_reply":"2023-03-20T03:01:21.740030Z"}}
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 1024, 3, 1, 1), # [1024, 8, 8]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [1024, 4, 4]
            
            nn.Conv2d(1024, 2048, 3, 1, 1), # [2048, 4, 4]
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [2048, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(2048*2*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 11),
            nn.BatchNorm1d(11),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# %% [markdown]
# ### Configurations
# 
# 

# %% [code]
# The number of batch size.
batch_size = 8

# The number of training epochs.
n_epochs = 250

# If no improvement in 'patience' epochs, lower learning rate.
patience = 5

# different train valid split per model
split_num = 2 

#TTA ratio
TTA_ratio = 0.8

#reload model
reload_model=True

#retrain model
retrain_model = True

# For the classification task, we use cross-entropy as the measurement of performance.
label_smoothing = 0.5
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# %% [markdown]
# ### Models

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.744070Z","iopub.execute_input":"2023-03-20T03:01:21.745153Z","iopub.status.idle":"2023-03-20T03:01:23.994961Z","shell.execute_reply.started":"2023-03-20T03:01:21.745114Z","shell.execute_reply":"2023-03-20T03:01:23.993924Z"}}
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model, optimizer and scheduler.
model_lists = []
lr = []
optimizers = []
schedules = []

output_dim = 11
threshold = 1e-4

    # 1) CNN model
for index in range(split_num):  
    lr_Classifier = 0.01
    lr.append(lr_Classifier)
    model_Classifier = Classifier().to(device)
    model_lists.append(["Classifier", model_Classifier])
    optimizers.append(torch.optim.SGD(model_Classifier.parameters(), momentum=0.9, lr=lr_Classifier, weight_decay=1e-5))
    schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

    # 2) alexnet model
# lr_Alexnet = 0.001
# lr.append(lr_Alexnet)
# model_Alexnet = models.alexnet(weights=None).to(device)
# model_Alexnet.classifier[6] = nn.Linear(4096, output_dim).to(device)
# model_lists.append(["Alexnet", model_Alexnet])
# optimizers.append(torch.optim.SGD(model_Alexnet.parameters(), momentum=0.9, lr=lr_Alexnet, weight_decay=5e-4))
# schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

#     # 3) vgg16
# lr_VGG = 0.001
# lr.append(lr_VGG)
# model_VGG = models.vgg16(weights=None).to(device)
# model_VGG.classifier[6] = nn.Linear(4096, output_dim).to(device)
# model_lists.append(["VGG", model_VGG])
# optimizers.append(torch.optim.SGD(model_VGG.parameters(), momentum=0.9, lr=lr_VGG, weight_decay=1e-5))
# schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

    # 4) RESNET
# for _ in range(split_num) :
#     lr_RESNET = 0.01
#     lr.append(lr_RESNET)
#     model_RESNET = models.resnet50(weights=None).to(device)
#     model_RESNET.fc = nn.Linear(2048, output_dim).to(device)
#     model_lists.append(["RESNET", model_RESNET])
#     optimizers.append(torch.optim.SGD(model_RESNET.parameters(), momentum=0.9, lr=lr_RESNET, weight_decay=1e-5))
#     schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

    # 5) squeezenet1_0
# model_Squeezenet = models.squeezenet1_0(weights=None).to(device)
# model_Alexnet.classifier[1] = nn.Conv2d(512, output_dim, kernel_size=(1, 1), stride=(1, 1))
# model_lists.append(["Squeezenet", model_Squeezenet])
# optimizers.append(torch.optim.Adam(model_Squeezenet.parameters(), lr=0.003, weight_decay=1e-5))
    # print all models
model_types_num = 1
model_num = len(model_lists)
for index, model_list in enumerate(model_lists):
    print(f"mode {index+1}: {model_list[0]}")
print(model_lists[0][1])
# %% [markdown]
# ### Dataloader

# %% [code] {"execution":{"iopub.status.busy":"2023-03-20T03:01:23.996646Z","iopub.execute_input":"2023-03-20T03:01:23.997018Z","iopub.status.idle":"2023-03-20T03:01:24.515874Z","shell.execute_reply.started":"2023-03-20T03:01:23.996978Z","shell.execute_reply":"2023-03-20T03:01:24.514627Z"},"jupyter":{"outputs_hidden":false}}
# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
from sklearn.model_selection import KFold

# trainPath = "/kaggle/input/ml2023spring-hw3/train"
# validPath = "/kaggle/input/ml2023spring-hw3/valid"
# testPath = "/kaggle/input/ml2023spring-hw3/test"

trainPath = "./train"
validPath = "./valid"
testPath = "./test"
modelPath = [(f"{_exp_name}_model{index + 1}_best.ckpt") for index in range(model_num)]

trainfiles = sorted([os.path.join(trainPath,x) for x in os.listdir(trainPath) if x.endswith(".jpg")])
validFiles = sorted([os.path.join(validPath,x) for x in os.listdir(validPath) if x.endswith(".jpg")])
testFiles = sorted([os.path.join(testPath,x) for x in os.listdir(testPath) if x.endswith(".jpg")])
train_valid_files = trainfiles + validFiles

kf = KFold(n_splits=split_num, shuffle=True)

train_loaders = []
valid_loaders = []
temp = []
for _ in range(model_types_num):
    for (train_index, valid_index) in kf.split(train_valid_files):  
        train_data = (np.array(train_valid_files)[train_index]).tolist()
        valid_data = (np.array(train_valid_files)[valid_index]).tolist()
        temp += valid_data
        train_set = FoodDataset(train_data, tfm=train_tfm)
        train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))
        valid_set = FoodDataset(valid_data, tfm=test_tfm)
        valid_loaders.append(DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))

assert sorted(temp) == sorted(train_valid_files)
# %% [markdown]
# ### Start Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:24.520000Z","iopub.execute_input":"2023-03-20T03:01:24.520293Z","iopub.status.idle":"2023-03-20T03:01:40.501271Z","shell.execute_reply.started":"2023-03-20T03:01:24.520263Z","shell.execute_reply":"2023-03-20T03:01:40.499471Z"}}
# Initialize trackers, these are not parameters and should not be changed

# reload model
if reload_model:
    for index in range(model_num):
        if os.path.exists(modelPath[index]):    
            print(f"[train] reload model {index+1} parameters, model_path:{modelPath[index]}")
            model_lists[index][1].load_state_dict(torch.load(modelPath[index]))
        else:
            print(f"model {index+1} not found")
else: 
    print(f"[train] restart with a new model")

stale = [0 for _ in range(model_num)]
best_accs = [0 for _ in range(model_num)]
stop_training = [False for _ in range(model_num)]

print(f"[HW3 parameters] : epoch={n_epochs}\t batch={batch_size}\t label_smoothing={label_smoothing}\t lr={lr}\t TTA={TTA_ratio}\t threshold={threshold}")

# model_lists, optimizers, schedules, train_loaders, valid_loaders = accelerator.prepare(
#     model_lists, optimizers, schedules, train_loaders, valid_loaders
# )

if retrain_model:
    for epoch in range(n_epochs):
        # These are used to record cerain train_loader information in training.
        index = 0
        for model_list, optimizer, schedule, train_loader, valid_loader in zip(model_lists, optimizers, schedules, train_loaders, valid_loaders):
            if stop_training[index] == True: 
                index += 1
                print(f"model_{index} pass...")
                continue

            model= model_list[1]
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()
            # These are used to record cerain train_loader information in training.
            train_loss = []
            train_accs = []
            for batch in tqdm(train_loader) :  
#                 with accelerator.accumulate(model):
                    # A batch consists of image data and corresponding labels.
                    imgs, labels = batch
                    #imgs = imgs.half()
                    #print(imgs.shape,labels.shape)
                    
                    # Forward the data. (Make sure data and model are on the same device.)
                    logits = model(imgs.to(device))

                    # Calculate the cross-entropy loss.
                    # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                    loss = criterion(logits, labels.to(device))

                    loss.backward()
                    # use gradient accumulate for more GPU memory
                    
#                    accelerator.backward(loss)

                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Compute the accuracy for current batch.
                    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                    # Record the loss and accuracy.
                    train_loss.append(loss.item())
                    train_accs.append(acc)
            
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            
            # ----------------Validation-----------------

            model.eval()

            valid_loss = []
            valid_accs = []
            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                #break

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            print(f"[ Train | epoch {epoch + 1:03d}/{n_epochs:03d} , model {index + 1:03d}/{model_num:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
            if valid_acc > best_accs[index]:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | epoch {epoch + 1:03d}/{n_epochs:03d} , model {index + 1:03d}/{model_num:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} => best")
            else:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | epoch {epoch + 1:03d}/{n_epochs:03d} , model {index + 1:03d}/{model_num:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # save models
            if valid_acc > best_accs[index]:
                print(f"Best model found at epoch {epoch+1}, saving model_{index + 1}")
                torch.save(model.state_dict(), f"{_exp_name}_model{index + 1}_best.ckpt") # only save best to prevent output memory exceed error
                best_accs[index] = valid_acc
                stale[index] = 0
            else:
                stale[index] += 1

            # change lr if model not improving
            schedule.step(valid_loss)
            if lr[index] != optimizer.param_groups[0]["lr"]:
                lr[index] = optimizer.param_groups[0]["lr"]
                stale[index] = 0

            # stop training
            if stale[index] == 15 :
                print(f"early stop for model_{index}")
                stop_training[index] = True
            index = index + 1

        print("")

# %% [markdown]
# ### Dataloader for test

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:40.502636Z","iopub.status.idle":"2023-03-20T03:01:40.503411Z","shell.execute_reply.started":"2023-03-20T03:01:40.503145Z","shell.execute_reply":"2023-03-20T03:01:40.503173Z"}}
# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_loaders = []
test_loader_num = 6 #( 1 for test_tfm, 5 for train_tfm)
for index in range(test_loader_num):
    if index == 0:
        test_set = FoodDataset(testFiles, tfm=test_tfm)
    else:
        test_set = FoodDataset(testFiles, tfm=train_tfm)
    test_loaders.append(DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True))

# %% [markdown]
# ### Testing and generate prediction CSV

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:40.504856Z","iopub.status.idle":"2023-03-20T03:01:40.505652Z","shell.execute_reply.started":"2023-03-20T03:01:40.505373Z","shell.execute_reply":"2023-03-20T03:01:40.505400Z"}}

model_bests = []
# model_weight = (sum(best_accs) == 0)?[1/]:[(best_acc/sum(best_accs)) for best_acc in best_accs]
# model_weight = [1, 1]
print("predict testing data...")
for index in range(model_num):
    model_best = model_lists[index][1]
    model_best.load_state_dict(torch.load(f"{_exp_name}_model{index + 1}_best.ckpt"))
    model_best.eval()
    print(f"model {index+1} ({model_lists[index][0]}) loaded")
    model_bests.append(model_best)

# model_ensemble = Ensemble(nn.ModuleList(model_bests), model_num, model_weight).to(device)
# model_ensemble.eval()
predictions = []
with torch.no_grad():
    for test_loader in test_loaders:
        predictions.append([])
        for data, _ in tqdm(test_loader):
            one_prediction = []
            for index, model in enumerate(model_bests):
                if index != 1 and index != 4:
                    continue 
                test_pred = model(data.to(device))
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                one_prediction.append(test_label)

            one_prediction = sum(one_prediction)/len(one_prediction)
            tmp_list = one_prediction.squeeze().tolist()
            predictions[-1].extend(tmp_list)

for prediction in predictions:
    for pred in prediction:
        assert pred <12 and pred > -1 

print([len(predictions[idx]) for idx in range(len(predictions))])
final_pred = []
for index, one_prediction in enumerate(predictions):
    # print(one_prediction)
    if index == 0:
        final_pred = [ ( TTA_ratio * prediction ) for prediction in one_prediction ]
    else:  # print(final_pred[pred_index], one_prediction[pred_index])
        final_pred = [ ( final_pred[pred_index] + ((1-TTA_ratio)/(test_loader_num-1)) * one_prediction[pred_index] ) for pred_index in range(len(one_prediction))]

# print(len(final_pred))
prediction = [round(pred) for pred in final_pred]

for pred in prediction:
    assert pred <12 and pred > -1 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:40.507032Z","iopub.status.idle":"2023-03-20T03:01:40.507795Z","shell.execute_reply.started":"2023-03-20T03:01:40.507533Z","shell.execute_reply":"2023-03-20T03:01:40.507560Z"}}
#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)
print("csv file saved")

# %% [markdown] {"id":"Ivk0hrE-V8Cu"}
# # Q1. Augmentation Implementation
# ## Implement augmentation by finishing train_tfm in the code with image size of your choice. 
# ## Directly copy the following block and paste it on GradeScope after you finish the code
# ### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.
# ### Your  train_tfm in the report can be different from train_tfm in your training code.
# 

# %% [code] {"id":"GSfKNo42WjKm","outputId":"156c3e9e-46ea-4805-aa33-43fffa592592"}

# %% [markdown] {"id":"3HemRgZ6WwRM"}
# # Q2. Visual Representations Implementation
# ## Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of both top & mid layers (You need to submit 2 images). 
# 

# %% [code] {"id":"iXd_SZnB2Wg8"}
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = models.resnet50(weights=None).to(device)
state_dict = torch.load(f"{_exp_name}_model4_best.ckpt")
model.load_state_dict(state_dict)
model.eval()

print(model)

# %% [code] {"id":"QcBKUNfc3BeL"}
# Load the vaildation set defined by TA
valid_set = FoodDataset("./valid", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Extract the representations for the specific layer of model
index = ... # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model.cnn[:index](imgs.to(device))
        logits = logits.view(logits.size()[0], -1)
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)
    
features = np.array(features)
colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

# Apply t-SNE to the features
features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
plt.legend()
plt.show()