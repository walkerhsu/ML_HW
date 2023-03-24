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

"""# Import Packages"""

# %% [markdown]
# ## HW3 Image Classification
# #### Solve image classification with convolutional neural networks(CNN).
# #### If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# %% [markdown]
# ### Import Packages

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.609337Z","iopub.execute_input":"2023-03-20T03:01:21.610227Z","iopub.status.idle":"2023-03-20T03:01:21.618891Z","shell.execute_reply.started":"2023-03-20T03:01:21.610180Z","shell.execute_reply":"2023-03-20T03:01:21.617836Z"}}
_exp_name = "sample"

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
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision import models
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from accelerate import Accelerator
import ttach

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
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.ColorJitter(brightness=0.15, contrast=0.15, hue=0.15, saturation=0.15),
    transforms.ElasticTransform(),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.RandomRotation(degrees=(0, 180)), 
    transforms.RandomHorizontalFlip(p=1),   
    transforms.RandomVerticalFlip(p=1),     
    transforms.RandomInvert(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
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
#         im.show()
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
#             nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 1024, 3, 1, 1), # [1024, 16, 16]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [1024, 8, 8]
            
            nn.Conv2d(1024, 2048, 3, 1, 1), # [2048, 8, 8]
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [2048, 4, 4]
            
            nn.Conv2d(2048, 2048, 3, 1, 1), # [2048, 4, 4]
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [2048, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(2048*2*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Ensemble(nn.Module):
    def __init__(self, models, model_num, weight, output_dim=11):
        assert len(models) == model_num and len(weight) == model_num
        super(Ensemble, self).__init__()
        self.models = models
        self.model_num = model_num
        self.output = []
        self.output_dim = output_dim
        self.weight = weight
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]

    def forward(self, x):
        for index, model in enumerate(self.models):
            out = model(x)
            if index == 0:
                self.output = [ [ float(out_element * self.weight[index]) for out_element in out_element_list ] for out_element_list in out]
            else:
                self.output = [ [ float(self.output[index1][index2] + out_element * self.weight[index]) for index2, out_element in enumerate(out_element_list) ]
                                    for index1, out_element_list in enumerate(out)]
        output = self.output

        self.output = []
        return torch.tensor(output)

# %% [markdown]
# ### Configurations

# The number of batch size.
batch_size = 16

# The number of training epochs.
n_epochs = 1000

# If no improvement in 'patience' epochs, lower learning rate.
patience = 5

#train_valid_split rate
train_valid_split = 0.8

#TTA ratio
TTA_ratio = 0.8

#reload model
reload_model=True

#retrain model
retrain_model = True

# gradient_accumulation_steps
gradient_accumulation_steps = 5
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

# For the classification task, we use cross-entropy as the measurement of performance.
label_smoothing = 0.5
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# %% [markdown]
# ### Models

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.744070Z","iopub.execute_input":"2023-03-20T03:01:21.745153Z","iopub.status.idle":"2023-03-20T03:01:23.994961Z","shell.execute_reply.started":"2023-03-20T03:01:21.745114Z","shell.execute_reply":"2023-03-20T03:01:23.993924Z"}}
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model_lists = []

# Initialize optimizer

output_dim = 11
lr = []
optimizers = []
schedules = []
threshold = 1e-3

    # 1) CNN model
# lr_Classifier = 0.001
# lr.append(lr_Classifier)
# model_Classifier = Classifier().to(device)
# model_lists.append(["Classifier", model_Classifier])
# optimizers.append(torch.optim.Adam(model_Classifier.parameters(), lr=lr_Classifier, weight_decay=1e-5))
# schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

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
lr_RESNET = 0.01
lr.append(lr_RESNET)
model_RESNET = models.resnet50(weights=None).to(device)
model_RESNET.fc = nn.Linear(2048, output_dim).to(device)
model_lists.append(["RESNET", model_RESNET])
optimizers.append(torch.optim.SGD(model_RESNET.parameters(), momentum=0.9, lr=lr_RESNET, weight_decay=1e-5))
schedules.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="min", factor=0.1, patience=patience, threshold=threshold, verbose=True))

    # 5) squeezenet1_0
# model_Squeezenet = models.squeezenet1_0(weights=None).to(device)
# model_Alexnet.classifier[1] = nn.Conv2d(512, output_dim, kernel_size=(1, 1), stride=(1, 1))
# model_lists.append(["Squeezenet", model_Squeezenet])
# optimizers.append(torch.optim.Adam(model_Squeezenet.parameters(), lr=0.003, weight_decay=1e-5))
    # print all models
model_num = len(model_lists)
for index, model_list in enumerate(model_lists):
    print(f"mode {index+1}: {model_list[0]}")

# %% [markdown]
# ### Dataloader

# %% [code] {"execution":{"iopub.status.busy":"2023-03-20T03:01:23.996646Z","iopub.execute_input":"2023-03-20T03:01:23.997018Z","iopub.status.idle":"2023-03-20T03:01:24.515874Z","shell.execute_reply.started":"2023-03-20T03:01:23.996978Z","shell.execute_reply":"2023-03-20T03:01:24.514627Z"}}

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.

# trainPath = "/kaggle/input/ml2023spring-hw3/train"  # or "./train" 
# validPath = "/kaggle/input/ml2023spring-hw3/valid" # or "./valid"
# testPath = "/kaggle/input/ml2023spring-hw3/test" # or "./test"

trainPath = "./train"
validPath = "./valid"
testPath = "./test"
modelPath = [(f"{_exp_name}_model{index + 1}_best.ckpt") for index in range(model_num)]

trainfiles = sorted([os.path.join(trainPath,x) for x in os.listdir(trainPath) if x.endswith(".jpg")])
validFiles = sorted([os.path.join(validPath,x) for x in os.listdir(validPath) if x.endswith(".jpg")])
testFiles = sorted([os.path.join(testPath,x) for x in os.listdir(testPath) if x.endswith(".jpg")])
train_set = FoodDataset(trainfiles, tfm=train_tfm)
valid_set = FoodDataset(validFiles, tfm=test_tfm)
# train_valid_files = trainfiles + validFiles
train_loaders = []
valid_loaders = []
for _ in range(model_num):
    # train_data_num = math.floor(len(train_valid_files)*train_valid_split)
    # random.shuffle(train_valid_files)
    # train_data = train_valid_files[ : train_data_num ]
    # valid_data = train_valid_files[ train_data_num : ]
    # train_set = FoodDataset(trainfiles, tfm=train_tfm)
    train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))
    # valid_set = FoodDataset(validFiles, tfm=test_tfm)
    valid_loaders.append(DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))



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
print(f"[HW3 parameters] : epoch={n_epochs}\t batch={batch_size}\t label_smoothing={label_smoothing}\t lr={lr}\t TTA={TTA_ratio}\t threshold={threshold}")
model_lists, optimizers, schedules, train_loaders, valid_loaders = accelerator.prepare(
    model_lists, optimizers, schedules, train_loaders, valid_loaders
)
if retrain_model:
    for epoch in range(n_epochs):
        # These are used to record cerain train_loader information in training.
        index = 0
        assert len(train_loaders) == len(valid_loaders)
        for model_list, optimizer, schedule, train_loader, valid_loader in zip(model_lists, optimizers, schedules, train_loaders, valid_loaders):

            model= model_list[1]
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()
            # These are used to record cerain train_loader information in training.
            train_loss = []
            train_accs = []
            for batch in tqdm(train_loader) :  
                with accelerator.accumulate(model):
                    # A batch consists of image data and corresponding labels.
                    imgs, labels = batch
                    #imgs = imgs.half()
                    #print(imgs.shape,labels.shape)

                    # Forward the data. (Make sure data and model are on the same device.)
                    logits = model(imgs.to(device))

                    # Calculate the cross-entropy loss.
                    # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                    loss = criterion(logits, labels.to(device))

                    # use gradient accumulate for more GPU memory
                    loss = loss / gradient_accumulation_steps
                    accelerator.backward(loss)

                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                    optimizer.step()
                    optimizer.zero_grad()
                    # Gradients stored in the parameters in the previous step should be cleared out first.
                    # optimizer.zero_grad()

                    # Compute the gradients for parameters.
                    # print(f"before backward: \t{torch.cuda.memory_allocated()}")
                    # loss.backward(retain_graph=True)

                    # Clip the gradient norms for stable training.
                    # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                    # Update the parameters with computed gradients.
                    # optimizer.step()

                    # Compute the accuracy for current batch.
                    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                    # Record the loss and accuracy.
                    train_loss.append(loss.item())
                    train_accs.append(acc)
            
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            print(f"[ Train | epoch {epoch + 1:03d}/{n_epochs:03d} , model {index + 1:03d}/{model_num:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

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

            # change lr if model not improving
            schedule.step(valid_loss)

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
            index = index + 1

# %% [markdown]
# ### Dataloader for test

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:40.502636Z","iopub.status.idle":"2023-03-20T03:01:40.503411Z","shell.execute_reply.started":"2023-03-20T03:01:40.503145Z","shell.execute_reply":"2023-03-20T03:01:40.503173Z"}}
# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_loaders = []
test_loader_num = 2
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
model_weight = [(best_acc/sum(best_accs)) for best_acc in best_accs]
# model_weight = [1, 1]
print("predict testing data...")
for index in range(model_num):
    model_best = model_lists[index][1]
    model_best.load_state_dict(torch.load(f"{_exp_name}_model{index + 1}_best.ckpt"))
    print(f"model {index+1} ({model_lists[index][0]}) loaded")
    model_bests.append(model_best)

model_ensemble = Ensemble(nn.ModuleList(model_bests), model_num, model_weight).to(device)
model_ensemble.eval()
prediction = []
with torch.no_grad():
    for index, test_loader in enumerate(test_loaders):
        one_prediction = []
        for data, _ in tqdm(test_loader):
            test_pred = model_ensemble(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            one_prediction = one_prediction + test_label.squeeze().tolist()
        if index == 0:
            prediction = [ ( TTA_ratio * prediction ) for prediction in one_prediction ]
        else:
            assert len(one_prediction) == len(prediction)
            prediction = [ ( prediction[index] + (1-TTA_ratio) * one_prediction[index] ) for index in range(len(one_prediction))]
        print(len(prediction))

prediction = [round(pred) for pred in prediction]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:40.507032Z","iopub.status.idle":"2023-03-20T03:01:40.507795Z","shell.execute_reply.started":"2023-03-20T03:01:40.507533Z","shell.execute_reply":"2023-03-20T03:01:40.507560Z"}}
#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)
print("csv file saved")