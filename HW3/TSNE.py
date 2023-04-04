import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.609337Z","iopub.execute_input":"2023-03-20T03:01:21.610227Z","iopub.status.idle":"2023-03-20T03:01:21.618891Z","shell.execute_reply.started":"2023-03-20T03:01:21.610180Z","shell.execute_reply":"2023-03-20T03:01:21.617836Z"}}
_exp_name = "RESNET"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-20T03:01:21.620915Z","iopub.execute_input":"2023-03-20T03:01:21.621831Z","iopub.status.idle":"2023-03-20T03:01:21.633641Z","shell.execute_reply.started":"2023-03-20T03:01:21.621787Z","shell.execute_reply":"2023-03-20T03:01:21.632544Z"}}
# Import necessary packages.
import math
import pandas as pd
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision import models
import random
from sklearn.model_selection import KFold
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

# The number of batch size.
batch_size = 8

# The number of training epochs.
n_epochs = 250

# If no improvement in 'patience' epochs, lower learning rate.
patience = 5

# different train valid split per model
split_num = 5 

#TTA ratio
TTA_ratio = 0.8

#reload model
reload_model=True

#retrain model
retrain_model = False

# For the classification task, we use cross-entropy as the measurement of performance.
label_smoothing = 0.5
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# Initialize model, optimizer and scheduler.

output_dim = 11

# Load the trained model
model_RESNET = models.resnet50(weights=None).to(device)
model_RESNET.fc = nn.Linear(2048, output_dim).to(device)
model = model_RESNET
state_dict = torch.load(f"{_exp_name}_model4_best.ckpt")
model.load_state_dict(state_dict)
model.eval()

print(model)

trainPath = "./train"
validPath = "./valid"
testPath = "./test"

trainfiles = sorted([os.path.join(trainPath,x) for x in os.listdir(trainPath) if x.endswith(".jpg")])
validFiles = sorted([os.path.join(validPath,x) for x in os.listdir(validPath) if x.endswith(".jpg")])
testFiles = sorted([os.path.join(testPath,x) for x in os.listdir(testPath) if x.endswith(".jpg")])
train_set = FoodDataset(trainfiles, tfm=train_tfm)
valid_set = FoodDataset(validFiles, tfm=test_tfm)
train_valid_files = trainfiles + validFiles

kf = KFold(n_splits=split_num, shuffle=True)

train_loaders = []
valid_loaders = []
temp = []
for _ in range(1):
    for (train_index, valid_index) in kf.split(train_valid_files):  
        train_data = (np.array(train_valid_files)[train_index]).tolist()
        valid_data = (np.array(train_valid_files)[valid_index]).tolist()
        temp += valid_data
        train_set = FoodDataset(train_data, tfm=train_tfm)
        train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))
        valid_set = FoodDataset(valid_data, tfm=test_tfm)
        valid_loaders.append(DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True))

# %% [code] {"id":"QcBKUNfc3BeL"}
# Load the vaildation set defined by TA
valid_loader = valid_loaders[3]

# Extract the representations for the specific layer of model
index = 1 # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
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
plt.savefig("tmp.png")
plt.show()