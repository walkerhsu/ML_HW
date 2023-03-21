# -*- coding: utf-8 -*-
"""「ML2023Spring - HW2」的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B88GWfqxb1OG7ZZMDY1V7LNWpw8sGsHb

# **Homework 2: Phoneme Classification**

Objectives:
* Solve a classification problem with deep neural networks (DNNs).
* Understand recursive neural networks (RNNs).

If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# Download Data
Download data from google drive, then unzip it.

You should have
- `libriphone/train_split.txt`: training metadata
- `libriphone/train_labels`: training labels
- `libriphone/test_split.txt`: testing metadata
- `libriphone/feat/train/*.pt`: training feature
- `libriphone/feat/test/*.pt`:  testing feature

after running the following block.

> **Notes: if the google drive link is dead, you can download the data directly from [Kaggle](https://www.kaggle.com/c/ml2023spring-hw2/data) and upload it to the workspace.**
"""

# !pip install --upgrade gdown

# Main link
# !gdown --id '1N1eVIDe9hKM5uiNRGmifBlwSDGiVXPJe' --output libriphone.zip
# !gdown --id '1qzCRnywKh30mTbWUEjXuNT2isOCAPdO1' --output libriphone.zip

# !unzip -q libriphone.zip
# !ls libriphone

"""# Some Utility Functions
**Fixes random number generator seeds for reproducibility.**
"""

import numpy as np
import torch
import random

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# %% [markdown] {"id":"_L_4anls8Drv"}
# **Helper functions to pre-process the training data from raw MFCC features of each utterance.**
# 
# A phoneme may span several frames and is dependent to past and future frames. \
# Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.
# 
# Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)

# %% [code] {"id":"IJjLT8em-y9G","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:50:41.812712Z","iopub.execute_input":"2023-03-16T14:50:41.813793Z","iopub.status.idle":"2023-03-16T14:50:41.843848Z","shell.execute_reply.started":"2023-03-16T14:50:41.813748Z","shell.execute_reply":"2023-03-16T14:50:41.842434Z"}}
import os
import torch
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8):
    class_num = 41 # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]
        
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
      print(y.shape)
      return X, y
    else:
      return X

# %% [markdown] {"id":"us5XW_x6udZQ"}
# # Dataset

# %% [code] {"id":"Fjf5EcmJtf4e","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:50:41.850799Z","iopub.execute_input":"2023-03-16T14:50:41.853786Z","iopub.status.idle":"2023-03-16T14:50:41.865139Z","shell.execute_reply.started":"2023-03-16T14:50:41.853748Z","shell.execute_reply":"2023-03-16T14:50:41.863907Z"}}
import torch
from torch.utils.data import Dataset

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

# %% [markdown] {"id":"IRqKNvNZwe3V"}
# # Model
# Feel free to modify the structure of the model.

# %% [code] {"id":"Bg-GRd7ywdrL","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:54:43.195334Z","iopub.execute_input":"2023-03-16T14:54:43.195701Z","iopub.status.idle":"2023-03-16T14:54:43.206960Z","shell.execute_reply.started":"2023-03-16T14:54:43.195670Z","shell.execute_reply":"2023-03-16T14:54:43.205882Z"}}
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout=0.0):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
#         DNN
#         self.fc = nn.Sequential(
#             BasicBlock(input_dim, hidden_dim),
#             *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
#             nn.Linear(hidden_dim, output_dim)
#         )
#         RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, hidden_layers, batch_first=True)
#         LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
    def forward(self, x):
#         DNN
#         x = self.fc(x)
#         return x
#         RNN
#         print(x.shape)
#         print(h0.shape)
        out, (h_m, _) = self.lstm(x)
        out = self.fc(out)
        
#         print(out)
        # out: batchSize, seq_length, hiddenSize, 
        return out

# %% [markdown] {"id":"TlIq8JeqvvHC"}
# # Hyper-parameters

# %% [code] {"id":"iIHn79Iav1ri","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:50:41.897473Z","iopub.execute_input":"2023-03-16T14:50:41.900395Z","iopub.status.idle":"2023-03-16T14:50:41.908938Z","shell.execute_reply.started":"2023-03-16T14:50:41.900352Z","shell.execute_reply":"2023-03-16T14:50:41.907881Z"}}
# data prarameters
concat_nframes = 31              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 10901036                        # random seed
batch_size = 256                # batch size
num_epoch = 10                   # the number of training epoch
learning_rate = 5e-4         # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 4               # the number of hidden layers
hidden_dim = 1500                # the hidden dim

reload_model = True        # reload model to do further epoch training   
dropout = 0.3              # dropout rate

# %% [markdown] {"id":"IIUFRgG5yoDn"}
# # Dataloader

# %% [code] {"id":"c1zI3v5jyrDn","outputId":"6e7eeb1b-b76a-4846-b9b4-055d66c05661","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:50:41.915095Z","iopub.execute_input":"2023-03-16T14:50:41.918393Z","iopub.status.idle":"2023-03-16T14:51:15.058412Z","shell.execute_reply.started":"2023-03-16T14:50:41.918352Z","shell.execute_reply":"2023-03-16T14:51:15.057281Z"}}
from torch.utils.data import DataLoader
import gc

same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

feature_dir = './libriphone/feat'
libriphone_path = './libriphone'

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir=feature_dir, phone_path=libriphone_path, concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir=feature_dir, phone_path=libriphone_path, concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# %% [markdown] {"id":"pwWH1KIqzxEr"}
# # Training

# %% [code] {"id":"CdMWsBs7zzNs","outputId":"426e0a6c-02bd-4f59-e45c-b05e3f28965d","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:54:46.347493Z","iopub.execute_input":"2023-03-16T14:54:46.347917Z","iopub.status.idle":"2023-03-16T14:54:53.144889Z","shell.execute_reply.started":"2023-03-16T14:54:46.347880Z","shell.execute_reply":"2023-03-16T14:54:53.143347Z"}}
# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, dropout=dropout).to(device)

if reload_model and os.path.exists(model_path):    
    print(f"[train] reload model parameters, model_path:{model_path}")
    model.load_state_dict(torch.load(model_path))
else: 
    print(f"[train] restart with a new model")

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"[HW2 parameters] : epoch={num_epoch}\t batch={batch_size}\t model=LSTM\t lr={learning_rate}\t hidden_layers={hidden_layers}\t hidden_dim={hidden_dim}\t frames={concat_nframes}\t dropout={dropout}")


best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train() # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
#         features = features.reshape(-1, concat_nframes, 39)
#         print(features.shape)
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
#         print(outputs)
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    
    # validation
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            
            loss = criterion(outputs, labels) 
            
            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()

    print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss: {val_loss/len(val_loader):3.5f}')

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')

# %% [code] {"id":"ab33MxosWLmG","outputId":"0d5e2cc6-46f3-41b9-ea09-79fdf660898f","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:51:20.515154Z","iopub.status.idle":"2023-03-16T14:51:20.515921Z","shell.execute_reply.started":"2023-03-16T14:51:20.515595Z","shell.execute_reply":"2023-03-16T14:51:20.515622Z"}}
del train_set, val_set
del train_loader, val_loader
gc.collect()

# %% [markdown] {"id":"1Hi7jTn3PX-m"}
# # Testing
# Create a testing dataset, and load model from the saved checkpoint.

# %% [code] {"id":"VOG1Ou0PGrhc","outputId":"3373d328-bb42-48ec-92f2-e2e935c3344c","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:51:20.517846Z","iopub.status.idle":"2023-03-16T14:51:20.518364Z","shell.execute_reply.started":"2023-03-16T14:51:20.518075Z","shell.execute_reply":"2023-03-16T14:51:20.518097Z"}}
# load data
test_X = preprocess_data(split='test', feat_dir=feature_dir, phone_path=libriphone_path, concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# %% [code] {"id":"ay0Fu8Ovkdad","outputId":"fe130106-a997-4985-fc4b-5102414afe31","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:51:20.520272Z","iopub.status.idle":"2023-03-16T14:51:20.520751Z","shell.execute_reply.started":"2023-03-16T14:51:20.520506Z","shell.execute_reply":"2023-03-16T14:51:20.520529Z"}}
# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

# %% [markdown] {"id":"zp-DV1p4r7Nz"}
# Make prediction.

# %% [code] {"id":"84HU5GGjPqR0","outputId":"b49ffee0-1785-419d-e56c-0ddd734b2c99","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:51:20.522157Z","iopub.status.idle":"2023-03-16T14:51:20.523111Z","shell.execute_reply.started":"2023-03-16T14:51:20.522850Z","shell.execute_reply":"2023-03-16T14:51:20.522875Z"}}
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# %% [markdown] {"id":"wyZqy40Prz0v"}
# Write prediction to a CSV file.
# 
# After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.

# %% [code] {"id":"GuljYSPHcZir","vscode":{"languageId":"python"},"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-16T14:51:20.524803Z","iopub.status.idle":"2023-03-16T14:51:20.525653Z","shell.execute_reply.started":"2023-03-16T14:51:20.525388Z","shell.execute_reply":"2023-03-16T14:51:20.525426Z"}}
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n') 
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))