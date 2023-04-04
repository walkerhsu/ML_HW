# %% [markdown] {"id":"C_jdZ5vHJ4A9"}
# # Task description
# - Classify the speakers of given features.
# - Main goal: Learn how to use transformer.
# - Baselines:
#   - Easy: Run sample code and know how to use transformer.
#   - Medium: Know how to adjust parameters of transformer.
#   - Strong: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer. 
#   - Boss: Implement [Self-Attention Pooling](https://arxiv.org/pdf/2008.01077v1.pdf) & [Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf) to further boost the performance.
# 
# - Other links
#   - Competiton: [link](https://www.kaggle.com/t/49ea0c385a974db5919ec67299ba2e6b)
#   - Slide: [link](https://docs.google.com/presentation/d/1LDAW0GGrC9B6D7dlNdYzQL6D60-iKgFr/edit?usp=sharing&ouid=104280564485377739218&rtpof=true&sd=true)
#   - Data: [link](https://github.com/googly-mingto/ML2023HW4/releases)
#   - ConformerBlock: [link](https://github.com/lucidrains/conformer)

# %% [markdown]
# ### Install needed package

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:31:51.408131Z","iopub.execute_input":"2023-04-03T03:31:51.408718Z","iopub.status.idle":"2023-04-03T03:32:02.865528Z","shell.execute_reply.started":"2023-04-03T03:31:51.408631Z","shell.execute_reply":"2023-04-03T03:32:02.863994Z"}}
# !pip install conformer

# %% [code] {"id":"E6burzCXIyuA","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.868741Z","iopub.execute_input":"2023-04-03T03:32:02.869189Z","iopub.status.idle":"2023-04-03T03:32:02.879910Z","shell.execute_reply.started":"2023-04-03T03:32:02.869147Z","shell.execute_reply":"2023-04-03T03:32:02.878102Z"}}
import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(87)

# %% [markdown] {"id":"k7dVbxW2LASN"}
# # Data
# 
# ## Dataset
# - Original dataset is [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
# - The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb2.
# - We randomly select 600 speakers from Voxceleb2.
# - Then preprocess the raw waveforms into mel-spectrograms.
# 
# - Args:
#   - data_dir: The path to the data directory.
#   - metadata_path: The path to the metadata.
#   - segment_len: The length of audio segment for training. 
# - The architecture of data directory \\
#   - data directory \\
#   |---- metadata.json \\
#   |---- testdata.json \\
#   |---- mapping.json \\
#   |---- uttr-{random string}.pt \\
# 
# - The information in metadata
#   - "n_mels": The dimention of mel-spectrogram.
#   - "speakers": A dictionary. 
#     - Key: speaker ids.
#     - value: "feature_path" and "mel_len"
# 
# 
# For efficiency, we segment the mel-spectrograms into segments in the traing step.

# %% [code] {"id":"KpuGxl4CI2pr","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.881950Z","iopub.execute_input":"2023-04-03T03:32:02.882410Z","iopub.status.idle":"2023-04-03T03:32:02.899512Z","shell.execute_reply.started":"2023-04-03T03:32:02.882371Z","shell.execute_reply":"2023-04-03T03:32:02.897987Z"}}
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
 
 
class myDataset(Dataset):
	def __init__(self, data_dir, segment_len=128):
		self.data_dir = data_dir
		self.segment_len = segment_len
	
		# Load the mapping from speaker neme to their corresponding id. 
		mapping_path = Path(data_dir) / "mapping.json"
		mapping = json.load(mapping_path.open())
		self.speaker2id = mapping["speaker2id"]
	
		# Load metadata of training data.
		metadata_path = Path(data_dir) / "metadata.json"
		metadata = json.load(open(metadata_path))["speakers"]
	
		# Get the total number of speaker.
		self.speaker_num = len(metadata.keys())
		self.data = []
		for speaker in metadata.keys():
			for utterances in metadata[speaker]:
				self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
	def __len__(self):
			return len(self.data)
 
	def __getitem__(self, index):
		feat_path, speaker = self.data[index]
		# Load preprocessed mel-spectrogram.
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		# Segmemt mel-spectrogram into "segment_len" frames.
		if len(mel) > self.segment_len:
			# Randomly get the starting point of the segment.
			start = random.randint(0, len(mel) - self.segment_len)
			# Get a segment with "segment_len" frames.
			mel = torch.FloatTensor(mel[start:start+self.segment_len])
		else:
			mel = torch.FloatTensor(mel)
		# Turn the speaker id into long for computing loss later.
		speaker = torch.FloatTensor([speaker]).long()
		return mel, speaker
 
	def get_speaker_number(self):
		return self.speaker_num

# %% [markdown] {"id":"668hverTMlGN"}
# ## Dataloader
# - Split dataset into training dataset(90%) and validation dataset(10%).
# - Create dataloader to iterate the data.

# %% [code] {"id":"B7c2gZYoJDRS","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.901576Z","iopub.execute_input":"2023-04-03T03:32:02.902789Z","iopub.status.idle":"2023-04-03T03:32:02.920489Z","shell.execute_reply.started":"2023-04-03T03:32:02.902720Z","shell.execute_reply":"2023-04-03T03:32:02.919164Z"}}
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
	# Process features within a batch.
	"""Collate a batch of data."""
	mel, speaker = zip(*batch)
	# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
	mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
	# mel: (batch size, length, 40)
	return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
	"""Generate dataloader"""
	dataset = myDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
	# Split dataset into training dataset and validation dataset
	trainlen = int(0.9 * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=n_workers,
		pin_memory=True,
		collate_fn=collate_batch,
	)
	valid_loader = DataLoader(
		validset,
		batch_size=batch_size,
		num_workers=n_workers,
		drop_last=True,
		pin_memory=True,
		collate_fn=collate_batch,
	)

	return train_loader, valid_loader, speaker_num

# %% [markdown] {"id":"5FOSZYxrMqhc"}
# # Model
# - TransformerEncoderLayer:
#   - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
#   - Parameters:
#     - d_model: the number of expected features of the input (required).
# 
#     - nhead: the number of heads of the multiheadattention models (required).
# 
#     - dim_feedforward: the dimension of the feedforward network model (default=2048).
# 
#     - dropout: the dropout value (default=0.1).
# 
#     - activation: the activation function of intermediate layer, relu or gelu (default=relu).
# 
# - TransformerEncoder:
#   - TransformerEncoder is a stack of N transformer encoder layers
#   - Parameters:
#     - encoder_layer: an instance of the TransformerEncoderLayer() class (required).
# 
#     - num_layers: the number of sub-encoder-layers in the encoder (required).
# 
#     - norm: the layer normalization component (optional).

# %% [code] {"id":"iXZ5B0EKJGs8","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.923613Z","iopub.execute_input":"2023-04-03T03:32:02.924006Z","iopub.status.idle":"2023-04-03T03:32:02.939863Z","shell.execute_reply.started":"2023-04-03T03:32:02.923969Z","shell.execute_reply":"2023-04-03T03:32:02.938949Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock

class Classifier(nn.Module):
	def __init__(self, d_model=200, n_spks=600, dim_feedforward=512, nhead=8, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer_transformer = nn.TransformerEncoderLayer(
			d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout
		)
		self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer_transformer, num_layers=1)
		self.encoder_layer_conformer = ConformerBlock(
			dim=d_model,
			dim_head=dim_feedforward,
			heads=nhead,  
			ff_mult=4,
			conv_expansion_factor=4,
			conv_kernel_size=32,
			attn_dropout=dropout,
			ff_dropout=dropout,             
			conv_dropout=dropout
		)
		self.W = nn.Linear(d_model, 1)
		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model*2),
			nn.BatchNorm1d(d_model*2),
			nn.Dropout(dropout),
			nn.Sigmoid(),
			nn.Linear(d_model*2, n_spks),
		)
		self.softmax = F.softmax

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# out: (batch size, length,  d_model)
		out = self.encoder_transformer(out)
		# out: (length, batch size, d_model)
		# out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer_conformer(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		# stats = out.mean(dim=1)
		# self_attention pooling
		att_w = self.softmax(self.W(out).squeeze(-1), dim=1).unsqueeze(-1)
		stats = torch.sum(out * att_w, dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out

# %% [markdown] {"id":"W7yX8JinM5Ly"}
# # Learning rate schedule
# - For transformer architecture, the design of learning rate schedule is different from that of CNN.
# - Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
# - The warmup schedule
#   - Set learning rate to 0 in the beginning.
#   - The learning rate increases linearly from 0 to initial learning rate during warmup period.

# %% [code] {"id":"ykt0N1nVJJi2","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.941119Z","iopub.execute_input":"2023-04-03T03:32:02.942213Z","iopub.status.idle":"2023-04-03T03:32:02.958784Z","shell.execute_reply.started":"2023-04-03T03:32:02.942174Z","shell.execute_reply":"2023-04-03T03:32:02.957822Z"}}
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

# %% [markdown] {"id":"-LN2XkteM_uH"}
# # Model Function
# - Model forward function.

# %% [code] {"id":"N-rr8529JMz0","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.959969Z","iopub.execute_input":"2023-04-03T03:32:02.960405Z","iopub.status.idle":"2023-04-03T03:32:02.976761Z","shell.execute_reply.started":"2023-04-03T03:32:02.960360Z","shell.execute_reply":"2023-04-03T03:32:02.975273Z"}}
import torch


def model_fn(batch, model, criterion, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs = model(mels)

	loss = criterion(outs, labels)

	# Get the speaker id with highest probability.
	preds = outs.argmax(1)
	# Compute accuracy.
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

# %% [markdown] {"id":"cwM_xyOtNCI2"}
# # Validate
# - Calculate accuracy of the validation set.

# %% [code] {"id":"YAiv6kpdJRTJ","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.978836Z","iopub.execute_input":"2023-04-03T03:32:02.979250Z","iopub.status.idle":"2023-04-03T03:32:02.992115Z","shell.execute_reply.started":"2023-04-03T03:32:02.979215Z","shell.execute_reply":"2023-04-03T03:32:02.990894Z"}}
from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update(dataloader.batch_size)
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)

	pbar.close()
	model.train()

	return running_accuracy / len(dataloader)

# %% [markdown] {"id":"g6ne9G-eNEdG"}
# # Main function

# %% [code] {"id":"Usv9s-CuJSG7","outputId":"f4f6a983-3559-4f36-efae-402bbf790473","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:02.993829Z","iopub.execute_input":"2023-04-03T03:32:02.994345Z","iopub.status.idle":"2023-04-03T03:32:04.100321Z","shell.execute_reply.started":"2023-04-03T03:32:02.994294Z","shell.execute_reply":"2023-04-03T03:32:04.092262Z"}}
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"save_path": "./model.ckpt",
		"batch_size":16,
		"n_workers": 8,
		"valid_steps": 2000,
		"warmup_steps": 1500,
		"save_steps": 10000,
		"total_steps": 300000,
		"reload": True,
	}
	print(f"[HW4 parameter]: {config}")
	return config 


def main(
	data_dir,
	save_path,
	batch_size,
	n_workers,
	valid_steps,
	warmup_steps,
	total_steps,
	save_steps,
	reload,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                
	print(f"[Info]: Use {device} now!")

	train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
	train_iterator = iter(train_loader)
	print(f"[Info]: Finish loading data!",flush = True)

	model = Classifier(n_spks=speaker_num).to(device)
	if reload and os.path.exists(save_path):
		model.load_state_dict(torch.load(save_path))         
		print(f"[Info]: reload model!",flush = True)
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=1e-3)
	scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	print(f"[Info]: Finish creating model!",flush = True)

	best_accuracy = -1.0
	best_state_dict = None
                     
	pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")                                   

	for step in range(total_steps):
		# Get data
		try:
			batch = next(train_iterator)
		except StopIteration:                                                             
			train_iterator = iter(train_loader)
			batch = next(train_iterator)
                  
		loss, accuracy = model_fn(batch, model, criterion, device)
		batch_loss = loss.item()
		batch_accuracy = accuracy.item()

		# Updata model
		loss.backward()
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		# Log
		pbar.update()
		pbar.set_postfix(
			loss=f"{batch_loss:.2f}",
			accuracy=f"{batch_accuracy:.2f}",
			step=step + 1,
		)

		# Do validation
		if (step + 1) % valid_steps == 0:
			pbar.close()

			valid_accuracy = valid(valid_loader, model, criterion, device)

			# keep the best model
			if valid_accuracy > best_accuracy:
				best_accuracy = valid_accuracy
				best_state_dict = model.state_dict()

			pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

		# Save the best model so far.
		if (step + 1) % save_steps == 0 and best_state_dict is not None:
			torch.save(best_state_dict, save_path)
			pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

	pbar.close()


if __name__ == "__main__":
	main(**parse_args())

# %% [markdown] {"id":"NLatBYAhNNMx"}
# # Inference
# 
# ## Dataset of inference

# %% [code] {"id":"efS4pCmAJXJH","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:04.106478Z","iopub.status.idle":"2023-04-03T03:32:04.111645Z","shell.execute_reply.started":"2023-04-03T03:32:04.111183Z","shell.execute_reply":"2023-04-03T03:32:04.111256Z"}}
import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)

# %% [markdown] {"id":"tl0WnYwxNK_S"}
# ## Main function of Inference

# %% [code] {"id":"i8SAbuXEJb2A","outputId":"3808f409-19c9-426c-dc15-1b88b0c21645","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-04-03T03:32:04.113736Z","iopub.status.idle":"2023-04-03T03:32:04.120395Z","shell.execute_reply.started":"2023-04-03T03:32:04.120019Z","shell.execute_reply":"2023-04-03T03:32:04.120075Z"}}
import json
import csv
from pathlib import Path
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader

def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"model_path": "./model.ckpt",
		"output_path": "./output.csv",
	}

	return config


def main(
	data_dir,
	model_path,
	output_path,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		drop_last=False,
		num_workers=8,
		collate_fn=inference_collate_batch,
	)
	print(f"[Info]: Finish loading data!",flush = True)

	speaker_num = len(mapping["id2speaker"])
	model = Classifier(n_spks=speaker_num).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(device)
			outs = model(mels)
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)


if __name__ == "__main__":
	main(**parse_args())