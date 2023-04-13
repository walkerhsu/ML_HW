# %% [markdown] {"id":"AFEKWoh3p1Mv"}
# # Homework Description
# - English to Chinese (Traditional) Translation
#   - Input: an English sentence         (e.g.		tom is a student .)
#   - Output: the Chinese translation  (e.g. 		湯姆 是 個 學生 。)
# 
# - TODO
#     - Train a simple RNN seq2seq to acheive translation
#     - Switch to transformer model to boost performance
#     - Apply Back-translation to furthur boost performance

# # %% [code] {"id":"3Vf1Q79XPQ3D","jupyter":{"outputs_hidden":false}}
# !nvidia-smi

# %% [markdown] {"id":"59neB_Sxp5Ub"}
# # Download and import required packages

# # %% [code] {"id":"rRlFbfFRpZYT","jupyter":{"outputs_hidden":false}}
# !pip install 'torch>=1.6.0' editdistance matplotlib sacrebleu sacremoses sentencepiece tqdm wandb
# !pip install --upgrade jupyter ipywidgets

# # %% [code] {"id":"fSksMTdmp-Wt","jupyter":{"outputs_hidden":false}}
# !git clone https://github.com/pytorch/fairseq.git
# !cd fairseq && git checkout 9a1c497
# !pip install --upgrade ./fairseq/

# %% [code] {"id":"uRLTiuIuqGNc","jupyter":{"outputs_hidden":false}}
import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils

import matplotlib.pyplot as plt

# %% [markdown] {"id":"0n07Za1XqJzA"}
# # Fix random seed

# %% [code] {"id":"xllxxyWxqI7s","jupyter":{"outputs_hidden":false}}
seed = 33
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% [markdown] {"id":"N5ORDJ-2qdYw"}
# # Dataset
# 
# ## En-Zh Bilingual Parallel Corpus
# * TED2020
#     - Raw: 400,726 (sentences)   
#     - Processed: 394,052 (sentences)
#     
# 
# ## Testdata
# - Size: 4,000 (sentences)
# - **Chinese translation is undisclosed. The provided (.zh) file is psuedo translation, each line is a '。'**

# %% [markdown] {"id":"GQw2mY4Dqkzd"}
# ## Dataset Download

# %% [code] {"id":"SXT42xQtqijD","jupyter":{"outputs_hidden":false}}
data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.data.tgz",
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.test.tgz"
)
file_names = (
    'ted2020.tgz', # train & dev
    'test.tgz', # test
)
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix/f
    if not path.exists():
        os.system(f"wget {u} -O {path}")
    if path.suffix == ".tgz":
        os.system(f"tar -xvf {path} -C {prefix}")
    elif path.suffix == ".zip":
        os.system(f"unzip -o {path} -d {prefix}")
os.system(f"mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}")
os.system(f"mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}")
os.system(f"mv {prefix/'test.en'} {prefix/'test.raw.en'}")
os.system(f"mv {prefix/'test.zh'} {prefix/'test.raw.zh'}")

# %% [markdown] {"id":"YLkJwNiFrIwZ"}
# ## Language

# %% [code] {"id":"_uJYkCncrKJb","jupyter":{"outputs_hidden":false}}
src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'

# %% [code] {"id":"0t2CPt1brOT3","jupyter":{"outputs_hidden":false}}
os.system(f"head {data_prefix+'.'+src_lang} -n 5")
os.system(f"head {data_prefix+'.'+tgt_lang} -n 5")

# %% [markdown] {"id":"pRoE9UK7r1gY"}
# ## Preprocess files

# %% [code] {"id":"3tzFwtnFrle3","jupyter":{"outputs_hidden":false}}
import re

def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)
                
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace('-', '') # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s) # Q2B
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # remove by ratio of length
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

# %% [code] {"id":"h_i8b1PRr9Nf","jupyter":{"outputs_hidden":false}}
clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

# %% [code] {"id":"gjT3XCy9r_rj","jupyter":{"outputs_hidden":false}}
os.system(f"head {data_prefix+'.clean.'+src_lang} -n 5")
os.system(f"head {data_prefix+'.clean.'+tgt_lang} -n 5")

# %% [markdown] {"id":"nKb4u67-sT_Z"}
# ## Split into train/valid

# %% [code] {"id":"AuFKeDz3sGHL","jupyter":{"outputs_hidden":false}}
valid_ratio = 0.01 # 3000~4000 would suffice
train_ratio = 1 - valid_ratio

# %% [code] {"id":"QR2NVldqsXyY","jupyter":{"outputs_hidden":false}}
if (prefix/f'train.clean.{src_lang}').exists() \
and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() \
and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        count = 0
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

# %% [markdown] {"id":"n1rwQysTsdJq"}
# ## Subword Units 
# Out of vocabulary (OOV) has been a major problem in machine translation. This can be alleviated by using subword units.
# - We will use the [sentencepiece](#kudo-richardson-2018-sentencepiece) package
# - select 'unigram' or 'byte-pair encoding (BPE)' algorithm

# %% [code] {"id":"Ecwllsa7sZRA","jupyter":{"outputs_hidden":false}}
import sentencepiece as spm
vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

# %% [code] {"id":"lQPRNldqse_V","jupyter":{"outputs_hidden":false}}
spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w') as out_f:
                with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)

# %% [code] {"id":"4j6lXHjAsjXa","jupyter":{"outputs_hidden":false}}
os.system(f"head {data_dir+'/'+dataset_name+'/train.'+src_lang} -n 5")
os.system(f"head {data_dir+'/'+dataset_name+'/train.'+tgt_lang} -n 5")

# %% [markdown] {"id":"59si_C0Wsms7"}
# ## Binarize the data with fairseq
# Prepare the files in pairs for both the source and target languages. \\
# In case a pair is unavailable, generate a pseudo pair to facilitate binarization.

# %% [code] {"id":"w-cHVLSpsknh","jupyter":{"outputs_hidden":false}}
binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    os.system(f"python -m fairseq_cli.preprocess \
        --source-lang {src_lang}\
        --target-lang {tgt_lang}\
        --trainpref {prefix/'train'}\
        --validpref {prefix/'valid'}\
        --testpref {prefix/'test'}\
        --destdir {binpath}\
        --joined-dictionary\
        --workers 2")

# %% [markdown] {"id":"szMuH1SWLPWA"}
# # Configuration for experiments

# %% [code] {"id":"5Luz3_tVLUxs","jupyter":{"outputs_hidden":false}}
config = Namespace(
    datadir = "./DATA/data-bin/ted2020",
    savedir = "./checkpoints/transformer",
    source_lang = src_lang,
    target_lang = tgt_lang,
    
    # cpu threads when fetching & processing data.
    num_workers=2,  
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,
    
    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,
    
    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,
    
    # beam size for beam search
    beam=5, 
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2, 
    max_len_b=10, 
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume="checkpoint_best.pt", # if resume from checkpoint name (under config.savedir)
    retrain = True,
    # logging
    use_wandb=False,
)

# %% [markdown] {"id":"cjrJFvyQLg86"}
# # Logging
# - logging package logs ordinary messages
# - wandb logs the loss, bleu, etc. in the training process

# %% [code] {"id":"-ZiMyDWALbDk","jupyter":{"outputs_hidden":false}}
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

# %% [markdown] {"id":"BNoSkK45Lmqc"}
# # CUDA Environments

# %% [code] {"id":"oqrsbmcoLqMl","jupyter":{"outputs_hidden":false}}
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %% [markdown] {"id":"TbJuBIHLLt2D"}
# # Dataloading

# %% [markdown] {"id":"oOpG4EBRLwe_"}
# ## We borrow the TranslationTask from fairseq
# * used to load the binarized data created above
# * well-implemented data iterator (dataloader)
# * built-in task.source_dictionary and task.target_dictionary are also handy
# * well-implemented beach search decoder

# %% [code] {"id":"3gSEy1uFLvVs","jupyter":{"outputs_hidden":false}}
from fairseq.tasks.translation import TranslationConfig, TranslationTask

## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)

# %% [code] {"id":"mR7Bhov7L4IU","jupyter":{"outputs_hidden":false}}
logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

# %% [code] {"id":"P0BCEm_9L6ig","jupyter":{"outputs_hidden":false}}
sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint(
    "Source: " + \
    task.source_dictionary.string(
        sample['source'],
        config.post_process,
    )
)
pprint.pprint(
    "Target: " + \
    task.target_dictionary.string(
        sample['target'],
        config.post_process,
    )
)

# %% [markdown] {"id":"UcfCVa2FMBSE"}
# # Dataset iterator

# %% [markdown] {"id":"yBvc-B_6MKZM"}
# * Controls every batch to contain no more than N tokens, which optimizes GPU memory efficiency
# * Shuffles the training set for every epoch
# * Ignore sentences exceeding maximum length
# * Pad all sentences in a batch to the same length, which enables parallel computing by GPU
# * Add eos and shift one token
#     - teacher forcing: to train the model to predict the next token based on prefix, we feed the right shifted target sequence as the decoder input.
#     - generally, prepending bos to the target would do the job (as shown below)
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
#     - in fairseq however, this is done by moving the eos token to the begining. Empirically, this has the same effect. For instance:
#     ```
#     # output target (target) and Decoder input (prev_output_tokens): 
#                    eos = 2
#                 target = 419,  711,  238,  888,  792,   60,  968,    8,    2
#     prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
#     ```

# %% [code] {"id":"OWFJFmCnMDXW","jupyter":{"outputs_hidden":false}}
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond 
        # first call of this method has no effect. 
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
sample

# %% [markdown] {"id":"p86K-0g7Me4M"}
# * each batch is a python dict, with string key and Tensor value. Contents are described below:
# ```python
# batch = {
#     "id": id, # id for each example 
#     "nsentences": len(samples), # batch size (sentences)
#     "ntokens": ntokens, # batch size (tokens)
#     "net_input": {
#         "src_tokens": src_tokens, # sequence in source language
#         "src_lengths": src_lengths, # sequence length of each example before padding
#         "prev_output_tokens": prev_output_tokens, # right shifted target, as mentioned above.
#     },
#     "target": target, # target sequence
# }
# ```

# %% [markdown] {"id":"9EyDBE5ZMkFZ"}
# # Model Architecture
# * We again inherit fairseq's encoder, decoder and model, so that in the testing phase we can directly leverage fairseq's beam search decoder.

# %% [code] {"id":"Hzh74qLIMfW_","jupyter":{"outputs_hidden":false}}
from fairseq.models import (
    FairseqEncoder, 
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)

# %% [markdown] {"id":"OI46v1z7MotH"}
# # Encoder

# %% [markdown] {"id":"Wn0wSeLLMrbc"}
# - The Encoder is a RNN or Transformer Encoder. The following description is for RNN. For every input token, Encoder will generate a output vector and a hidden states vector, and the hidden states vector is passed on to the next step. In other words, the Encoder sequentially reads in the input sequence, and outputs a single vector at each timestep, then finally outputs the final hidden states, or content vector, at the last timestep.
# - Parameters:
#   - *args*
#       - encoder_embed_dim: the dimension of embeddings, this compresses the one-hot vector into fixed dimensions, which achieves dimension reduction
#       - encoder_ffn_embed_dim is the dimension of hidden states and output vectors
#       - encoder_layers is the number of layers for Encoder RNN
#       - dropout determines the probability of a neuron's activation being set to 0, in order to prevent overfitting. Generally this is applied in training, and removed in testing.
#   - *dictionary*: the dictionary provided by fairseq. it's used to obtain the padding index, and in turn the encoder padding mask. 
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# 
# - Inputs: 
#     - *src_tokens*: integer sequence representing english e.g. 1, 28, 29, 205, 2 
# - Outputs: 
#     - *outputs*: the output of RNN at each timestep, can be furthur processed by Attention
#     - *final_hiddens*: the hidden states of each timestep, will be passed to decoder for decoding
#     - *encoder_padding_mask*: this tells the decoder which position to ignore

# %% [code] {"id":"WcX3W4iGMq-S","jupyter":{"outputs_hidden":false}}
class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=True
        )
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        self.padding_idx = dictionary.pad()
        
    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        bsz, seqlen = src_tokens.size()
        
        # get embeddings
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        # pass thru bidirectional RNN
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim)
        x, final_hiddens = self.rnn(x, h0)
        outputs = self.dropout_out_module(x)
        # outputs = [sequence len, batch size, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        
        # Since Encoder is bidirectional, we need to concatenate the hidden states of two directions
        final_hiddens = self.combine_bidir(final_hiddens, bsz)
        # hidden =  [num_layers x batch x num_directions*hidden]
        
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                outputs,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
            )
        )
    
    def reorder_encoder_out(self, encoder_out, new_order):
        # This is used by fairseq's beam search. How and why is not particularly important here.
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
            )
        )

# %% [markdown] {"id":"6ZlE_1JnMv56"}
# ## Attention

# %% [markdown] {"id":"ZSFSKt_ZMzgh"}
# - When the input sequence is long, "content vector" alone cannot accurately represent the whole sequence, attention mechanism can provide the Decoder more information.
# - According to the **Decoder embeddings** of the current timestep, match the **Encoder outputs** with decoder embeddings to determine correlation, and then sum the Encoder outputs weighted by the correlation as the input to **Decoder** RNN.
# - Common attention implementations use neural network / dot product as the correlation between **query** (decoder embeddings) and **key** (Encoder outputs), followed by **softmax**  to obtain a distribution, and finally **values** (Encoder outputs) is **weighted sum**-ed by said distribution.
# 
# - Parameters:
#   - *input_embed_dim*: dimensionality of key, should be that of the vector in decoder to attend others
#   - *source_embed_dim*: dimensionality of query, should be that of the vector to be attended to (encoder outputs)
#   - *output_embed_dim*: dimensionality of value, should be that of the vector after attention, expected by the next layer
# 
# - Inputs: 
#     - *inputs*: is the key, the vector to attend to others
#     - *encoder_outputs*:  is the query/value, the vector to be attended to
#     - *encoder_padding_mask*: this tells the decoder which position to ignore
# - Outputs: 
#     - *output*: the context vector after attention
#     - *attention score*: the attention distribution

# %% [code] {"id":"1Atf_YuCMyyF","jupyter":{"outputs_hidden":false}}
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
        # inputs: T, B, dim
        # encoder_outputs: S x B x dim
        # padding mask:  S x B
        
        # convert all to batch first
        inputs = inputs.transpose(1,0) # B, T, dim
        encoder_outputs = encoder_outputs.transpose(1,0) # B, S, dim
        encoder_padding_mask = encoder_padding_mask.transpose(1,0) # B, S
        
        # project to the dimensionality of encoder_outputs
        x = self.input_proj(inputs)

        # compute attention
        # (B, T, dim) x (B, dim, S) = (B, T, S)
        attn_scores = torch.bmm(x, encoder_outputs.transpose(1,2))

        # cancel the attention at positions corresponding to padding
        if encoder_padding_mask is not None:
            # leveraging broadcast  B, S -> (B, 1, S)
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        # softmax on the dimension corresponding to source sequence
        attn_scores = F.softmax(attn_scores, dim=-1)

        # shape (B, T, S) x (B, S, dim) = (B, T, dim) weighted sum
        x = torch.bmm(attn_scores, encoder_outputs)

        # (B, T, dim)
        x = torch.cat((x, inputs), dim=-1)
        x = torch.tanh(self.output_proj(x)) # concat + linear + tanh
        
        # restore shape (B, T, dim) -> (T, B, dim)
        return x.transpose(1,0), attn_scores

# %% [markdown] {"id":"doSCOA2gM7fK"}
# # Decoder

# %% [markdown] {"id":"2M8Vod2gNABR"}
# * The hidden states of **Decoder** will be initialized by the final hidden states of **Encoder** (the content vector)
# * At the same time, **Decoder** will change its hidden states based on the input of the current timestep (the outputs of previous timesteps), and generates an output
# * Attention improves the performance
# * The seq2seq steps are implemented in decoder, so that later the Seq2Seq class can accept RNN and Transformer, without furthur modification.
# - Parameters:
#   - *args*
#       - decoder_embed_dim: is the dimensionality of the decoder embeddings, similar to encoder_embed_dim，
#       - decoder_ffn_embed_dim: is the dimensionality of the decoder RNN hidden states, similar to encoder_ffn_embed_dim
#       - decoder_layers: number of layers of RNN decoder
#       - share_decoder_input_output_embed: usually, the projection matrix of the decoder will share weights with the decoder input embeddings
#   - *dictionary*: the dictionary provided by fairseq
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# - Inputs: 
#     - *prev_output_tokens*: integer sequence representing the right-shifted target e.g. 1, 28, 29, 205, 2 
#     - *encoder_out*: encoder's output.
#     - *incremental_state*: in order to speed up decoding during test time, we will save the hidden state of each timestep. see forward() for details.
# - Outputs: 
#     - *outputs*: the logits (before softmax) output of decoder for each timesteps
#     - *extra*: unsused

# %% [code] {"id":"QfvgqHYDM6Lp","jupyter":{"outputs_hidden":false}}
class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        assert args.decoder_layers == args.encoder_layers, f"""seq2seq rnn requires that encoder 
        and decoder have same layers of rnn. got: {args.encoder_layers, args.decoder_layers}"""
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim*2, f"""seq2seq-rnn requires 
        that decoder hidden to be 2*encoder hidden dim. got: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim*2}"""
        
        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers
        
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=False
        )
        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        ) 
        # self.attention = None
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None
        
        if args.share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        # extract the outputs from encoder
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out
        # outputs:          seq_len x batch x num_directions*hidden
        # encoder_hiddens:  num_layers x batch x num_directions*encoder_hidden
        # padding_mask:     seq_len x batch
        
        if incremental_state is not None and len(incremental_state) > 0:
            # if the information from last timestep is retained, we can continue from there instead of starting from bos
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # incremental state does not exist, either this is training time, or the first timestep of test time
            # prepare for seq2seq: pass the encoder_hidden to the decoder hidden states
            prev_hiddens = encoder_hiddens
        
        bsz, seqlen = prev_output_tokens.size()
        
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
                
        # decoder-to-encoder attention
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)
                        
        # pass thru unidirectional RNN
        x, final_hiddens = self.rnn(x, prev_hiddens)
        # outputs = [sequence len, batch size, hid dim]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        x = self.dropout_out_module(x)
                
        # project to embedding size (if hidden differs from embed size, and share_embedding is True, 
        # we need to do an extra projection)
        if self.project_out_dim != None:
            x = self.project_out_dim(x)
        
        # project to vocab size
        x = self.output_projection(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        
        # if incremental, record the hidden states of current timestep, which will be restored in the next timestep
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        
        return x, None
    
    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        # This is used by fairseq's beam search. How and why is not particularly important here.
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prev_hiddens = cache_state["prev_hiddens"]
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return

# %% [markdown] {"id":"UDAPmxjRNEEL"}
# ## Seq2Seq
# - Composed of **Encoder** and **Decoder**
# - Recieves inputs and pass to **Encoder** 
# - Pass the outputs from **Encoder** to **Decoder**
# - **Decoder** will decode according to outputs of previous timesteps as well as **Encoder** outputs  
# - Once done decoding, return the **Decoder** outputs

# %% [code] {"id":"oRwKdLa0NEU6","jupyter":{"outputs_hidden":false}}
class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

# %% [markdown] {"id":"zu3C2JfqNHzk"}
# # Model Initialization

# %% [code] {"id":"nyI9FOx-NJ2m","jupyter":{"outputs_hidden":false}}
# # HINT: transformer architecture
from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)

def build_model(args, task):
    """ build a model instance based on hyperparameters """
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    
    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)
    
    # initialization for seq2seq model is important, requires extra handling
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)
            
    # weight initialization
    model.apply(init_params)
    return model

# %% [markdown] {"id":"ce5n4eS7NQNy"}
# ## Architecture Related Configuration
# 
# For strong baseline, please refer to the hyperparameters for *transformer-base* in Table 3 in [Attention is all you need](#vaswani2017)

# %% [code] {"id":"Cyn30VoGNT6N","jupyter":{"outputs_hidden":false}}

# Transformer
# arch_args = Namespace(
#     encoder_embed_dim=512,
#     encoder_ffn_embed_dim=512,
#     encoder_layers=4,
#     decoder_embed_dim=512,
#     decoder_ffn_embed_dim=2048,
#     decoder_layers=4,
#     share_decoder_input_output_embed=True,
#     dropout=0.1,
# )

# RNN
arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=512,
    encoder_layers=1,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=1,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)

# HINT: these patches on parameters for Transformer
def add_transformer_args(args):
    args.encoder_attention_heads=8
    args.encoder_normalize_before=True
    
    args.decoder_attention_heads=8
    args.decoder_normalize_before=True
    
    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024
    
    # patches on default parameters for Transformer (those not set above)
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

# add_transformer_args(arch_args)

# %% [code] {"id":"Nbb76QLCNZZZ","jupyter":{"outputs_hidden":false}}
if config.use_wandb:
    wandb.config.update(vars(arch_args))

# %% [code] {"id":"7ZWfxsCDNatH","jupyter":{"outputs_hidden":false}}
# from torch.nn.functional import cosine_similarity as cos_sim

model = build_model(arch_args, task)
logger.info(model)

# position = model.decoder.embed_positions.weights.cpu().detach()
# sim = cos_sim(position.unsqueeze(1), position, dim=2)
# plt.figure(figsize=(8,8))
# plt.matshow(sim)
# plt.savefig('similarity.png')

# %% [markdown] {"id":"aHll7GRNNdqc"}
# # Optimization

# %% [markdown] {"id":"rUB9f1WCNgMH"}
# ## Loss: Label Smoothing Regularization
# * let the model learn to generate less concentrated distribution, and prevent over-confidence
# * sometimes the ground truth may not be the only answer. thus, when calculating loss, we reserve some probability for incorrect labels
# * avoids overfitting
# 
# code [source](https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html)

# %% [code] {"id":"IgspdJn0NdYF","jupyter":{"outputs_hidden":false}}
class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，the cross-entropy when target is one-hot. following line is same as F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  reserve some probability for other labels. thus when calculating cross-entropy, 
        # equivalent to summing the log probs of all labels
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # when calculating cross-entropy, add the loss of other labels
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss

# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)

# %% [markdown] {"id":"aRalDto2NkJJ"}
# ## Optimizer: Adam + lr scheduling
# Inverse square root scheduling is important to the stability when training Transformer. It's later used on RNN as well.
# Update the learning rate according to the following equation. Linearly increase the first stage, then decay proportionally to the inverse square root of timestep.
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$

# %% [code] {"id":"sS7tQj1ROBYm","jupyter":{"outputs_hidden":false}}
def get_rate(d_model, step_num, warmup_step):
    # TODO: Change lr from constant to the equation shown above
    lr = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_step ** (-1.5))
    return lr

# %% [code] {"id":"J8hoAjHPNkh3","jupyter":{"outputs_hidden":false}}
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)

# %% [markdown] {"id":"VFJlkOMONsc6"}
# ## Scheduling Visualized

# %% [code] {"id":"A135fwPCNrQs","jupyter":{"outputs_hidden":false}}
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim, 
    factor=config.lr_factor, 
    warmup=config.lr_warmup, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
None

# %% [markdown] {"id":"TOR0g-cVO5ZO"}
# # Training Procedure

# %% [markdown] {"id":"f-0ZjbK3O8Iv"}
# ## Training

# %% [code] {"id":"foal3xM1O404","jupyter":{"outputs_hidden":false}}
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

grad_norms = []
def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps) # gradient accumulation: update every accum_steps samples
    
    stats = {"loss": []}
    scaler = GradScaler() # automatic mixed precision (amp) 
    
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)            
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                
                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0)) # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) # grad norm clipping prevents gradient exploding
        grad_norms.append(gnorm.cpu().item())
        
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats

# %% [markdown] {"id":"Gt1lX3DRO_yU"}
# ## Validation & Inference
# To prevent overfitting, validation is required every epoch to validate the performance on unseen data.
# - the procedure is essensially same as training, with the addition of inference step
# - after validation we can save the model weights
# 
# Validation loss alone cannot describe the actual performance of the model
# - Directly produce translation hypotheses based on current model, then calculate BLEU with the reference translation
# - We can also manually examine the hypotheses' quality
# - We use fairseq's sequence generator for beam search to generate translation hypotheses

# %% [code] {"id":"2og80HYQPAKq","jupyter":{"outputs_hidden":false}}
# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search
sequence_generator = task.build_generator([model], config)

def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs

# %% [code] {"id":"y1o7LeDkPDsd","jupyter":{"outputs_hidden":false}}
import shutil
import sacrebleu

def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # do inference
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])
    
    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

# %% [markdown] {"id":"1sRF6nd4PGEE"}
# # Save and Load Model Weights

# %% [code] {"id":"edBuLlkuPGr9","jupyter":{"outputs_hidden":false}}
def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu    
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")

# %% [markdown] {"id":"KyIFpibfPJ5u"}
# # Main
# ## Training loop

# %% [code] {"id":"hu7RZbCUPKQr","jupyter":{"outputs_hidden":false}}
if config.retrain:
    model = model.to(device=device)
    criterion = criterion.to(device=device)

    # %% [code] {"id":"5xxlJxU2PeAo","jupyter":{"outputs_hidden":false}}
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

    # %% [code] {"id":"MSPRqpQUPfaX","jupyter":{"outputs_hidden":false}}
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    try_load_checkpoint(model, optimizer, name=config.resume)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

    # %% [markdown] {"id":"KyjRwllxPjtf"}
    # # Submission

    # %% [code] {"id":"N70Gc6smPi1d","jupyter":{"outputs_hidden":false}}
    # averaging a few checkpoints can have a similar effect to ensemble
    checkdir=config.savedir
    os.system(f"python ./fairseq/scripts/average_checkpoints.py \
    --inputs {checkdir} \
    --num-epoch-checkpoints 5 \
    --output {checkdir}/avg_last_5_checkpoint.pt")

    # %% [markdown] {"id":"BAGMiun8PnZy"}
    # ## Confirm model weights used to generate submission

    # %% [code] {"id":"tvRdivVUPnsU","jupyter":{"outputs_hidden":false}}
    # checkpoint_last.pt : latest epoch
    # checkpoint_best.pt : highest validation bleu
    # avg_last_5_checkpoint.pt: the average of last 5 epochs
    try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
    validate(model, task, criterion, log_to_wandb=False)
    None

    # %% [markdown] {"id":"ioAIflXpPsxt"}
    # ## Generate Prediction

    # %% [code] {"id":"oYMxA8FlPtIq","jupyter":{"outputs_hidden":false}}
    def generate_prediction(model, task, split="test", outfile="./prediction.txt"):    
        task.load_dataset(split=split, epoch=1)
        itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
        
        idxs = []
        hyps = []

        model.eval()
        progress = tqdm.tqdm(itr, desc=f"prediction")
        with torch.no_grad():
            for i, sample in enumerate(progress):
                # validation loss
                sample = utils.move_to_cuda(sample, device=device)

                # do inference
                s, h, r = inference_step(sample, model)
                
                hyps.extend(h)
                idxs.extend(list(sample['id']))
                
        # sort based on the order before preprocess
        hyps = [x for _,x in sorted(zip(idxs,hyps))]
        
        with open(outfile, "w") as f:
            for h in hyps:
                f.write(h+"\n")

    # %% [code] {"id":"Le4RFWXxjmm0","jupyter":{"outputs_hidden":false}}
    generate_prediction(model, task)

# %% [code] {"id":"wvenyi6BPwnD","jupyter":{"outputs_hidden":false}}
# print(len(grad_norms))
plt.plot(range(1, len(grad_norms)+1), grad_norms)
plt.plot(range(1, len(grad_norms)+1), [config.clip_norm]*len(grad_norms), '-')
plt.xlabel("step")
plt.ylabel("gnorm")
plt.xlim(1, len(grad_norms)+1)
plt.savefig("gnorm_vs_step.png")
raise

# %% [markdown] {"id":"1z0cJE-wPzaU"}
# # Back-translation

# %% [markdown] {"id":"5-7uPJ2CP0sm"}
# ## Train a backward translation model

# %% [markdown] {"id":"ppGHjg2ZP3sV"}
# 1. Switch the source_lang and target_lang in **config** 
# 2. Change the savedir in **config** (eg. "./checkpoints/transformer-back")
# 3. Train model

# %% [markdown] {"id":"waTGz29UP6WI"}
# ## Generate synthetic data with backward model

# %% [markdown] {"id":"sIeTsPexP8FL"}
# ### Download monolingual data

# %% [code] {"id":"i7N4QlsbP8fh","jupyter":{"outputs_hidden":false}}
mono_dataset_name = 'mono'
mono_filename = "ted_zh_corpus.deduped"

# %% [code] {"id":"396saD9-QBPY","jupyter":{"outputs_hidden":false}}
mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_prefix.mkdir(parents=True, exist_ok=True)

urls = (
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ted_zh_corpus.deduped.gz",
)
file_names = (
    'ted_zh_corpus.deduped.gz',
)

for u, f in zip(urls, file_names):
    path = mono_prefix/f
    if not path.exists():
        os.system(f"wget {u} -O {path}")
    else:
        print(f'{f} exists, skip downloading')
    if path.suffix == ".tgz":
        os.system(f"tar -xvf {path} -C {prefix}")
    elif path.suffix == ".zip":
        os.system(f"unzip -o {path} -d {prefix}")
    elif path.suffix == ".gz":
        os.system(f"gzip -fkd {path}")


# %% [markdown] {"id":"JOVQRHzGQU4-"}
# ### TODO: clean corpus
# 
# 1. remove sentences that are too long or too short
# 2. unify punctuation
# 
# hint: you can use clean_s() defined above to do this
def clean_corpus_bk(prefix, l1, l2, ratio=-1, max_len=1000, min_len=1):
    prefix = prefix / mono_filename
    if Path(f'{prefix}.clean.{l1}').exists():
        print(f'{prefix}.clean.{l1} exists. skipping clean.')
        return
    with open(f'{prefix}', 'r') as in_f:
        with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
            with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                for s1 in in_f:
                    s1 = s1.strip()
                    s1 = clean_s(s1, l1)
                    s1_len = len_s(s1, l1)
                    if min_len > 0: # remove short sentence
                        if s1_len < min_len:
                            continue
                    if max_len > 0: # remove long sentence
                        if s1_len > max_len:
                            continue
                    print(s1, file=out_f)

clean_corpus_bk(mono_prefix, src_lang. tgt_lang, ratio=-1, min_len=-1, max_len=-1)


os.system(f"head {str(mono_prefix / mono_filename) +'.clean.'+src_lang} -n 5")
# %% [code] {"id":"eIYmxfUOQSov","jupyter":{"outputs_hidden":false}}


# %% [markdown] {"id":"jegH0bvMQVmR"}
# ### TODO: Subword Units
# 
# Use the spm model of the backward model to tokenize the data into subword units
# 
# hint: spm model is located at DATA/raw-data/\[dataset\]/spm\[vocab_num\].model

# %% [code] {"id":"vqgR4uUMQZGY","jupyter":{"outputs_hidden":false}}
vocab_size = 8000
if (mono_prefix/f'spm{vocab_size}.model').exists():
    print(f'{mono_prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=(f'{mono_prefix}/{mono_filename}.clean.{src_lang}'),
        model_prefix=mono_prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

spm_model = spm.SentencePieceProcessor(model_file=str(mono_prefix/f'spm{vocab_size}.model'))
out_path = mono_prefix / f"{mono_filename}.{src_lang}"
if out_path.exists():
    print(f"{out_path} exists. skipping spm_encode.")
else:
    with open(out_path, 'w') as out_f:
        with open(mono_prefix / f'{mono_filename}.clean.{src_lang}', 'r') as in_f:
            for line in in_f:
                line = line.strip()
                tok = spm_model.encode(line, out_type=str)
                print(' '.join(tok), file=out_f)

# %% [markdown] {"id":"a65glBVXQZiE"}
# ### Binarize
# 
# use fairseq to binarize data

# %% [code] {"id":"b803qA5aQaEu","jupyter":{"outputs_hidden":false}}
binpath = Path('./DATA/data-bin', mono_dataset_name)
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = str(mono_prefix / f"{mono_filename}") # whatever filepath you get after applying subword tokenization
print(f"monopref: {monopref}")
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    os.system(f"python -m fairseq_cli.preprocess\
        --source-lang 'zh'\
        --trainpref {monopref}\
        --destdir {binpath}\
        --srcdict {src_dict_file}\
        --tgtdict {tgt_dict_file}\
        --workers 2")

print(f"done")

# %% [markdown] {"id":"smA0JraEQdxz"}
# ### TODO: Generate synthetic data with backward model
# 
# Add binarized monolingual data to the original data directory, and name it with "split_name"
# 
# ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# 
# then you can use 'generate_prediction(model, task, split="split_name")' to generate translation prediction

# %% [code] {"id":"jvaOVHeoQfkB","jupyter":{"outputs_hidden":false}}
# Add binarized monolingual data to the original data directory, and name it with "split_name"
# ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
os.system(f"cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin")
os.system(f"cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx")
os.system(f"cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin")
os.system(f"cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx")

# %% [code] {"id":"fFEkxPu-Qhlc","jupyter":{"outputs_hidden":false}}
# hint: do prediction on split='mono' to create prediction_file
# generate_prediction( ... ,split=... ,outfile=... )

# %% [markdown] {"id":"Jn4XeawpQjLk"}
# ### TODO: Create new dataset
# 
# 1. Combine the prediction data with monolingual data
# 2. Use the original spm model to tokenize data into Subword Units
# 3. Binarize data with fairseq

# %% [code] {"id":"3R35JTaTQjkm","jupyter":{"outputs_hidden":false}}
# Combine prediction_file (.en) and mono.zh (.zh) into a new dataset.
# 
# hint: tokenize prediction_file with the spm model
# spm_model.encode(line, out_type=str)
# output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh
#
# hint: use fairseq to binarize these two files again
binpath = Path('./DATA/data-bin/synthetic')
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = f"./DATA/rawdata/mono/mono.tok" # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    os.system(f"python -m fairseq_cli.preprocess\
        --source-lang 'zh'\
        --target-lang 'en'\
        --trainpref {monopref}\
        --destdir {binpath}\
        --srcdict {src_dict_file}\
        --tgtdict {tgt_dict_file}\
        --workers 2")

# %% [code] {"id":"MSkse1tyQnsR","jupyter":{"outputs_hidden":false}}
# create a new dataset from all the files prepared above
os.system(f"cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/")

os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin")
os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx")
os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin")
os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx")

# %% [markdown] {"id":"YVdxVGO3QrSs"}
# Created new dataset "ted2020_with_mono"
# 
# 1. Change the datadir in **config** ("./DATA/data-bin/ted2020_with_mono")
# 2. Switch back the source_lang and target_lang in **config** ("en", "zh")
# 2. Change the savedir in **config** (eg. "./checkpoints/transformer-bt")
# 3. Train model

# %% [markdown] {"id":"z-m3IsoJrhmd"}
# # References

# %% [markdown] {"id":"_CZU2beUQtl3"}
# 1. <a name=ott2019fairseq></a>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., ... & Auli, M. (2019, June). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 48-53).
# 2. <a name=vaswani2017></a>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017, December). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000-6010).
# 3. <a name=reimers-2020-multilingual-sentence-bert></a>Reimers, N., & Gurevych, I. (2020, November). Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4512-4525).
# 4. <a name=tiedemann2012parallel></a>Tiedemann, J. (2012, May). Parallel Data, Tools and Interfaces in OPUS. In Lrec (Vol. 2012, pp. 2214-2218).
# 5. <a name=kudo-richardson-2018-sentencepiece></a>Kudo, T., & Richardson, J. (2018, November). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 66-71).
# 6. <a name=sennrich-etal-2016-improving></a>Sennrich, R., Haddow, B., & Birch, A. (2016, August). Improving Neural Machine Translation Models with Monolingual Data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 86-96).
# 7. <a name=edunov-etal-2018-understanding></a>Edunov, S., Ott, M., Auli, M., & Grangier, D. (2018). Understanding Back-Translation at Scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 489-500).
# 8. https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
# 9. https://ithelp.ithome.com.tw/articles/10233122
# 10. https://nlp.seas.harvard.edu/2018/04/03/attention.html
# 11. https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW05/HW05.ipynb

# %% [code] {"id":"Rrfm6iLJQ0tS","jupyter":{"outputs_hidden":false}}
