# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow_datasets as tfds
import tensorflow as tf
import datasets
import os
import struct
import numpy as np
from transformers import GPT2Tokenizer
import multiprocessing as mp

import argparse

parser = argparse.ArgumentParser(description='Load a dataset.')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--dataset_format', type=str, default="tfds")
parser.add_argument('--split', type=str)
parser.add_argument('--hf_ds_subset', type=str, default=None)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--text_feature_key', type=str, default="text")
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()

if args.tokenize:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

split = args.split
data_dir = args.data_dir
save_dir = args.save_dir
dataset_name = args.name
dataset_format = args.dataset_format
key = args.text_feature_key
num_workers = args.num_workers


if dataset_format == "tfds":
    ds = tfds.load(dataset_name, split=split, shuffle_files=False, batch_size=2**16,
                   data_dir=data_dir)
    assert isinstance(ds, tf.data.Dataset)
    print(ds)
elif dataset_format == "hf":
    subset = args.hf_ds_subset
    ds = datasets.load_dataset(dataset_name, subset, split=split)
    assert isinstance(ds, datasets.Dataset), "This is not a HF-dataset. It might be a DatasetDict. Try passing `split`?"
    print(ds)

UID = 0
def sep():
    global UID
    UID += 1
    return b"\xff\xff"+struct.pack("<I", UID)

def tok(x):
    if args.tokenize:
        if dataset_format == "tfds":
            x = x.numpy().decode("utf8")
        out = tokenizer.encode(x)
        out = np.array(out, dtype=np.uint16).view(np.uint8).tobytes()
    else:
        if dataset_format == "hf":
            out = x.encode("utf8")
        else:
            out = x.numpy()
    return out


os.makedirs(save_dir, exist_ok=True)
fout = open(os.path.join(save_dir, dataset_name+"."+split), "wb")

if num_workers > 1:
    p = mp.Pool(96)
else:
    p = None

i = 0
sizes = [0]
if dataset_format == "hf":
    ds = ds[key]
for text in ds:
    print(i)

    if dataset_format == "tfds":
        text = text[key]
    if num_workers > 1:
        text = p.map(tok, text)
    else:
        text = map(tok, text)
    
    for x in text:
        next_line = sep()+x
        fout.write(next_line)
        sizes.append(sizes[-1]+len(next_line))
    i += 1

open(os.path.join(save_dir,dataset_name+"."+split+".size"), "wb").write(np.array(sizes,dtype=np.uint64).tobytes())
