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

import datasets
import os
import struct
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm
import glob

import argparse


FILE_EXTENSIONS = {"text": "txt", "json": "jsonl", "csv": "csv"}

parser = argparse.ArgumentParser(description='Load a dataset.')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--split', type=str)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--num_workers', type=int, default=None)
parser.add_argument('--text_feature_key', type=str, default="text")

args = parser.parse_args()

if args.tokenize:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

save_dir = args.save_dir
data_dir = args.data_dir
dataset_name = args.name
split = args.split
subset = args.subset
tokenize = args.tokenize
num_workers = args.num_workers
key = args.text_feature_key

if dataset_name in FILE_EXTENSIONS:
    assert data_dir is not None
    data_files = glob.glob(f"{data_dir}/*.{FILE_EXTENSIONS[dataset_name]}")
    ds = datasets.load_dataset(dataset_name, subset, data_files=data_files, split=split)
else:
    ds = datasets.load_dataset(dataset_name, subset, split=split)
assert isinstance(ds, datasets.Dataset), "This is not a HF-dataset. It might be a DatasetDict. Try passing `split`?"

UID = 0


def sep():
    global UID
    UID += 1
    return b"\xff\xff" + struct.pack("<I", UID)


def tokenize_to_bytes(examples):
    tokenized = tokenizer(examples[key])
    tokenized["input_ids"] = [np.array(input_ids, dtype=np.uint16).view(np.uint8).tobytes() for input_ids in
                              tokenized["input_ids"]]
    return tokenized


os.makedirs(save_dir, exist_ok=True)
fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")
sizes = [0]

if tokenize:
    ds = ds.map(tokenize_to_bytes, batched=True, num_proc=num_workers)
    key = "input_ids"

for example in tqdm(ds):
    out = example[key] if tokenize else example[key].encode("utf8")
    next_line = sep() + out
    fout.write(next_line)
    sizes.append(sizes[-1] + len(next_line))

open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
    np.array(sizes, dtype=np.uint64).tobytes())
