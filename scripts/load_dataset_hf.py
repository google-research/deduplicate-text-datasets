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

import argparse

parser = argparse.ArgumentParser(description='Load a dataset.')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--text_feature_key', type=str, default="text")

args = parser.parse_args()

if args.tokenize:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

data_dir = args.data_dir
save_dir = args.save_dir
split = args.split
subset = args.subset
dataset_name = args.name
key = args.text_feature_key
tokenize = args.tokenize

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


def str_to_bytes(examples):
    examples["text"] = [text.encode("utf8") for text in examples["text"]]
    return examples


os.makedirs(save_dir, exist_ok=True)
fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")
sizes = [0]

if tokenize:
    ds = ds.map(tokenize_to_bytes, batched=True)
    key = "input_ids"
else:
    ds = ds.map(str_to_bytes, batched=True)

for x in ds[key]:
    next_line = sep() + x
    fout.write(next_line)
    sizes.append(sizes[-1] + len(next_line))

open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
    np.array(sizes, dtype=np.uint64).tobytes())
