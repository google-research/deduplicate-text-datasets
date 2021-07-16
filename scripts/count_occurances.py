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

import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Count occurances of sequence.')
parser.add_argument('--suffix', type=str, required=True)
parser.add_argument('--query', type=str)
parser.add_argument('--query_file', type=str)
parser.add_argument('--tokenize', action='store_true')

args = parser.parse_args()

if args.tokenize:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

assert args.query or args.query_file

if args.query:
    if args.tokenize:
        arr = np.array(tokenizer.encode(args.query), dtype=np.uint16).view(np.uint8).tobytes()
    else:
        arr = args.query.encode('utf-8')
    print(arr)
    open("/tmp/fin","wb").write(arr)
    print(os.popen("./target/debug/dedup_dataset count_occurances %s /tmp/fin"%(args.suffix)).read())
else:
    q = open(args.query_file).read()
    if args.tokenize:
        arr = np.array(tokenizer.encode(q), dtype=np.uint16).view(np.uint8).tobytes()
    else:
        arr = q.encode('utf-8')
    print(arr)
    open("/tmp/fin","wb").write(arr.tobytes())
    print(os.popen("./target/debug/dedup_dataset count_occurances %s /tmp/fin"%(args.suffix)).read())
