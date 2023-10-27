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

parser = argparse.ArgumentParser(description='Count occurrences of sequence.')
parser.add_argument('--suffix', type=str, required=True)
parser.add_argument('--query', type=str)
parser.add_argument('--query_file', type=str)
parser.add_argument('--print_location', action="store_true")
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--load_disk', action='store_true')
parser.add_argument('--tokenizer', type=str, default="gpt2")

args = parser.parse_args()

if args.tokenize:
    if args.tokenizer == 'gpt2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif args.tokenizer == 't5':
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
    elif args.tokenizer == "mytik":
        import tiktoken
        from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
        import numpy as np
        
        import os
        import json
        
        
        def mycl100k_base():
            mergeable_ranks = load_tiktoken_bpe(
                "/home/ncarlini/deduplicate-text-datasets/scripts/my.bpe"
            )
            
            ENDOFTEXT = "<|enZdo||ftext|>"
            FIM_PREFIX = "<|fiZm_||prefix|>"
            FIM_MIDDLE = "<|fiZm_||middle|>"
            FIM_SUFFIX = "<|fiZm_||suffix|>"
            ENDOFPROMPT = "<|endZo||fprompt|>"
        
            special_tokens = {
                ENDOFTEXT: 65002,
                FIM_PREFIX: 65003,
                FIM_MIDDLE: 65004,
                FIM_SUFFIX: 65005,
                ENDOFPROMPT: 65006,
            }
        
            return {
                "name": "mycl100k_base",
                "pat_str": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                "mergeable_ranks": mergeable_ranks,
                "special_tokens": special_tokens,
            }
        
        
        
        tokenizer = tiktoken.Encoding(**mycl100k_base())
    else:
        raise

assert args.query or args.query_file

print_location = "--print-location" if args.print_location else ""
print_location += " --load-disk" if args.load_disk else ""

if args.query:
    if args.tokenize:
        arr = np.array(tokenizer.encode(args.query), dtype=np.uint16).view(np.uint8).tobytes()
    else:
        arr = args.query.encode('utf-8')
    #print(arr)
    open("/tmp/fin","wb").write(arr)
    print(os.popen("./target/debug/dedup_dataset count-occurrences --data-file %s --query-file /tmp/fin %s"%(args.suffix,print_location)).read())
else:
    if args.tokenize:
        q = open(args.query_file).read()
        arr = np.array(tokenizer.encode(q), dtype=np.uint16).view(np.uint8).tobytes()
    else:
        arr = open(args.query_file,"rb").read()
    #print(arr)
    open("/tmp/fin","wb").write(arr)
    print(os.popen("./target/debug/dedup_dataset count-occurrences --data-file %s --query-file /tmp/fin %s"%(args.suffix, print_location)).read())
