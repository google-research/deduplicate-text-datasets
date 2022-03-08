# Copyright 2022 Google LLC
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

import sys

original = sys.argv[1]
remove_file = sys.argv[2]
deduped = sys.argv[3]

remove = []
fin = open(remove_file)
for line in fin:
    if 'out' in line: break
for line in fin:
    remove.append(list(map(int,line.split())))
remove = remove[::-1]

ds = open(original,"rb")
new_ds = open(deduped,"wb")

start = 0
while len(remove) > 0:
    a,b = remove.pop()
    new_ds.write(ds.read(a-start))
    ds.seek(b)
    start = b
new_ds.write(ds.read())
