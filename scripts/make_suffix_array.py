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
import time
import sys
import multiprocessing as mp
import numpy as np

data_size = os.path.getsize(sys.argv[1])

HACK = 100000


started = []

if data_size > 10e9:
    total_jobs = 100
    jobs_at_once = 20
elif data_size > 1e9:
    total_jobs = 96
    jobs_at_once = 96
elif data_size > 10e6:
    total_jobs = 4
    jobs_at_once = 4
else:
    total_jobs = 1
    jobs_at_once = 1

S = data_size//total_jobs


for jobstart in range(0, total_jobs, jobs_at_once):
    wait = []
    for i in range(jobstart,jobstart+jobs_at_once):
        s, e = i*S, min((i+1)*S+HACK, data_size)
        cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(sys.argv[1], s, e)
        started.append((s, e))
        print(cmd)
        wait.append(os.popen(cmd))
        
        if e == data_size:
            break

    print("Waiting for jobs to finish")
    [x.read() for x in wait]

print("Checking all wrote correctly")

while True:
    files = ["%s.part.%d-%d"%(sys.argv[1],s, e) for s,e in started]
    
    wait = []
    for x,(s,e) in zip(files,started):
        size_data = os.path.getsize(x)
        FACT = np.ceil(np.log(size_data)/np.log(2)/8)
        print("FACT", FACT)
        size_table = os.path.getsize(x+".table.bin")
        if not os.path.exists(x) or not os.path.exists(x+".table.bin") or size_table == 0 or size_data*FACT != size_table:
            cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(sys.argv[1], s, e)
            print(cmd)
            wait.append(os.popen(cmd))
    print("Rerunning", len(wait), "jobs because they failed.")
    [x.read() for x in wait]
    time.sleep(1)
    if len(wait) == 0:
        break
        

print("Merging suffix trees")

torun = " --suffix-path ".join(files)
print("./target/debug/dedup_dataset merge --output-file %s --suffix-path %s --num-threads %d"%("tmp/out.table.bin", torun, mp.cpu_count()))
os.popen("./target/debug/dedup_dataset merge --output-file %s --suffix-path %s --num-threads %d"%("tmp/out.table.bin", torun, mp.cpu_count())).read()
#exit(0)
print("Now merging individual tables")
os.popen("cat tmp/out.table.bin.* > tmp/out.table.bin").read()
print("Cleaning up")
#os.popen("rm tmp/out.table.bin.*").read()
os.popen("mv tmp/out.table.bin %s.table.bin"%sys.argv[1]).read()

