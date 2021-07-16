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
        cmd = "./target/debug/dedup_dataset save_part %s %d %d"%(sys.argv[1], s, e)
        started.append((s, e))
        print(cmd)
        wait.append(os.popen(cmd))
        
        if e == data_size:
            break

    print("Waiting for jobs to finish")
    [x.read() for x in wait]

print("Checking all wrote correctly")

while True:
    files = ["%s.%d-%d"%(sys.argv[1],s, e) for s,e in started]
    
    wait = []
    for x,(s,e) in zip(files,started):
        if not os.path.exists(x) or not os.path.exists(x+".table.bin") or os.path.getsize(x) == 0 or os.path.getsize(x)*8 != os.path.getsize(x+".table.bin"):
            cmd = "./target/debug/dedup_dataset save_part %s %d %d"%(sys.argv[1], s, e)
            print(cmd)
            wait.append(os.popen(cmd))
    print("Rerunning", len(wait), "jobs because they failed.")
    [x.read() for x in wait]
    time.sleep(1)
    if len(wait) == 0:
        break
        

print("Merging suffix trees")

torun = " ".join(files)
print(torun)
os.popen("./target/debug/dedup_dataset merge_parallel %s tmp/out"%torun).read()
#exit(0)
print("Now merging individual tables")
os.popen("cat tmp/out.table.bin.* > tmp/out.table.bin").read()
print("Cleaning up")
#os.popen("rm tmp/out.table.bin.*").read()
os.popen("mv tmp/out.table.bin %s.table.bin"%sys.argv[1]).read()

