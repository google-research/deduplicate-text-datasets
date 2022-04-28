# Deduplicating Training Data Makes Language Models Better

This repository contains code to deduplicate language model datasets as descrbed in the paper ["Deduplicating Training Data Makes Language Models Better"](https://arxiv.org/abs/2107.06499) by Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch and Nicholas Carlini.
We release the ExactSubstr deduplication implementation (written in Rust) along with the scripts we used in the paper to perform ExactSubstr deduplication and inspect the results (written in Python).
We also release the document clusters resulting from running NearDup deduplication on C4, RealNews, LM1B, and Wiki-4B-en.

This is not an officially supported Google product.

## Why deduplicate?
When datasets are created by scraping raw text from the Internet, this will often result in the same sequences being repeated multiple times (e.g., we find a single 50 word sequence that is repeated in the C4 dataset 60,000 times).
Training models on deduplicated datasets is faster (because they see fewer total examples) and experimentally results in models with similar or better perplexity to models trained on data that hasn't been deduplicated. Moreover, language models are less likely to exhibit memorization when their training data has been well-deduplicated.

## Citing this work

If you use this repository or our deduplicated datasets you can cite

```
@inproceedings{lee2021deduplicating,
      title={Deduplicating Training Data Makes Language Models Better}, 
      author={Katherine Lee and Daphne Ippolito and Andrew Nystrom and Chiyuan Zhang and Douglas Eck and Chris Callison-Burch and Nicholas Carlini},
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```

# Exact Deduplication Code

We provide an implementation of the exact deduplication technique used in the paper.
This is very much research code: it works well for what we designed it to do, and deduplicate text datasets, but it might not directly do what you want it to do.
We did clean it up fairly significantly for a Version 1.0.0 release (see below for release history).
If you want to deduplicate small (<10GB) datasets, it should work on any modern machine with ~16GB of RAM and a few CPU cores. As always, bigger machines are better.
If you want to deduplicate something the size of C4 (~300GB) you will want a machine with as many cores as you can get (we used 96 cores) and >600GB of RAM. You will also need >1TB hard drive space. If your machine is big enough, there should be no upper bound on the size of the dataset it can handle (well, 2^64-1 bytes is the limit, but I think we can all agree that's essentially unlimited).

We build a suffix array (based on [Andrew Gallant's suffix array implementation](https://github.com/BurntSushi/suffix/)) in [src/table.rs](src/table.rs). It has some minor changes from the original version that make it so we can't just import this library as a crate. First, we need 64-bit integers. The original implementation says that u32 works for "reasonably sized documents (~4GB)" but we're working with unreasonably sized documents. So we need u64. Second, we don't want UTF8 strings. Everything is a [u8] byte array, because we might be working over token sequences which aren't valid UTF8.
The main complication in the rest of [src/main.rs](src/main.rs) is the fact that we want things to run in parallel, and we probably can't fit the entire suffix array into memory. And so all of our algorithms are designed around these constraints.


## Installing

To run the rust deduplicator you will need to install Rust:

```curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh```

You'll also need a C compiler, `sudo apt-get install gcc` will do that if you don't already.

If you additionally want to generate datasets to run the rust script on (and you probably do, at least to follow this demo) then you will need python dependencies:

```pip3 install numpy scipy tensorflow tensorflow_datasets transformers sentencepiece```

## Basic Usage

This section walks through the code for getting started using it.
Later we'll cover how to actually deduplicate a dataset, for now we'll just walk through the basics for how it works.

Start by running

```cargo build```

to compile the rust code, and then run

```python3 scripts/load_dataset.py --data_dir $LOAD_DIR --save_dir $SAVE_DIR --name $DATASET --split $SPLIT [--tokenize]```

For example, to get the Wik40B test set (you should do this, to walk through the demo) run

```python3 scripts/load_dataset.py --data_dir ~/tensorflow_datasets --save_dir data --name wiki40b --split test```

This should will take just a minute or so to run on the test set.

If the dataset is really big, you might want to add the `--tokenize` flag. This will shrink the dataset by roughly a factor of two by tokenizing it with the GPT-2 tokenizer.

This will create a file that's called `data/wiki40b.test` and `data/wiki40b.test.size`.
The first file contains the entire Wiki40B test set smashed together, and the second file has the byte offset of where each individual training example begins, in sorted order.

From here we can now build a suffix array of this entire dataset that's now in a single file.

```python3 scripts/make_suffix_array.py [path/to/dataset]```

For example, if you run (you should do this to follow along!)
```python3 scripts/make_suffix_array.py data/wiki40b.test```

This will create a file `data/wiki40b.test.table.bin` containing the suffix array. Again, this should be fast. The test set should process in about a minute.

(When running on larger files, if you get an error that you have too many open files, that's because this script opens lots of files. You should run `ulimit -Sn 1000000` to "fix" the error. You might want to do this preemptively before hitting this crash after hour ten of the job.)

### Querying a suffix array to find duplicated examples

We're not yet going to deduplicate a dataset.
To start, let's just see how to count how often a particular example has been repeated.
To do this, run

```python3 scripts/count_occurrences.py --suffix [path/to/dataset] [--query query_string] [--query_file /path/to/query]```

This should be very fast. Even when you run on a dataset that's 100s of gigabytes, it should take a few seconds, most of which is dominated by Python starting up. The actual core lookup just requires O(log(dataset_size)) time, which often is on the order of ~miliseconds.

On the LM1B test set, running `python3 scripts/count_occurrences.py --suffix data/wiki40b.test --query " on Tuesday"` should return 289. If you tokenized the dataset, then you should pass `--tokenize` to `count_occurrences.py` as well, to get the same result (plus or minus tokenization differences).

If you want to confirm this the outputted number is correct (assuming you haven't tokenized), you can run `cat data/wiki40b.test | grep -ao " on Tuesday" | wc -l` and get the same result (slower).

## Deduplicating a Dataset

Now let's explain how to deduplicate a dataset as we do in the paper. As a running example we'll continue with the LM1b test set.


### Finding all repeated substrings within a document

The first step in deduplicating a dataset is identifying all substrings of a given length that are repeated more than some threshold number of times. To do this we run the `self-similar` command:

```
cargo run self-similar --data-file data/wiki40b.test --length-threshold 100 --cache-dir /tmp/cache --num-threads 8
```

For larger datasets, you may want to replace num-threads with as many cores as you have on your machine. It parallelizes perfectly, so there's no reason not to. For now though, keep it at 8 just for the sake of keeping things on track with this guide.

The output of this should be the string

```
Duplicates found: 3374227
```

This means that the deduplicator found 3,374,227 sequences of length 100 that existed somewhere else in the dataset. The length threshold here is entirely dataset-dependent. In our paper, we used 50 tokens (which is 100 bytes---so remember that if you pass --tokenize you'll need to double the number of bytes for the length threshold).

At this point the deduplicator will have dumped a bunch of files to a cache directory. There are two kinds of files here
- /cache/dups_$DATASET_A-B
- /cache/sizes_$DATASET_A-B

Each `dups` file is a list of pointers into the dataset that corresponds to sequences repeated multiple times. Each file has the duplicates that correspond to items A through B in the suffix array. There should be 28,464 total entries when added up across all of these files. The duplicates are all clustered together, so all duplicates of the same string should appear sequentially.

Each `sizes` file says how large the cluster sizes are. This is typically a small number.

All pointers are the same size, but the size of the pointers depends on the size of the dataset. We use the smallest pointer size that could address the entire dataset. For the LM1B test set, this is a 32-bit pointer. For the training set it would be a 40-bit pointer. For larger documents it might be 48 bits. This helps save memory on disk.

The above explanation might be confusing. Let's see an example. Let's fine the first duplicate in the dataset:
```
$ xxd /tmp/cache/sizes_wiki40b.test_0-64596445 | head -n 1 
00000000: 0200 0000 0200 0000 0200 0000 0200 0000  ................
$ xxd /tmp/cache/dups_wiki40b.test_0-64596445 | head -n 1 
00000000: daa4 ae05 8c7a 8505 c7a4 ae05 797a 8505  .....z......yz
```

Recall these pointers are 32-bit pointers. You can determine this by checking the ratio in size between /tmp/data/lm1b.test and /tmp/data/lm1b.test.table.bin.
So this says that the first cluster of duplicates is of size 2, and starts at location 0x05aea4da in the data file,
with the second occurrence at location 0x05857a8c. To confirm this, you can run
```
$ python3
>>> open("data/wiki40b.test","rb").read()[0x05aea4da:0x05aea4da+100]
b'\n        \n          t\n          \n            0\n          \n        \n        ,\n        \n          t\n  '
>>> open("data/wiki40b.test","rb").read()[0x05857a8c:0x05857a8c+100]
b'\n        \n          t\n          \n            0\n          \n        \n        ,\n        \n          t\n  '
```

And we've confirmed that this example is correctly identified twice in the dataset.
This is a fairly boring and benign duplicate, but it's definitely correct.
(Exercise for the reader: how would you count how many times this string is repeated in the dataset? It should be twice. Can you check that?)


### Collecting the duplicates together

The next step is to take all of the length-100 sequences we've found and collect them together to figure out what we should be removing from our dataset.
To see why this is necessary, imagine that we have a length-200 sequence that's repeated more than once.
The current data we have would tag this sequence as being a duplicate 99 times---once for each initial byte where a match occurs.

This step reduces that down to just find ranges of bytes [a,b) which are duplicated more than once.
To do this, run
```
cargo run collect --data-file data/wiki40b.test --cache-dir /tmp/cache --length-threshold 100 > /tmp/wiki40b.test.remove.byterange
```

The output here will be a long list of byte pair ranges
```
...
out
41887 41999
42347 42479
42507 42715
42741 42931
43101 43315
43891 43993
44021 44220
44366 44604
...
```

What this means is that the substring in the dataset from byte 41887 to byte 41999 is repeated more than once and should be removed, as should the data from bytes 42347 to 42479 and so on.
Let's check this.
```
$ python3
>>> data=open("data/wiki40b.test","rb").read()
>>> data[41887:41999]
b'8\xc2\xa0km\xc2\xb2), all of it land.\n_START_SECTION_\n2010 census\n_START_PARAGRAPH_\nAs of the census of 2010, there were 2,5'
>>> data.count(data[41887:41999])
1 ## WHAT??? See below
>>> data[42347:42479]
b'% from other races, and 0.9% from two or more races. Hispanic or Latino of any race were 2.5% of the population._NEWLINE_There were '
>>> data.count(data[42347:42479])
2
```

Okay so what's going on here? The first of these look like it's repeated just once (but the second looks correct).
Well if you actually check what we're saying here is the following: every byte contained in the range 41887 to 41999 is a member of at least one length-100 duplicate match.
So while the whole sequence isn't repeated, the sub-sequences are. So for example:

```
>>> data.count(data[41887:41887+100])
9
>>> data.count(data[41999-100:41999])
2
```

In our paper we suggest just taking all of these duplicate sequences that have been identified and completely striking them from the dataset.
This somewhat breaks the flow of text, for example if previously had an example "Alice wanted to go to the store" and we deduplicated at the level of 10 characters, we might completely strike " to go to the " and be left with "Alice wantedstore".
In practice we have found this doesn't break the language model because we remove relatively little text, and so these breaks don't cause harm.

How exactly how you write out a dataset that's been deduplicated depends on the format the dataset started as.
If you're just running this on wiki40b, we've provided a script to do this conversion for you which will output another valid TensorFlow Dataset directory. But if you're using some other dataset, this is the part you'll have to take over and write the rest.

To run the wiki40b script, you can just run this command

```
python3 scripts/finish_dedup_wiki40b.py --data_dir ~/tensorflow_datasets/ --save_dir /tmp/tfds_wiki40b --name wiki40b --split test --suffixarray_dir data --remove /tmp/wiki40b.test.remove.byterange
```

This will create a new directory called `/tmp/tfds_wiki40b_dedup`, and will take a few minutes to process completely.

You can verify the deduplication has succeeded by then re-running the pipeline using the resulting output. Instead of finding 3,374,227 duplicate sequences during the deduplication phase, it should instead find 374. Importantly, you can check that these 374 duplicates are not errors of the pipeline: they are new sequences that are now duplicated when previously they were not. You can check this by running `count-occurrences` in the original dataset for the sequences that (now) have two occurrences.

To do this, just re-run everything top-down:
```
python3 scripts/load_dataset.py --data_dir /tmp/tfds_wiki40b_dedup --save_dir data_dedup --name wiki40b --split test
python3 scripts/make_suffix_array.py data_dedup/wiki40b.test
cargo run self-similar --data-file data/wiki40b.test --length-threshold 100 --cache-dir /tmp/cache --num-threads 8
```

and observe the output

```
Duplicates found: 374
```

Why do we get new duplicates? Consider the following example where we're going to remove all sequences of 4 characters that repeat twice: `e a b c d f g h . e f a b c d g h`. Initially the sequence `a b c d` is repeated twice. So we remove them both, and are now left with the file `e f g h . e f g h`. This file still has duplicates! It's not that the first run failed, it's that in doing the first deduplication, we ended up with more (new) duplicates.

To generate the result of our paper, we ran the deduplicator twice. This often cuts the number of duplicates down by over 100,000x, which in practice means to ~zero for normal datasets or ~a few hundred for massive 100GB+ datasets.



## A full end-to-end dataset deduplication example

Okay so maybe you don't like reading. You skipped the entire section above. (Honestly I don't blame you.) You just want it to run.
Then just do this

```
bash scripts/run_pipeline.sh
python3 scripts/finish_dedup_wiki40b.py --data_dir ~/tensorflow_datasets/ --save_dir /tmp/dedup --name wiki40b --split test --suffixarray_dir data --remove /tmp/wiki40b.test.remove.byterange
```

This will run the entire deduplication pipeline top-to-bottom, starting with loading the wiki40b test set, then creating a suffix array, finding all repeated sequences, merging them together to sequence ranges, and finally spitting out a deduplicated TF Dataset that you can use exactly as normal.

Note that this finish script is often the slowest part of the pipeline, depsite doing the least work. I'm sure this is something that could be parallelized or made faster, but it's not an algorithms problem, it's an engineering problem. And that's not particularly fun. If you want to do this and submit a PR we'd gladly take it.

## A full end-to-end single file deduplication example

If you have a large single file and want to remove all length-N duplicates from within that file, we also provide the helper script here

```
bash scripts/deduplicate_single_file.sh [path/to/source] [path/to/destination] [dup_length_threshold] [num_cores]
```



## Advanced Usage

The above scripts work by calling into the core Rust suffix array deduplicator. If you want to do each step yourself, the following options are available:

### Single threaded suffix array construction

To build a suffix array for any particular file, you can run

```cargo run make --data-file [file_path]```

This will create a file called `[file_path].table.bin` which contains the suffix array for the file provided. This algorithm is linear time, but (a) only runs on a single core, and (b) has memory requirement `O(big * len(file))` which is prohibitive for large files.

### Parallel suffix array construction

To build a suffix array for an extremely large file (e.g., ~about as much RAM as available) it is better to run the script

```python scripts/make_suffix_array.py [file_path]```

This script will build the suffix array in parallel by splitting the single file into chunks, generating suffix arrays for each chunk, and then merging the suffix arrays together to form the full suffix array. Note that in general this algorithm is quadratic, but when the maximum substring length is short relative to the total file length (as it is, when generating suffix arrays for N independent training examples) it will never reach this worst case behavior.

The two steps are described below.

#### Building a piece of a suffix array from a piece of a file

The first generates a suffix array from a piece of a file. This is implemented by running

```cargo run make_part --data-file [file_path] --start_byte [byte_offset] --end_byte [byte_offset]```

And builds a suffix array for the byte sequence between [byte_start] and [byte_end] for the given file. Multiple of these can be run in parallel to build a suffix array for a file quickly.

#### Merging suffix array pieces to create a single suffix array

Given the several independent suffix arrays, merging them is now just a matter of calling

```cargo run merge --suffix-path [path_to_partial_suffix_tree] [--suffix-path [another_path_to_partial] ...] -- output-file [tmp_output_directory] --num-threads [number-of-machine-cores]```

to generate a collection of ordered suffix arrays pieces in the output directory. The final step just requires merging these together

```cat [tmp_output_directory]/* > [file_path].table.bin```

### Finding Duplicates

Given a suffix array file, as generated in the previous section, it can now be queried for interesting statistics.
The simplest operation, counting occurrences of particular substrings, takes O(log(N)) time and O(query_length) memory requirements, (as shown above with `scripts/count_occurrences.py`). To do this you can run:

```cargo run count-occurrences --data-file /path/to/dataset --query-file /path/to/query_file```

(Indeed, the python script is just a wrapper that makes calling this nicer, with the option for tokenization.)
This is useful mainly as a commandline interface to interact with the dataset to find interesting properties. To run more sophisticated analysis, use the tools described below:

#### Finding duplicates between two different documents

Given a document A and another document B, we can find all duplicates between the two by (1) constructing suffix arrays for both, and then (2) linearly walking the suffix arrays in order to find all duplicates of a given length.

Once the suffix array for the dataset has been constructed, this algorithm therefore requires time O(len(dataset) + len(query)) and space O(len(dataset)). It is better to run this algorithm when the number of queries into the dataset is greater than O(len(dataset)/log(len(query))). However note that the prior code requires *disk seeks* and and this implementation is a linear scan through the suffix array table, so in practice there is at least a factor-of-10 speedup here. As a rough order of magnitude, for a dataset with ~100GB, it is faster to run `across-similar` (described below) when querying with more than a few megabytes of text. Otherwise it is probably faster to run `count_occurances`.

Notice that this command also requires that the entire dataset fits in memory. For many datasets this is not a problem, but the C4 dataset is 350 GB and the Pile dataset is 750 GB (both even after tokenization). The machine must therefore have a lot of RAM for this to work.

```cargo run across-similar --data-file-1 [dataset1] --data-file-2 [dataset2] --length-threshold [num_bytes] --cache-dir [where/to/save] --num-threads [N]```

This creates files (similar to the self-similar command containing the position of all examples in dataset2 that are also in dataset1, and also at the same time the position of all examples in dataset1 that are also in dataset2. As before, the output is both `dups` files that have the byte offset of where the `length-threshold` duplicates occur, and also `sizes` files that give the sizes of each cluster.

It's again possible to run

```cargo run collect --data-name [dataset1 or dataset2]```

This will write to stdout the byte ranges [a,b) where all tokens in this range are part of an overlap contained in the other document.

#### Finding duplicates within one document

To find duplicates that are contained within one document (for example, to actually deduplicate a dataset as we do in the paper) run the command

```cargo run self-similar --data-file [path] --length-threshold [bytes] --cache-dir [where/to/save] --num-threads [cpu cores]```

This will find all repeated substrings contained in the dataset above a given length threshold.
To see how it is used look above where it's called as part of the dataset deduplication process.
Again run collect_similar to find the indexs of repeated examples.

## Rust Deduplicator Version History

Version 0.1.0 was an initial code release that reproduces the paper.
- The code worked, but was rather terrible.
- I am sorry if you had to look at it.
- You don't want to look at this code unless you're explicitly trying to reproduce our paper.

Version 1.0.0 is complete restructuring of the code. IT IS NOT BACKWARDS COMPATIBLE.
- The suffix array data structure is basically the only thing that remains unchanged (thanks to Andrew Gallant who actually understood how to write code). You won't need to re-generate the suffix array tables if you upgrade from 0.1 to 1.0.
- The rust code now uses argument parsing, instead of relying on the order arguments are passed. So the CLI interface has changed.
- Added one-line scripts to deduplicate a single file, or a TFDS dataset.
- The intermediate data files have changed. This shouldn't matter unless you were looking at the internals of the code. If you were, then you *will* need to re-generate intermediate data files
- The code is not entirely terrible to read, and has comments.


# Approx Deduplication Results

The following CSVs contain three columns: the document ID, a boolean indicating whether or not this document was deleted during deduplication, and a cluster ID.
Documents with the same cluster ID were identified as near-duplicates. For C4 and RealNews, the document ID is the url associated with the document. For Wiki-40B, it is the `wikidata_id`. LM1B coming soon.

**Name**|**Link**|**Size**
:-----:|:-----:|:-----:
C4|[link](https://storage.googleapis.com/gresearch/data_deduplication/c4.tar.gz)|13GB
RealNews|[link](https://storage.googleapis.com/gresearch/data_deduplication/realnews.tar.gz)|1.4GB
Wiki-40B|[link](https://storage.googleapis.com/gresearch/data_deduplication/wiki40b.tar.gz)|26MB
