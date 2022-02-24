/* Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* Create and use suffix arrays for deduplicating language model datasets.
 *
 * A suffix array A for a sequence S is a datastructure that contains all
 * suffixes of S in sorted order. To be space efficient, instead of storing
 * the actual suffix, we just store the pointer to the start of the suffix.
 * To be time efficient, it uses fancy algorithms to not require quadratic 
 * (or worse) work. If we didn't care about either, then we could literally
 * just define (in python)
 * A = sorted(S[i:] for i in range(len(S)))
 *
 * Suffix arrays are amazing because they allow us to run lots of string
 * queries really quickly, while also only requiring an extra 8N bytes of
 * storage (one 64-bit pointer for each byte in the sequence).
 * 
 * This code is designed to work with Big Data (TM) and most of the
 * complexity revolves around the fact that we do not require the
 * entire suffix array to fit in memory. In order to keep things managable,
 * we *do* require that the original string fits in memory. However, even
 * the largest language model datasets (e.g., C4) are a few hundred GB
 * which on todays machines does fit in memory.
 *
 * With all that amazing stuff out of the way, just a word of warning: this
 * is the first program I've ever written in rust. I still don't actually
 * understand what borrowing something means, but have found that if I
 * add enough &(&&x.copy()).clone() then usually the compiler just loses
 * all hope in humanity and lets me do what I want. I apologize in advance 
 * to anyone actually does know rust and wants to lock me in a small room
 * with the Rust Book by Klabnik & Nichols until I repent for my sins.
 * (It's me, two months in the future. I now more or less understand how
 * to borrow. So now instead of the code just being all awful, you'll get
 * a nice mix of sane rust and then suddenly OH-NO-WHAT-HAVE-YOU-DONE-WHY!?!)
 */ 

use std::path::Path;
use std::time::Instant;
use std::env;
use std::fs;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use std::convert::TryInto;
use std::cmp::Reverse;


//use std::ffi::OsString;
use std::path::PathBuf;

extern crate filebuffer;
extern crate zstd;
extern crate crossbeam;
extern crate clap;
extern crate fasthash;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use clap::{Parser, Subcommand};

mod table;


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    
    #[clap(arg_required_else_help = true)]
    Make {
	#[clap(short, long)]
	data_file: String,
    },

    MakePart {
	#[clap(short, long)]
	data_file: String,
	#[clap(short, long)]
	start_byte: usize,
	#[clap(short, long)]
	end_byte: usize,
    },

    CountOccurrences {
	#[clap(short, long)]
	data_file: String,
	#[clap(short, long)]
	query_file: String,
    },

    SelfSimilar {
	#[clap(short, long)]
	data_file: String,
	#[clap(short, long)]
	length_threshold: usize,
	#[clap(short, long, default_value_t = 0)]
	frequency_threshold: usize,
	#[clap(short, long)]
	only_save_one: bool,
	#[clap(short, long)]
	cache_dir: String,
	#[clap(short, long, default_value_t = 8)]
	num_threads: i64,
    },

    AcrossSimilar {
	#[clap(long)]
	data_file_1: String,
	#[clap(long)]
	data_file_2: String,
	#[clap(short, long)]
	length_threshold: usize,
	#[clap(short, long)]
	cache_dir: String,
	#[clap(short, long, default_value_t = 8)]
	num_threads: i64,
    },

    Merge {
	#[clap(short, long)]
	suffix_path: Vec<String>,
	#[clap(short, long)]
	output_file: String,
	#[clap(short, long, default_value_t = 8)]
	num_threads: i64,
    },

    Collect {
	#[clap(short, long)]
	data_name: String,
	#[clap(short, long)]
	cache_dir: String,
	#[clap(short, long)]
	length_threshold: u64,
    }
    
}

/* Convert a uint64 array to a uint8 array. 
 * This doubles the memory requirements of the program, but in practice
 * we only call this on datastructures that are smaller than our assumed
 * machine memory so it works.
 */
pub fn to_bytes(input: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 * input.len());

    for value in input {
        bytes.extend(&value.to_le_bytes());
    }
    bytes
}

/* Convert a uint8 array to a uint64. Only called on (relatively) files. */
pub fn from_bytes(input: Vec<u8>) -> Vec<u64> {
    println!("S {}", input.len());
    let mut bytes:Vec<u64> = Vec::with_capacity(input.len()/8);

    for i in 0..input.len()/8 {
	let b = u64::from_le_bytes(input[i*8..i*8+8].try_into().expect("WAT ERR"));
        bytes.push(b);
	
    }
    
    bytes
}

/* Get the next word from the suffix table. */
fn get_next_pointer_from_table(mut tablestream:&mut TableStream) -> u64 {
    if tablestream.ptr >= tablestream.cache.len() {
	let _ = tablestream.file.read_exact(&mut tablestream.cache);
	tablestream.ptr = 0;
    }
    let out = u64::from_le_bytes(tablestream.cache[tablestream.ptr..tablestream.ptr+8].try_into().expect("sdf")) as u64;
    tablestream.ptr += 8;
    return out;
}

fn table_load_disk(table:&mut BufReader<File>, index:usize) -> usize{
    table.seek(std::io::SeekFrom::Start ((index*8) as u64)).expect ("Seek failed!");
    let mut tmp = [0u8; 8];
    table.read_exact(&mut tmp).unwrap();
    return u64::from_le_bytes(tmp) as usize;
}

fn off_disk_position(text: &[u8], table: &mut BufReader<File>, query: &[u8]) -> usize {
    let (mut left, mut right) = (0, text.len());
    while left < right {
        let mid = (left + right) / 2;
	if query < &text[table_load_disk(table, mid)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

struct TableStream {
    file: BufReader<File>,
    cache: Vec<u8>,
    ptr: usize
}


#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeState<'a> {
    suffix: &'a [u8],
    position: u64,
    table_index: usize
}

impl<'a> Ord for MergeState<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix.cmp(&self.suffix)
    }
}

impl<'a> PartialOrd for MergeState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


fn make_table(path: std::string::String, offset: usize) -> TableStream {
    let mut table = TableStream {
	file: std::io::BufReader::new(fs::File::open(path).unwrap()),
	cache: vec![0u8; 1024*1024],
	ptr: 1024*1024
    };
    table.file.seek (std::io::SeekFrom::Start ((offset*8) as u64)).expect ("Seek failed!");
    return table;
}


const HACKSIZE:usize=100000;

fn count_occurances(text: &mut File, mut table: &mut BufReader<File>, size: u64, str: &[u8]) -> u64{
    let mut buf = vec![0u8; str.len()];
    
    let mut low = 0;
    let mut high = size/8-buf.len() as u64;
    while low < high {
	let mid = (high+low)/2;
	let pos = table_load_disk(&mut table, mid as usize);
	text.seek(std::io::SeekFrom::Start(pos as u64)).expect ("Seek failed!");
	text.read_exact(&mut buf).unwrap();
	if str <= &buf {
	    high = mid;
	} else {
	    low = mid+1;
	}
    }
    let start = low;

    let pos = table_load_disk(&mut table, low as usize);
    text.seek(std::io::SeekFrom::Start(pos as u64)).expect ("Seek failed!");
    text.read_exact(&mut buf).unwrap();

    if str != buf {
	return 0; // not found
    }
    
    high = size/8-buf.len() as u64;
    while low < high {
	let mid = (high+low)/2;
	let pos = table_load_disk(&mut table, mid as usize);
	text.seek(std::io::SeekFrom::Start(pos as u64)).expect ("Seek failed!");
	text.read_exact(&mut buf).unwrap();
	if str != &buf {
	    high = mid;
	} else {
	    low = mid+1;
	}
    }
    return low-start;
}

fn cmd_build(fpath: &String)   -> std::io::Result<()> {
    /* Create a suffix array for a given file in one go.
     * Calling this method is memory heavy---it's technically linear in the
     * length of the file, but the constant is quite big.
     * As a result, this method should only be called for files that comfortably
     * fit into memory.
     *
     * The result of calling this method is a new file with ".table.bin" appended
     * to the name which is the suffix array of sorted suffix pointers. This file
     * should be exactly 8x larger than the original file (one u64 pointer per
     * byte of the original).
     * 
     * If the file does not fit into memory, then instead you should use the
     * alternate save_part and then merge_parallel in two steps. See the comments
     * below for how those work.
     */
    let now = Instant::now();
    println!("Reading the dataset at time t={}ms", now.elapsed().as_millis());
    let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
    fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_)?;
    let text = &text_;
    println!("Done reading the dataset at time t={}ms", now.elapsed().as_millis());

    println!("... and now starting the suffix array construction.");
    
    
    let st = table::SuffixTable::new(text);
    println!("Done building suffix array at t={}ms",now.elapsed().as_millis());
    let parts = st.into_parts();
    let table = parts.1;
    
    let mut buffer = File::create(fpath.clone() + ".table.bin")?;
    let bufout = to_bytes(&table);
    println!("Writing the suffix array at time t={}ms", now.elapsed().as_millis());
    buffer.write_all(&bufout)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

fn cmd_build_part(fpath: &String, start: u64, end: u64)   -> std::io::Result<()> {
    /* Create a suffix array for a subsequence of bytes.
     * As with save, this method is linear in the number of bytes that are
     * being saved but the constant is rather high. This method does exactly 
     * the same thing as save except on a range of bytes.
     */
    let now = Instant::now();
    println!("Opening up the dataset files");

    let space_available = std::fs::metadata(fpath.clone()).unwrap().len() as u64;
    assert!(start < end);
    assert!(end <= space_available);
    
    let mut text_ = vec![0u8; (end-start) as usize];
    let mut file = fs::File::open(fpath.clone()).unwrap();
    println!("Loading part of file from byte {} to {}", start, end);
    file.seek(std::io::SeekFrom::Start(start)).expect ("Seek failed!");
    file.read_exact(&mut text_).unwrap();
    let text = &text_;
    println!("Done reading the dataset at time t={}ms", now.elapsed().as_millis());
    println!("... and now starting the suffix array construction.");
    
    let st = table::SuffixTable::new(text);
    println!("Done building suffix array at t={}ms",now.elapsed().as_millis());
    let parts = st.into_parts();
    let table = parts.1;
    
    let mut buffer = File::create(format!("{}.part.{}-{}.table.bin", fpath, start, end))?;
    let mut buffer2 = File::create(format!("{}.part.{}-{}", fpath, start, end))?;
    let bufout = to_bytes(&table);
    println!("Writing the suffix array at time t={}ms", now.elapsed().as_millis());
    buffer.write_all(&bufout)?;
    buffer2.write_all(text)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}    

fn cmd_count_occurrences(fpath: &String, querypath: &String)   -> std::io::Result<()> {
    /* Count the numberof times a particular sequence occurs in the table.
     */
    let mut text = fs::File::open(fpath.clone()).unwrap();

    let mut table = std::io::BufReader::new(fs::File::open(format!("{}.table.bin", fpath)).unwrap());

    let metadata = fs::metadata(format!("{}.table.bin", fpath))?;
    let size = metadata.len();

    let mut str = Vec::with_capacity(std::fs::metadata(querypath.clone()).unwrap().len() as usize);
    fs::File::open(querypath.clone()).unwrap().read_to_end(&mut str)?;

    let occurances = count_occurances(&mut text, &mut table, size, &str[0..str.len()]);

    println!("Number of times present: {}\n", occurances);
    Ok(())
}

fn cmd_self_similar(data_file: &String, length_threshold: &usize, frequency_threshold: &usize,
		    only_save_one: &bool, cache_dir: &String, num_threads: i64)  -> std::io::Result<()> {
    /* Given a string S and suffix array A, compute statistics about how many
     * sequences in A are duplicated (and do it using as many threads as possible).
     * 
     * The basic algorithm is simple. For every pair of items (i,i+1) in the
     * suffix array, we compare the suffixes S[A[i]..] and S[A[i+i]..] and count
     * how many characters they have in common. We then report various statistics
     * about this (e.g., the length of the match, which sequences match each other
     * with at least T tokens, etc).
     * 
     * The first complication is that we can't load all of A into memory at once.
     * This is too big. (e.g., the suffix array for C4 is 2.7 terabytes (!).
     * We might be able to fit 345GB in memory on current hardware, but not
     * 2.7TB. (If you're reading this in 2030, hello there. This must all look
     * very silly to you. But I promise that, today, 2.7TB of memory is just too
     * much. By the way, has AGI taken over the world? I hope not.)
     *
     * Fortunately our algorithm doesn't require random access into A, so we can
     * just stream it off disk and then immediately throw away the old data.
     * 
     * The second complication is that we want this to be fast. Very fast. So
     * we're going to parallelize the algorithm over as many threads as possible.
     * Fortunately this is Rust, and not Python, so the GIL is not going to make
     * life terrible. We set up one copy of the string S in memory, and then we
     * can have each of the threads in parallel stream over A starting at different
     * offsets. 
     */
    println!("Start load!");

    let text = filebuffer::FileBuffer::open(data_file).unwrap();

    let metadata = fs::metadata(format!("{}.table.bin", data_file))?;

    assert!(metadata.len() % (text.len() as u64) == 0);
    
    let ratio = metadata.len()/(text.len() as u64);

    assert!(ratio == 8);

    if !Path::new(&cache_dir).exists() {
	fs::create_dir(cache_dir)?;
    }

    fn sdf(text:&[u8], start:usize, end:usize,
	   length_threshold: usize, frequency_threshold: usize, only_save_one: bool,
	   data_file: String, cache_dir: String) -> usize {
	let mut table = make_table(format!("{}.table.bin", data_file), start);
	let mut prev_location = get_next_pointer_from_table(&mut table);

	let mut outfile = std::io::BufWriter::new(fs::File::create(
	    format!("{}/dups_{}_{}-{}", cache_dir,
		    data_file.split("/").last().unwrap(), start, end)).unwrap());
	let mut outfile_sizes = std::io::BufWriter::new(fs::File::create(
	    format!("{}/sizes_{}_{}-{}", cache_dir,
		    data_file.split("/").last().unwrap(), start, end)).unwrap());

	let mut duplicate_count = 0;
	let mut i = start;
	while i < end {
	    if i%1000000000 == 0 { println!("{} / {} ", i-start, end-start); }
	    let suf1 = &text[prev_location as usize..];
	    
	    let mut cur_location;

	    let mut pairs:Vec<u64> = Vec::with_capacity(4);
	    let mut first = true;
	    
	    loop {
		cur_location = get_next_pointer_from_table(&mut table);
		i += 1;

		let suf2 = &text[cur_location as usize..];
		let does_match =  suf2.len() >= length_threshold && suf1.len() >= length_threshold && suf1[..length_threshold] == suf2[..length_threshold];
		if does_match {
		    if !first {
			pairs.push(cur_location);
		    } else {
			pairs.push(prev_location);
			pairs.push(cur_location);
			first = false;
		    }
		} else {
		    break;
		}
	    }

	    if pairs.len() > frequency_threshold {
		if only_save_one {
		    let seq = &text[pairs[0] as usize..pairs[0] as usize+length_threshold];
		    if pairs[0]%2 == 0 {
			outfile.write_all(seq).expect("Ok");
		    }
		} else {
		    outfile.write_all(&to_bytes(&pairs[..])[..]).expect("Ok");
		    outfile_sizes.write_all(&to_bytes(&[pairs.len() as u64][..])[..]).expect("Ok");
		    duplicate_count += pairs.len();
		}
	    }

	    prev_location = cur_location;
	}

	return duplicate_count;
    }


    let increment:i64 = (text.len() as i64-num_threads)/num_threads;
    let _answer = crossbeam::scope(|scope| {
	let mut result = Vec::with_capacity(num_threads as usize);
	let text = &text;
	for i in 0..num_threads {
	    let one_result = scope.spawn(move || {
		return sdf(text,
			   std::cmp::max(0i64,i*increment-1) as usize,
			   std::cmp::min(((i+1)*increment) as usize, text.len()),
			   *length_threshold, *frequency_threshold, *only_save_one,
			   data_file.clone(), cache_dir.clone());
	    });
	    result.push(one_result);
	}

	let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
	println!("Duplicates found: {:?}", thread_sum);
        
    });
    Ok(())
}

fn cmd_across_similar(data_file_1: &String, data_file_2: &String, cache_dir: &String,
		      length_threshold: usize, num_threads: i64)  -> std::io::Result<()> {
    let text1 = filebuffer::FileBuffer::open(data_file_1).unwrap();
    let text2 = filebuffer::FileBuffer::open(data_file_2).unwrap();

    let metadata1 = fs::metadata(format!("{}.table.bin", data_file_1)).expect("suffix array exists for arg 0");
    let metadata2 = fs::metadata(format!("{}.table.bin", data_file_2)).expect("suffix array exists for arg 1");

    assert!(metadata1.len() % (text1.len() as u64) == 0);
    let ratio = metadata1.len()/(text1.len() as u64);
    assert!(ratio == 8);

    assert!(metadata2.len() % (text2.len() as u64) == 0);
    let ratio = metadata2.len()/(text2.len() as u64);
    assert!(ratio == 8);

    if !Path::new(&cache_dir).exists() {
	fs::create_dir(cache_dir)?;
    }

    fn sdf(text1:&[u8], text2:&[u8],
	   start1:usize, end1:usize,
	   start2:usize, end2:usize,
	   data_file_1: String, data_file_2: String, 
	   cache_dir: String, length_threshold: usize) -> usize {
	let mut table1 = make_table(format!("{}.table.bin", data_file_1), start1);
	let mut location1 = get_next_pointer_from_table(&mut table1);

	let mut table2 = make_table(format!("{}.table.bin", data_file_2), start2);
	let mut location2 = get_next_pointer_from_table(&mut table2);

	// What do you mean this looks ugly. I see no problem here!
	let mut outfile2 = std::io::BufWriter::new(fs::File::create(
	    format!("{}/dups_{}_{}-{}_{}_{}-{}",
		    cache_dir,
		    data_file_1.split("/").last().unwrap(), start1, end1,
		    data_file_2.split("/").last().unwrap(), start2, end2,
	    )).unwrap());
	let mut outfile2_sizes = std::io::BufWriter::new(fs::File::create(
	    format!("{}/sizes_{}_{}-{}_{}_{}-{}",
		    cache_dir,
		    data_file_1.split("/").last().unwrap(), start1, end1,
		    data_file_2.split("/").last().unwrap(), start2, end2,
	    )).unwrap());

	let mut outfile1 = std::io::BufWriter::new(fs::File::create(
	    format!("{}/dups_{}_{}-{}_{}_{}-{}",
		    cache_dir,
		    data_file_2.split("/").last().unwrap(), start2, end2,
		    data_file_1.split("/").last().unwrap(), start1, end1,
	    )).unwrap());
	let mut outfile1_sizes = std::io::BufWriter::new(fs::File::create(
	    format!("{}/sizes_{}_{}-{}_{}_{}-{}",
		    cache_dir,
		    data_file_2.split("/").last().unwrap(), start2, end2,
		    data_file_1.split("/").last().unwrap(), start1, end1,
	    )).unwrap());
	

	let mut i = start1;
	let mut j = start2;
	while i < end1 && j < end2 {
	    if (i+j)%1000000000 == 0 { println!("{} / {} ", i, text1.len()); }
	    
	    let mut suf1 = &text1[location1 as usize..];
	    let mut suf2 = &text2[location2 as usize..];


	    let does_match = suf1.len() >= length_threshold && suf2.len() >= length_threshold && suf1[..length_threshold] == suf2[..length_threshold];

	    if does_match {
		// We have a match between a subsequence in text1 and text2
		let target_suf = &suf1[..length_threshold]; // wlog. equals suf2[..length_threshold]
		let start = i;
		while suf1.len() >= length_threshold && &suf1[..length_threshold] == target_suf {
		    outfile2.write_all(&to_bytes(&[location1 as u64][..])[..]).expect("Ok");

		    location1 = get_next_pointer_from_table(&mut table1);
		    suf1 = &text1[location1 as usize..];
		    i += 1;
		}
		outfile1_sizes.write_all(&to_bytes(&[(i-start) as u64][..])[..]).expect("Ok");

		let start = j;
		while suf2.len() >= length_threshold && &suf2[..length_threshold] == target_suf {
		    outfile1.write_all(&to_bytes(&[location2 as u64][..])[..]).expect("Ok");
		    
		    location2 = get_next_pointer_from_table(&mut table2);
		    suf2 = &text2[location2 as usize..];
		    j += 1;
		}
		outfile2_sizes.write_all(&to_bytes(&[(j-start) as u64][..])[..]).expect("Ok");
	    } else if suf1 < suf2 {
		i += 1;
		location1 = get_next_pointer_from_table(&mut table1);
	    } else if suf2 < suf1 {
		j += 1;
		location2 = get_next_pointer_from_table(&mut table2);
	    } else {
		// This happens only when
		// 1. The two suffixes are identical
		// 2. But they're not yet long enough for it to "count"
		// so we just increment one of the poitners WLOG
		assert!(&suf1 == &suf2);
		assert!(suf1.len() < 100 || suf2.len() < 100);
		i += 1;
		location1 = get_next_pointer_from_table(&mut table1);
	    }
	}

	return 0;
    }


    let increment:i64 = (text1.len() as i64-num_threads)/num_threads;
    let _answer = crossbeam::scope(|scope| {
	let mut result = Vec::with_capacity(num_threads as usize);
	let text1 = &text1;
	let text2 = &text2;
	let mut last_end = 0;
	for i in 0..num_threads {
	    let a = std::cmp::max(0i64,i*increment-1) as usize;
	    let b = std::cmp::min(((i+1)*increment) as usize, text1.len());
	    
	    let mut table1 = std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_file_1)).unwrap());
	    let mut table2 = std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_file_2)).unwrap());
	    let this_start = last_end;
	    
	    let end_seq = &text1[table_load_disk(&mut table1, b)..];
	    let this_end = off_disk_position(text2, &mut table2, end_seq);
	    
	    last_end = this_end;
	    println!("start {} {}", this_start, this_end);
	    let one_result = scope.spawn(move || {
		
		return sdf(text1, text2,
			   a, b,
			   this_start, this_end,
			   data_file_1.clone(), data_file_2.clone(),
			   cache_dir.clone(),
			   length_threshold);
	    });
	    result.push(one_result);
	}

	let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
	println!("Final answer {:?}", thread_sum);
        
    });
    Ok(())
}

fn cmd_merge(data_files: &Vec<String>, output_file: &String, num_threads: i64)  -> std::io::Result<()> {
    /* Merge together M different suffix arrays (probably created with save_part).
     * That is, given strings S_i and suffix arrays A_i compute the suffix array
     * A* = make-suffix-array(concat S_i)
     * In order to do this we just implement mergesort's Merge operation on each
     * of the arrays A_i to construct a sorted array A*.
     *
     * This algorithm is *NOT A LINEAR TIME ALGORITHM* in the worst case. If you run
     * it on a dataset consisting entirely of the character A it will be quadratic.
     * Fortunately for us, language model datasets typically don't just repeat the same
     * character a hundred million times in a row. So in practice, it's linear time.
     *
     * There are thre complications here.
     * 
     * As with selfsimilar_parallel, we can't fit all A_i into memory at once, and
     * we want to make things fast and so parallelize our execution. So we do the
     * same tricks as before to make things work.
     * 
     * However we have one more problem. TODO
     */
    let nn:usize = data_files.len();

    fn load_text2<'s,'t>(fpath:String) -> Vec<u8> {
	println!("Setup buffer");
	let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
	println!("Done buffer {}", text_.len());
	fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_).unwrap();
	println!("Done read buffer");
	return text_;
    }
    
    let texts:Vec<Vec<u8>> = (0..nn).map(|x| load_text2(data_files[x].clone())).collect();

    let texts_len:Vec<usize> = texts.iter().enumerate().map(|(i,x)| x.len() - (if i+1 == texts.len() {0} else {HACKSIZE})).collect();


    let metadatas:Vec<u64> = (0..nn).map(|x| {
	let meta = fs::metadata(format!("{}.table.bin", data_files[x].clone())).unwrap();
	assert!(meta.len()%(texts[x].len() as u64) == 0);
	return meta.len();
    }).collect();

    let ratio = metadatas[0] / (texts[0].len() as u64);
    assert!(ratio == 8);

    println!("Loading ratio is {}", ratio);
    
    fn sdf(texts:&Vec<Vec<u8>>, starts:Vec<usize>, ends:Vec<usize>, texts_len:Vec<usize>, part:usize,
	   output_file: String, data_files: Vec<String>) {

	let nn = texts.len();
	let mut tables:Vec<TableStream> = (0..nn).map(|x| {
	    make_table(format!("{}.table.bin", data_files[x]), starts[x])
	}).collect();
	
	let mut idxs:Vec<u64> = starts.iter().map(|&x| x as u64).collect();
	
	let delta:Vec<u64> = (0..nn).map(|x| {
	    let pref:Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect();
	    pref.iter().sum::<u64>() - (HACKSIZE * x) as u64
	}).collect();

        let mut next_table = std::io::BufWriter::new(File::create(format!("{}.table.bin.{:04}", output_file.clone(), part)).unwrap());

	fn get_next_maybe_skip(mut tablestream:&mut TableStream,
			       index:&mut u64, thresh:usize) -> u64 {
	    let mut location = get_next_pointer_from_table(&mut tablestream);
	    *index += 1;
	    while location >= thresh as u64 {
		location = get_next_pointer_from_table(&mut tablestream);
		*index += 1;
	    }
	    return location;
	}
	
	let mut heap = BinaryHeap::new();

	for x in 0..nn {
	    let position = get_next_maybe_skip(&mut tables[x],
					       &mut idxs[x], texts_len[x]);
	    heap.push(MergeState {
		suffix: &texts[x][position as usize..],
		position: position,
		table_index: x
	    });
	}
	

	let mut prev_position = 0;
	let mut prev = &texts[0][0..];
	while let Some(MergeState {suffix: _suffix, position, table_index}) = heap.pop() {
	    next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()).expect("Write OK");

	    let position = get_next_maybe_skip(&mut tables[table_index],
					       &mut idxs[table_index], texts_len[table_index],);
	    if idxs[table_index] <= ends[table_index] as u64 {
		let next = &texts[table_index][position as usize..];

		let match_len = (0..50000000).find(|&j| !(j < next.len() && j < prev.len() && next[j] == prev[j]));
		if let Some(match_len_) = match_len {
		    if match_len_ > 5000000 {
			println!("{} match len: {}\n", part, match_len_);
			println!("Index {} {}", position, prev_position);
			println!("ugly {:?}", &next[..300]);
		    }
		} else {
		    println!("{} match len: xx\n", part);
		}
		
		heap.push(MergeState {
		    suffix: &texts[table_index][position as usize..],
		    position: position,
		    table_index: table_index
		});
		prev = next;
		prev_position = position;
	    }
	}
    }


    let _answer = crossbeam::scope(|scope| {

	let mut tables:Vec<BufReader<File>> = (0..nn).map(|x| {
	    std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_files[x])).unwrap())
	}).collect();

	let mut starts = vec![0; nn];
	
	for i in 0..num_threads as usize {
	    let texts = &texts;
	    let mut ends: Vec<usize> = vec![0; nn];
	    if i < num_threads as usize-1 {
		ends[0] = (texts[0].len()+(num_threads as usize))/(num_threads as usize)*(i+1);
		let end_seq = &texts[0][table_load_disk(&mut tables[0], ends[0])..];

		for j in 1..ends.len() {
		    ends[j] = off_disk_position(&texts[j], &mut tables[j], end_seq);
		}
	    } else {
		for j in 0..ends.len() {
		    ends[j] = texts[j].len();
		}
	    }

	    for j in 0..ends.len() {
		let l = &texts[j][table_load_disk(&mut tables[j], starts[j])..];
		let l = &l[..std::cmp::min(l.len(), 20)];
		println!("Text{} {:?}", j, l);
	    }

	    println!("Spawn {}: {:?} {:?}", i, starts, ends);

	    let starts2 = starts.clone();
	    let ends2 = ends.clone();
	    let texts_len2 = texts_len.clone();
	    let _one_result = scope.spawn(move || {
		sdf(texts,
		    starts2,
		    ends2,
		    texts_len2,
		    i,
		    (*output_file).clone(),
		    (*data_files).clone(),
		);
	    });

	    for j in 0..ends.len() {
		starts[j] = ends[j];
	    }
	}
    });
    
    println!("Finish writing");
    let mut buffer = File::create(output_file)?;
    for i in 0..texts.len()-1 {
        buffer.write_all(&texts[i][..texts[i].len()-HACKSIZE])?;
    }
    buffer.write_all(&texts[texts.len()-1])?;
    Ok(())
}

/*
 * Given the output of either self-similar or across-similar, 
 * compute byte ranges that are duplicates.
 *
 * The similar outputs are just byte values 
 * [A_0, A_1, ..., A_N] 
 * meaning that the bytes from (A_i, A_i + length_threshold) are duplicated somewhere.
 * 
 * This script converts this to ranges [a, b) for complete ranges that should be removed.
 * For example if we have a long duplicate sequence
 *    abcdefg
 * then we might have a match for `abcde` and `bcdef` and `cdefg`
 * So instead of just saying tokens 0, 1, and 2 match, here we say that [0, 7) match.
 * 
 * To do this we
 *   (a) sort the output lists, and then 
 *   (b) collapse overlapping buckets.
 *
 * Note that as a result of doing this, we might have a sequence `qwerty` where the
 * entire sequence is never repeated in the dataset multiple times, but each byte
 * in the sequence is part of some length_threshold duplicate.
 */
fn cmd_collect(ds_name: &String, cache_dir: &String, length_threshold: u64)  -> std::io::Result<()> {
    let paths = fs::read_dir(cache_dir).unwrap();
    
    let mut path_list = Vec::with_capacity(1000);
    for path in paths {
        let path = path.unwrap().path().as_path().to_str().unwrap().to_string();
	if !path.starts_with(&Path::new(cache_dir).join(format!("dups_{}_", ds_name.clone())).into_os_string().into_string().unwrap()) {
	    continue;
	}
	path_list.push(path);
    }

    // 1. Perform an initial sort of each of the found duplicates
    
    let mut result = Vec::with_capacity(100);
    crossbeam::scope(|scope| {
	for path in path_list.into_iter() {
	    let path = path.clone();
	    let out = scope.spawn(move || {
		let all_items = from_bytes(fs::read(path.clone()).unwrap());
		let mut all_items:Vec<u64> = all_items.into_iter().filter(|&x| x%2 == 0).collect();
		all_items.sort_unstable();
		println!("Done {}", all_items.len());
		return all_items;
	    });
	    result.push(out);
	}
    });
    let outputs:Vec<Vec<u64>> = result.into_iter().map(|t| t.join()).collect();

    let mut all_items:Vec<u64> = Vec::with_capacity(1000);
    println!("Merging.");

    // 2. Perform a merge of the now-sorted lists
    
    let mut heap = BinaryHeap::new();

    // Seed the heap with the first element of each
    for (i, output) in outputs.iter().enumerate() {
	heap.push(Reverse((output[0], 0, i)));
    }

    let mut ranges:Vec<(u64,u64)> = Vec::with_capacity(1000);
    let mut prev_start;
    let mut prev_end;

    // Unroll first iteration of the loop for performance
    if let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
	prev_start = data_pointer;
	prev_end = data_pointer + length_threshold;
	heap.push(Reverse((outputs[which_array][index+1], index+1, which_array)));
    } else {
	println!("No duplicates found! Either the dataset is duplicate-free or something went wrong.");
	return Ok(());
    }
	
    // Now walk the the rest of the merging
    while let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
	all_items.push(data_pointer);

	if data_pointer <= prev_end {
	    prev_end = data_pointer+length_threshold;
	} else {
	    ranges.push((prev_start, prev_end));
	    prev_start = data_pointer;
	    prev_end = data_pointer+length_threshold;
	}
	
	// If this array has more data, consume it
	if index+1 < outputs[which_array].len() {
	    heap.push(Reverse((outputs[which_array][index+1], index+1, which_array)));
	}
    }
    ranges.push((prev_start, prev_end));
    
    let strout:Vec<String> = ranges.iter().map(|&x| format!("{} {}", x.0, x.1)).collect();
    println!("out\n{}", strout.join("\n"));
    Ok(())
}

fn main()  -> std::io::Result<()> {
    
    let args = Args::parse();

    
    match &args.command {
        Commands::Make { data_file } => {
	    cmd_build(data_file)?;
	}

        Commands::MakePart { data_file, start_byte, end_byte } => {
	    cmd_build_part(data_file, *start_byte as u64, *end_byte as u64)?;
	}

        Commands::CountOccurrences { data_file, query_file } => {
	    cmd_count_occurrences(data_file,
				  query_file)?;
	}

        Commands::SelfSimilar { data_file, length_threshold, frequency_threshold, only_save_one, cache_dir, num_threads } => {
	    cmd_self_similar(data_file, length_threshold, frequency_threshold, only_save_one, cache_dir, *num_threads)?;
	}

        Commands::AcrossSimilar { data_file_1, data_file_2, cache_dir, length_threshold, num_threads } => {
	    cmd_across_similar(data_file_1,
			       data_file_2,
			       cache_dir,
			       *length_threshold,
			       *num_threads)?;
	}

        Commands::Merge { suffix_path, output_file, num_threads } => {
	    cmd_merge(suffix_path, output_file, *num_threads)?;
	}

        Commands::Collect { data_name, cache_dir, length_threshold } => {
	    cmd_collect(data_name, cache_dir, *length_threshold)?;
	}
    }
    
    Ok(())
}
