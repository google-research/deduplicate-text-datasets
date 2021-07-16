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

use std::time::Instant;
use std::borrow::Cow;
use std::env;
use std::fs;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use std::convert::TryInto;
use std::collections::HashMap;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

extern crate filebuffer;
extern crate zstd;
extern crate crossbeam;
extern crate fasthash;

mod table;


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
    println!("{}", bytes.len());
    bytes
}

/* Convert a uint16 array to a uint8 array.
 * Again, we only call this on datastructures that are smaller than our
 * assumed machine memory.
 */
pub fn to_bytes_16(input: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(2 * input.len());

    for value in input {
            bytes.extend(&value.to_le_bytes());
    }
    println!("{}", bytes.len());
    bytes
}

/* Convert a uint8 array to a uint64. Only called on small files. */
pub fn from_bytes(input: Vec<u8>) -> Vec<u64> {
    println!("S {}", input.len());
    let mut bytes:Vec<u64> = Vec::with_capacity(input.len()/8);

    for i in 0..input.len()/8 {
	    let b = u64::from_le_bytes(input[i*8..i*8+8].try_into().expect("WAT ERR"));
            bytes.push(b);

    }

    bytes
}

/* General binary search algorithm duplicated from table.rs because
 * I don't know how to import methods across files... Sorry.
 */
fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = (left + right) / 2;
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}


/* Find the longest match of a query [u8] in a suffix array.
 * This algorithm is fairly simple. 
 * Given the suffix array, we just need to find out where the query
 * should be inserted into the suffix array. This takes O(log(n)) time.
 * However, after this, we then need to then figure out what the longest
 * match is. It's either the suffix above the insertion position, or below
 * the insertion position. So we compare to both, and return whichever is
 * longer of the two.
 * When it succeeds, it returns (length of match, index of match in array)
 */
fn find_longest_match(st: &table::SuffixTable, query: &[u8]) -> (usize, usize) {
    //println!("Finding longest match for {}", query);
    let start_idx = binary_search(st.table(), |&sufix| {
         query <= &st.text()[sufix as usize..]
    });

    let back = if start_idx == 0 {
        vec![0]
    } else if start_idx == st.table().len() {
        vec![1]
    } else {
        vec![0, 1]
    };
    
    let res = back.iter().map(|offset| {
        let start_pos = st.table()[start_idx-offset] as usize;
        let match_len_ = (1..std::cmp::min(query.len()+1, st.text().len()-start_pos))
  	.filter(|&count| {
	    //println!("{} {}", st.text().len(), start_pos);
            let a = &query[..count as usize];
	    let b = &st.text()[start_pos..start_pos+count];
	    a == b
        }).max();
        if let Some(match_len) = match_len_ {
            match_len
        } else {
	    0
	}
    }).max();
    if let Some(res_) = res {
        (res_, start_idx)
    } else {
        (0, 0)
    }
}

/* TODO */
fn all_longest_matches(st: &table::SuffixTable, str: &[u8]) -> Vec<u16>{
    let mut longest_match = vec![0; str.len()/2];
    let now = Instant::now();
    for i in (0..str.len()).step_by(2) {
        if i%1000000 == 0 {println!("{}/{} {}", i, str.len(), now.elapsed().as_millis());}
        if true {
	    let mut max_potential_match = 25;
	    loop {
	        let mpm_fix = std::cmp::min(str.len()-i, max_potential_match);
                let matchlen = find_longest_match(st, &str[i..i+mpm_fix]).0;
		if matchlen == max_potential_match {
		    max_potential_match *= 2;
		} else {
	            for j in i/2..(i+matchlen)/2 {
	                longest_match[j] = std::cmp::max(longest_match[j], matchlen as u16);
	            }
		    break;
		}
	    }
	}
    }
    longest_match
}

/* TODO */
fn find_longest_match_disk(text: &[u8], table: &mut BufReader<File>, query: &[u8]) -> (usize, usize) {
    let (mut left, mut right) = (0, text.len());
    while left < right {
        let mid = (left + right) / 2;
	if query <= &text[table_load_disk(table, mid)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    let start_idx = left;

	
    let back = if start_idx == 0 {
        vec![0]
    } else if start_idx == text.len() {
        vec![1]
    } else {
        vec![0, 1]
    };
    
    let res = back.iter().map(|offset| {
        let start_pos = table_load_disk(table, start_idx-offset);
        let match_len_ = (1..std::cmp::min(query.len()+1, text.len()-start_pos))
  	.filter(|&count| {
	    //println!("{} {}", st.text().len(), start_pos);
            let a = &query[..count as usize];
	    let b = &text[start_pos..start_pos+count];
	    a == b
        }).max();
        if let Some(match_len) = match_len_ {
            match_len
        } else {
	    0
	}
    }).max();
    if let Some(res_) = res {
        (res_, start_idx)
    } else {
        (0, 0)
    }
}

/* Just a dup of without disk but call find_longest_match_disk */
fn all_longest_matches_disk(text: &[u8], mut table: &mut BufReader<File>, str: &[u8]) -> Vec<u16>{
    let mut longest_match = vec![0; str.len()/2];
    let now = Instant::now();
    for i in (0..str.len()).step_by(2) {
        if i%100000 == 0 {println!("{}/{} {}", i, str.len(), now.elapsed().as_millis());}
        if true {
	    let mut max_potential_match = 25;
	    loop {
	        let mpm_fix = std::cmp::min(str.len()-i, max_potential_match);
                let matchlen = find_longest_match_disk(&text, &mut table, &str[i..i+mpm_fix]).0;
		if matchlen == max_potential_match {
		    max_potential_match *= 2;
		} else {
	            for j in i/2..(i+matchlen)/2 {
	                longest_match[j] = std::cmp::max(longest_match[j], matchlen as u16);
	            }
		    break;
		}
	    }
	}
    }
    longest_match
}

/* TODO */
pub fn fast_positions(st: &table::SuffixTable, query: &[u8],
		      _start: usize, _end: usize) -> usize {
    // We can quickly decide whether the query won't match at all if
    // it's outside the range of suffixes.
    if st.text().len() == 0
        || query.len() == 0
        || (query < st.suffix_bytes(0)
            && !st.suffix_bytes(0).starts_with(query))
        || query > st.suffix_bytes(st.len() - 1)
    {
        return 0;
    }

    // Maybe later: start_ and end_ can help us narrow down the search range
    let start = binary_search(&st.table(), |&sufi| {
        query <= &st.text()[sufi as usize..]
    });
    //println!("{} {}", start, start_);
    //assert!(start == start_);
    let end = start
        + binary_search(&st.table()[start..], |&sufi| {
            !st.text()[sufi as usize..].starts_with(query)
        });

    // Whoops. If start is somehow greater than end, then we've got
    // nothing.
    if start > end {
        0
    } else {
        end-start
    }
}


/* TODO */
fn all_count_matches(st: &table::SuffixTable, str: &[u8]) -> Vec<u16>{
    let mut all_matches = vec![0; str.len()*150];
    let now = Instant::now();
    for i in (0..str.len()).step_by(2) {
        if i%1000000 == 0 {println!("{}/{} {}", i, str.len(), now.elapsed().as_millis());}
        if true {
	    let mut max_potential_match = 25;
	    loop {
	        let mpm_fix = std::cmp::min(str.len()-i, max_potential_match);
                let (matchlen, st_start) = find_longest_match(st, &str[i..i+mpm_fix]);
		if matchlen == max_potential_match {
		    max_potential_match *= 2;
		} else {
		    //let mut prior = 1;
	            //println!("Len {}:", matchlen);
		    for j in (2..(matchlen+2)).step_by(2) {
			let mut cur = fast_positions(st, &str[i..i+j], st_start, 0);
			//println!("  {}:{}", j, cur);
			//std::assert!(cur >= prior);
			//assert!(cur < 65535);
			cur = std::cmp::min(cur, 65535);
			all_matches[i*150+j/2] = cur as u16;
			//prior = cur;
		    }
		    break;
		}
	    }
	}
    }
    all_matches
}

/* TODO */
fn load_text<'s,'t>(arg:usize) -> Vec<u8> {
    let fpath = env::args().nth(arg).unwrap();
    println!("Setup buffer");
    let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
    println!("Done buffer {}", text_.len());
    fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_).unwrap();
    println!("Done read buffer");
    return text_;
}

/* TODO */
fn load_table_64<'s,'t>(arg:usize) -> table::SuffixTable<'s,'t> {
    let fpath = env::args().nth(arg).unwrap();
    let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
    fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_).unwrap();
    
    let table = from_bytes(fs::read(fpath.clone() + ".table.bin").unwrap());
    let st = table::SuffixTable::from_parts(Cow::Owned(text_), Cow::Owned(table));
    return st;
}

/* Get the next word from the suffix table. */
fn get_next_82(mut tablestream:&mut TableStream) -> u64 {
    if tablestream.ptr >= tablestream.cache.len() {
	tablestream.file.read_exact(&mut tablestream.cache);
	tablestream.ptr = 0;
    }
    let out = u64::from_le_bytes(tablestream.cache[tablestream.ptr..tablestream.ptr+8].try_into().expect("sdf")) as u64;
    tablestream.ptr += 8;
    return out;
}

/* At some point I should remove this and replace witthis
 * with get_next_82.  */
fn get_next_8(file:&mut BufReader<File>, mut cache:&mut [u8], ptr:&mut usize) -> u64 {
    if *ptr >= cache.len() {
	file.read_exact(&mut cache);
	*ptr = 0;
    }
    let out = u64::from_le_bytes(cache[*ptr..*ptr+8].try_into().expect("sdf")) as u64;
    *ptr += 8;
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
struct State<'a> {
    suffix: &'a [u8],
    position: u64,
    table_index: usize
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

impl<'a> Ord for State<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix.cmp(&self.suffix)
    }
}

impl<'a> PartialOrd for State<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


const HACKSIZE:usize=100000;

fn get_example_index(table:&[u64], position:u64) -> usize{
    return binary_search(table, |&value| {
        position < value
    });
}

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
    
    high = size/8-buf.len() as u64;
    while low < high {
	let mid = (high+low)/2;
	let pos = table_load_disk(&mut table, mid as usize);
	text.seek(std::io::SeekFrom::Start(pos as u64)).expect ("Seek failed!");
	text.read_exact(&mut buf).unwrap();
	//println!("{:?}", buf);
	if str != &buf {
	    high = mid;
	} else {
	    low = mid+1;
	}
    }
    return low-start;
}


fn main()  -> std::io::Result<()> {

    let op = env::args().nth(1).unwrap();

    if op == "save" {
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
	println!("Start load {}", now.elapsed().as_millis());
	let fpath = env::args().nth(2).unwrap();
	let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
	fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_)?;
	let text = &text_;
	println!("Done read {}", now.elapsed().as_millis());
	
	
        let st = table::SuffixTable::new(text);
	println!("Done table {}",now.elapsed().as_millis());
        let parts = st.into_parts();
        let table = parts.1;
    
        let mut buffer = File::create(fpath.clone() + ".table.bin")?;
	println!("Create buffer {}", now.elapsed().as_millis());
	let bufout = to_bytes(&table);
	println!("Write buffer {}", now.elapsed().as_millis());
        buffer.write_all(&bufout)?;
	println!("Done buffer {}", now.elapsed().as_millis());
    } else if op == "save_part" {
	/* Create a suffix array for a subsequence of bytes.
	 * As with save, this method is linear in the number of bytes that are
	 * being saved but the constant is rather high. This method does exactly 
	 * the same thing as save except on a range of bytes.
	 */
	let now = Instant::now();
	println!("Start load {}", now.elapsed().as_millis());
	let fpath = env::args().nth(2).unwrap();
	let start = env::args().nth(3).unwrap().parse::<u64>().unwrap();
	let end = env::args().nth(4).unwrap().parse::<u64>().unwrap();

	let space_available = std::fs::metadata(fpath.clone()).unwrap().len() as u64;
	assert!(start < end);
	assert!(end <= space_available);
	
	let mut text_ = vec![0u8; (end-start) as usize];
	let mut file = fs::File::open(fpath.clone()).unwrap();
	println!("Loading part of file {} {}", start, end);
	file.seek(std::io::SeekFrom::Start(start)).expect ("Seek failed!");
	file.read_exact(&mut text_).unwrap();
	let text = &text_;
	println!("Done read {}", now.elapsed().as_millis());
	
        let st = table::SuffixTable::new(text);
	println!("Done table {}",now.elapsed().as_millis());
        let parts = st.into_parts();
        let table = parts.1;
    
        let mut buffer = File::create(format!("{}.{}-{}.table.bin", fpath, start, end))?;
        let mut buffer2 = File::create(format!("{}.{}-{}", fpath, start, end))?;
	println!("Create buffer {}", now.elapsed().as_millis());
	let bufout = to_bytes(&table);
	println!("Write buffer {}", now.elapsed().as_millis());
        buffer.write_all(&bufout)?;
        buffer2.write_all(text)?;
	println!("Done buffer {}", now.elapsed().as_millis());
    } else if op == "load" || op == "loadall" {
	/* Compute, for each byte of input, how long the overlap is in the suffix array.
	 */
	let fpath = env::args().nth(2).unwrap();
        let st = load_table_64(2);

        let loadpath = env::args().nth(3).unwrap();
        let paths = fs::read_dir(loadpath).unwrap();
	for path in paths {
            let path = path.unwrap().path().as_path().to_str().unwrap().to_string();

	    println!("\nRunning for {}", path);

	    let mut str = Vec::with_capacity(std::fs::metadata(path.clone()).unwrap().len() as usize);	
	    fs::File::open(path.clone()).unwrap().read_to_end(&mut str)?;
	    
	    println!("Loaded file!");

            let mut encoder = {
		let target = File::create(["outs3/data",
					   (fpath.clone().split("/").last().unwrap()),
					   (path.clone().split("/").last().unwrap()),
					   "out.zst"].join("."))?;
		zstd::Encoder::new(target, 1)
	    }?;
	    
            let ans;
            if op == "load" {
		ans = all_longest_matches(&st, &str);
		encoder.write_all(&to_bytes_16(&ans))?;
	    } else {
		ans = all_count_matches(&st, &str);
		encoder.write_all(&to_bytes_16(&ans))?;
	    }
	    encoder.finish()?;
	}

    } else if op == "load_parallel" {
	/* Do the same thing as the above loading, but in parallel.
	 */
        let st = load_table_64(2);

        let loadpath = env::args().nth(3).unwrap();
        let paths = fs::read_dir(loadpath).unwrap();

	let _answer = crossbeam::scope(|scope| {
	    let st = &st;
	    for path in paths {
		let _one_result = scope.spawn(move || {
		    let path = path.unwrap().path().as_path().to_str().unwrap().to_string();
		    
		    println!("\nRunning for {}", path);

		    let mut str = Vec::with_capacity(std::fs::metadata(path.clone()).unwrap().len() as usize);	
		    fs::File::open(path.clone()).unwrap().read_to_end(&mut str).unwrap();
		    
		    println!("Loaded file!");

		    let fpath = env::args().nth(2).unwrap();
		    let mut encoder = {
			let target = File::create(["outs3/data",
						   (fpath.clone().split("/").last().unwrap()),
						   (path.clone().split("/").last().unwrap()),
						   "out.zst"].join(".")).unwrap();
			zstd::Encoder::new(target, 1)
		    }.unwrap();
		    
		    let ans = all_longest_matches(st, &str);
		    encoder.write_all(&to_bytes_16(&ans)).unwrap();
		    encoder.finish().unwrap();
		});
	    }
	});
	
    } else if op == "load_disk" {
	/* And do the load again, but this time off disk.
	 */
        let text = load_text(2);

	let metadata = fs::metadata(env::args().nth(2).unwrap() + ".table.bin")?;

	assert!(metadata.len() % (text.len() as u64) == 0);
	
	let ratio = metadata.len()/(text.len() as u64);
	assert!(ratio == 8);

	let mut table = std::io::BufReader::new(fs::File::open(env::args().nth(2).unwrap() + ".table.bin").unwrap());

	let path = env::args().nth(3).unwrap();
	println!("\nRunning for {}", path);

	let mut str = Vec::with_capacity(std::fs::metadata(path.clone()).unwrap().len() as usize);
	fs::File::open(path.clone()).unwrap().read_to_end(&mut str)?;
	
	println!("Loaded file!");

        let mut encoder = {
	    let target = File::create(["/tmp/qout.zst"].join("."))?;
	    zstd::Encoder::new(target, 1)
	}?;
	
        let ans = all_longest_matches_disk(&text, &mut table, &str);
	encoder.write_all(&to_bytes_16(&ans))?;
	encoder.finish()?;

    } else if op == "count_occurances" {
	/* Count the numberof times a particular sequence occurs in the table.
	 */
	let mut text = fs::File::open(env::args().nth(2).unwrap()).unwrap();

	let mut table = std::io::BufReader::new(fs::File::open(env::args().nth(2).unwrap() + ".table.bin").unwrap());

	let metadata = fs::metadata(env::args().nth(2).unwrap() + ".table.bin")?;
	let size = metadata.len();

	let path = env::args().nth(3).unwrap();

	let mut str = Vec::with_capacity(std::fs::metadata(path.clone()).unwrap().len() as usize);
	fs::File::open(path.clone()).unwrap().read_to_end(&mut str)?;

	let occurances = count_occurances(&mut text, &mut table, size, &str[0..str.len()]);

	println!("Number of times present: {}\n", occurances);

    } else if op == "selfsimilar_parallel" {
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
        //let text = load_text(2);
	let text = filebuffer::FileBuffer::open(env::args().nth(2).unwrap()).unwrap();

	let metadata = fs::metadata(env::args().nth(2).unwrap() + ".table.bin")?;

	assert!(metadata.len() % (text.len() as u64) == 0);
	
	let ratio = metadata.len()/(text.len() as u64);

	assert!(ratio == 8);

	println!("Loading ratio is {}", ratio);

	fn sdf(text:&[u8], start:usize, end:usize) -> usize {
	    let mut table = make_table(env::args().nth(2).unwrap() + ".table.bin", start);
	    let mut prev_location = get_next_82(&mut table);

	    //let mut counts = vec![0u64; 5000];

	    let mut outfile = std::io::BufWriter::new(fs::File::create(
		format!("/tmp/dups_{}_{}-{}", env::args().nth(2).unwrap().split("/").last().unwrap(), start, end)).unwrap());

	    let mut i = start;
	    while i < end {
		if i%1000000000 == 0 { println!("{} / {} ", i-start, end-start); }
		let suf1 = &text[prev_location as usize..];


		/*
		// This block of code generates the data to count the frequency
		// of substring matches
		i += 1;
		let cur_location = get_next_82(&mut table);
		let suf2 = &text[cur_location as usize..];
		let match_len = (0..5000).find(|&j| !(j < suf1.len() && j < suf2.len() && suf1[j] == suf2[j]));
		if let Some(match_len_) = match_len {
		    counts[match_len_] += 1;
		}
		prev_location = cur_location;
		continue;
		// */

		/*
		// Find sequences that occur at least 1000 times in the data

		let step_by = 4000;
		i += step_by+1;
		table.ptr += 8*step_by;
		let cur_location = get_next_82(&mut table);
		let suf2 = &text[cur_location as usize..];

		let does_match =  suf2.len() >= 100 && suf1.len() >= 100 && suf1[..100] == suf2[..100];
		if does_match {
		    if prev_location%2 == 0{
			println!("Match {:?}", &suf1[..100])
		    } else {
			println!("Match {:?}", &suf1[1..101])
		    }
		}
		prev_location = cur_location;
		continue;
		// */
		
		let mut cur_location;

		let mut pairs:Vec<u64> = Vec::with_capacity(4);
		let mut first = true;
		
		loop {
		    cur_location = get_next_82(&mut table);
		    i += 1;

		    let suf2 = &text[cur_location as usize..];
		    let does_match =  suf2.len() >= 100 && suf1.len() >= 100 && suf1[..100] == suf2[..100];
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

		if pairs.len() > 0 {
		    let str:Vec<String> = pairs.iter().map(|&x| format!("{}", x)).collect();
		    outfile.write_all(str.join(" ").as_bytes()).expect("Write ok");
		    outfile.write_all(b"\n").expect("Write ok");
		}

		prev_location = cur_location;
	    }

	    /*
	    let str2:Vec<String> = counts.iter().map(|&x| format!("{}", x)).collect();
	    println!("out {}", str2.join(","));
	     */

	    //let str:Vec<String> = seen.iter().map(|&x| format!("{}", x)).collect();
	    //println!("Matches {}", str.join(", "));
	    //return counts[100..].iter().sum::<u32>() as usize;
	    //println!("Seen {}.\n>{}", seen.len(), str.join("\n>"));
	    return 0;
	}


	let jobs:i64 = 96;
	let increment:i64 = (text.len() as i64-jobs)/jobs;
	let _answer = crossbeam::scope(|scope| {
	    let mut result = Vec::with_capacity(jobs as usize);
	    let text = &text;
	    for i in 0..jobs {
		let one_result = scope.spawn(move || {
		    return sdf(text,
			       std::cmp::max(0i64,i*increment-1) as usize,
			       std::cmp::min(((i+1)*increment) as usize, text.len()));
			       //sizes_cumsum);
		});
		result.push(one_result);
	    }

	    let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
	    println!("Final answer {:?}", thread_sum);
        
	});

    } else if op == "similar_parallel" {
	let text1 = filebuffer::FileBuffer::open(env::args().nth(2).unwrap()).unwrap();
	let text2 = filebuffer::FileBuffer::open(env::args().nth(3).unwrap()).unwrap();
        //let text1 = load_text(2);
        //let text2 = load_text(3);

	let metadata1 = fs::metadata(env::args().nth(2).unwrap() + ".table.bin")?;
	let metadata2 = fs::metadata(env::args().nth(3).unwrap() + ".table.bin")?;

	assert!(metadata1.len() % (text1.len() as u64) == 0);
	let ratio = metadata1.len()/(text1.len() as u64);
	assert!(ratio == 8);

	assert!(metadata2.len() % (text2.len() as u64) == 0);
	let ratio = metadata2.len()/(text2.len() as u64);
	assert!(ratio == 8);

	fn sdf(text1:&[u8], text2:&[u8],
	       start1:usize, end1:usize,
	       start2:usize, end2:usize) -> usize {
	    let mut table1 = make_table(env::args().nth(2).unwrap() + ".table.bin", start1);
	    let mut location1 = get_next_82(&mut table1);

	    let mut table2 = make_table(env::args().nth(3).unwrap() + ".table.bin", start2);
	    let mut location2 = get_next_82(&mut table2);
	    
	    let mut outfile = std::io::BufWriter::new(fs::File::create(
		format!("/tmp/dups_{}_{}-{}_{}_{}-{}",
			env::args().nth(2).unwrap().split("/").last().unwrap(), start1, end1,
			env::args().nth(3).unwrap().split("/").last().unwrap(), start2, end2,
		)).unwrap());

	    let mut i = start1;
	    let mut j = start2;
	    while i < end1 && j < end2 {
		if (i+j)%1000000000 == 0 { println!("{} / {} ", i, text1.len()); }
		
		let mut suf1 = &text1[location1 as usize..];
		let mut suf2 = &text2[location2 as usize..];


		let matchlen = 100;
		let does_match = suf1.len() >= matchlen && suf2.len() >= matchlen && suf1[..matchlen] == suf2[..matchlen];

		if does_match {
		    // We have a match between a subsequence in text1 and text2
		    let target_suf = &suf1[..matchlen]; // wlog. equals suf2[..matchlen]
		    while suf1.len() >= matchlen && &suf1[..matchlen] == target_suf {
			location1 = get_next_82(&mut table1);
			suf1 = &text1[location1 as usize..];
			i += 1;
		    }

		    while suf2.len() >= matchlen && &suf2[..matchlen] == target_suf {
			outfile.write_all(format!(" {}",location2).as_bytes())
			    .expect("Write ok");
			
			location2 = get_next_82(&mut table2);
			suf2 = &text2[location2 as usize..];
			j += 1;
		    }
		    outfile.write_all("\n".as_bytes())
			.expect("Write ok");
		} else if suf1 < suf2 {
		    i += 1;
		    location1 = get_next_82(&mut table1);
		} else if suf2 < suf1 {
		    j += 1;
		    location2 = get_next_82(&mut table2);
		} else {
		    // This happens only when
		    // 1. The two suffixes are identical
		    // 2. But they're not yet long enough for it to "count"
		    // so we just increment one of the poitners WLOG
		    assert!(&suf1 == &suf2);
		    assert!(suf1.len() < 100 || suf2.len() < 100);
		    i += 1;
		    location1 = get_next_82(&mut table1);
		}
	    }

	    return 0;
	}


	let jobs:i64 = 96;
	let increment:i64 = (text1.len() as i64-jobs)/jobs;
	let _answer = crossbeam::scope(|scope| {
	    let mut result = Vec::with_capacity(jobs as usize);
	    let text1 = &text1;
	    let text2 = &text2;
	    let mut last_end = 0;
	    for i in 0..jobs {
		let a = std::cmp::max(0i64,i*increment-1) as usize;
		let b = std::cmp::min(((i+1)*increment) as usize, text1.len());
		
		let mut table1 = std::io::BufReader::new(fs::File::open(env::args().nth(2).unwrap() + ".table.bin").unwrap());
		let mut table2 = std::io::BufReader::new(fs::File::open(env::args().nth(3).unwrap() + ".table.bin").unwrap());
		let this_start = last_end;
		
		let end_seq = &text1[table_load_disk(&mut table1, b)..];
		let this_end = off_disk_position(text2, &mut table2, end_seq);
		
		last_end = this_end;
		println!("start {} {}", this_start, this_end);
		let one_result = scope.spawn(move || {
		    
		    return sdf(text1, text2,
			       a, b,
			       this_start, this_end);
		});
		result.push(one_result);
	    }

	    let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
	    println!("Final answer {:?}", thread_sum);
            
	});
	
    } else if op == "merge_parallel" {
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
	let nn:usize = env::args().len()-3;

	let texts:Vec<Vec<u8>> = (0..nn).map(|x| load_text(x+2)).collect();

	let texts_len:Vec<usize> = texts.iter().enumerate().map(|(i,x)| x.len() - (if i+1 == texts.len() {0} else {HACKSIZE})).collect();


	let metadatas:Vec<u64> = (0..nn).map(|x| {
	    let meta = fs::metadata(env::args().nth(x+2).unwrap() + ".table.bin").unwrap();
	    assert!(meta.len()%(texts[x].len() as u64) == 0);
	    return meta.len();
	}).collect();

	let ratio = metadatas[0] / (texts[0].len() as u64);
	assert!(ratio == 8);

	println!("Loading ratio is {}", ratio);
	
	fn sdf(texts:&Vec<Vec<u8>>, starts:Vec<usize>, ends:Vec<usize>, texts_len:Vec<usize>, part:usize) {

	    let nn = texts.len();
	    let mut tables:Vec<BufReader<File>> = (0..nn).map(|x| {
		std::io::BufReader::new(fs::File::open(env::args().nth(x+2).unwrap() + ".table.bin").unwrap())
	    }).collect();

	    for i in 0..starts.len() {
		let x = starts[i];
		tables[i].seek (std::io::SeekFrom::Start ((x*8) as u64)).expect ("Seek failed!");
	    }
	    
	    let mut cache = vec![[0u8; 1024*128]; nn];
	    let mut cacheptr = vec![1024*128; nn];
	    
	    let mut idxs:Vec<u64> = starts.iter().map(|&x| x as u64).collect();
	    
	    let delta:Vec<u64> = (0..nn).map(|x| {
		let pref:Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect();
		pref.iter().sum::<u64>() - (HACKSIZE * x) as u64
	    }).collect();

	    let foutpath = env::args().last().unwrap();
            let mut next_table = std::io::BufWriter::new(File::create(format!("{}.table.bin.{:04}", foutpath.clone(), part)).unwrap());

	    fn get_next_maybe_skip(mut file:&mut BufReader<File>, mut cache:&mut [u8],
				   mut ptr:&mut usize,
				   index:&mut u64, thresh:usize, cond: bool) -> u64 {
		let mut location = get_next_8(&mut file, &mut cache, &mut ptr);
		*index += 1;
		while cond && location >= thresh as u64 {
		    location = get_next_8(&mut file, &mut cache, &mut ptr);
		    *index += 1;
		}
		return location;
	    }
	    
	    let mut heap = BinaryHeap::new();

	    for x in 0..nn {
		let position = get_next_maybe_skip(&mut tables[x], &mut cache[x], &mut cacheptr[x],
						   &mut idxs[x], texts_len[x],
						   true);
		heap.push(State {
		    suffix: &texts[x][position as usize..],
		    position: position,
		    table_index: x
		});
	    }
	    

	    let mut prev_position = 0;
	    let mut prev = &texts[0][0..];
	    while let Some(State {suffix: _suffix, position, table_index}) = heap.pop() {
		next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()).expect("Write OK");

		let position = get_next_maybe_skip(&mut tables[table_index], &mut cache[table_index],
						   &mut cacheptr[table_index],
						   &mut idxs[table_index], texts_len[table_index],
						   true);
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
		    
		    heap.push(State {
			suffix: &texts[table_index][position as usize..],
			position: position,
			table_index: table_index
		    });
		    prev = next;
		    prev_position = position;
		}
	    }
	}


	let jobs = 96;
	let _answer = crossbeam::scope(|scope| {

	    let mut tables:Vec<BufReader<File>> = (0..nn).map(|x| {
		std::io::BufReader::new(fs::File::open(env::args().nth(x+2).unwrap() + ".table.bin").unwrap())
	    }).collect();

	    let mut starts = vec![0; nn];
	    
	    for i in 0..jobs {
		let texts = &texts;
		let mut ends: Vec<usize> = vec![0; nn];
		if i < jobs-1 {
		    ends[0] = (texts[0].len()+jobs)/jobs*(i+1);
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
			i);
		});

		for j in 0..ends.len() {
		    starts[j] = ends[j];
		}
	    }
	});
	
	println!("Finish writing");
	let foutpath = env::args().last().unwrap();
        let mut buffer = File::create(foutpath)?;
	for i in 0..texts.len()-1 {
            buffer.write_all(&texts[i][..texts[i].len()-HACKSIZE])?;
	}
	buffer.write_all(&texts[texts.len()-1])?;
    } else if op == "find_exact_dups" {
	/* Find sequences that are exact duplicates. */
	let text:Vec<u8> = load_text(2);

	let sizespath = env::args().nth(2).unwrap() + ".size";
	let mut text_ = Vec::with_capacity(std::fs::metadata(sizespath.clone()).unwrap().len() as usize);
	fs::File::open(sizespath.clone()).unwrap().read_to_end(&mut text_).unwrap();
	
	let sizes = from_bytes(fs::read(sizespath.clone()).unwrap());

	fn do_hash(text: &Vec<u8>, sizes: &Vec<u64>, start: usize, end: usize) -> Vec<u64>{
	    let mut out = vec![0u64; end-start];
	    for j in start..end {
		out[j-start] = fasthash::metro::hash64(&text[6+sizes[j] as usize..sizes[j+1] as usize]);
	    }
	    return out;
	}

	let jobs = 96;
	let mut result = Vec::with_capacity(jobs as usize);

	crossbeam::scope(|scope| {
	    for j in 0..jobs {
		let text = &text;
		let sizes = &sizes;
		
		let start = sizes.len()*j/jobs;
		let end = std::cmp::min(sizes.len()*(j+1)/jobs,
					sizes.len()-1);
		let out = scope.spawn(move || {
		    return do_hash(text, sizes, start, end);
		});
		result.push(out)
	    }
	    
	});

	println!("Done parallel!");
	let hashes:Vec<Vec<u64>> = result.into_iter().map(|t| t.join()).collect();

	// hash -> index
	let mut map: HashMap<u64, usize> = HashMap::new();
	let mut offset = 0;
	for hash_list in hashes.iter() {
	    for (i,hash) in hash_list.iter().enumerate() {
		if map.contains_key(hash) {
		    let j = *map.get(hash).unwrap() as usize;
		    println!("Match! {} {} {}", hash, j, i+offset);
		    println!("Compare {}",
			     &text[6+sizes[i+offset] as usize..sizes[i+offset+1] as usize]==
			     &text[6+sizes[j] as usize..sizes[j+1] as usize]);
		} else {
		    //println!("Insert {} {}", hash, i);
		    map.insert(*hash, i+offset);
		}
	    }
	    offset += hash_list.len();
	}
    } else if op == "collect_similar" {

	let ds_name = env::args().nth(2).unwrap();
	let paths = fs::read_dir("/tmp").unwrap();
	
	let mut path_list = Vec::with_capacity(1000);
	for path in paths {
            let path = path.unwrap().path().as_path().to_str().unwrap().to_string();
	    if !path.starts_with(&format!("/tmp/dups_{}_", ds_name.clone())) {
		continue;
	    }
	    path_list.push(path);
	}

	let mut result = Vec::with_capacity(100);
	crossbeam::scope(|scope| {
	    for path in path_list.into_iter() {
		let path = path.clone();
		let out = scope.spawn(move || {
		    let lines = std::io::BufReader::new(File::open(path).unwrap()).lines();
		    let mut all_items:Vec<u64> = Vec::with_capacity(1000);
		    for line in lines {
			let line = line.unwrap(); 
			let line:Vec<&str> = line.trim().split(" ").collect();
			let mut line:Vec<u64> = line.iter().map(|x| (*x).parse::<u64>().unwrap()).collect();
			all_items.append(&mut line)
		    }
		    all_items.sort_unstable();
		    println!("Done {}", all_items.len());
		    return all_items;
		});
		result.push(out);
	    }
	});
	let outputs:Vec<Vec<u64>> = result.into_iter().map(|t| t.join()).collect();
	let mut all_items:Vec<u64> = outputs.into_iter().flatten().collect();
	println!("Sorting.");
	all_items.sort_unstable();
	println!("Sorted.");
	let mut ranges:Vec<(u64,u64)> = Vec::with_capacity(1000);
	let mut prev_start = all_items[0];
	let mut prev_end = all_items[0]+100;
	for x in all_items[1..].iter() {
	    if *x <= prev_end {
		prev_end = *x+100;
	    } else {
		ranges.push((prev_start, prev_end));
		prev_start = *x;
		prev_end = *x+100;
	    }
	}
	ranges.push((prev_start, prev_end));
	    
	let strout:Vec<String> = ranges.iter().map(|&x| format!("{} {}", x.0, x.1)).collect();
	println!("out {}", strout.join("\n"));

    } else {
	println!("Command `{}` not known.", op)
    }
    
    Ok(())
}
