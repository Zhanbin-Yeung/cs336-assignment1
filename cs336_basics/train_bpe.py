import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
from collections import Counter, defaultdict
from typing import Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
import time
import heapq

import cProfile
import pstats
import io
from pathlib import Path
import argparse

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPE_PAT: Optional[re.Pattern] = None
PRE_PAT: Optional[re.Pattern] = None
CACHED_SPECIAL_TOKENS: Optional[tuple[str, ...]] = None

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pattern_compile(special_token:list[str] = None):
    global SPE_PAT, PRE_PAT, CACHED_SPECIAL_TOKENS
    st = tuple(special_token) if special_token else ()

    if SPE_PAT is None or PRE_PAT is None or CACHED_SPECIAL_TOKENS != st:
        escaped = [re.escape(t) for t in special_token] if special_token else []
        SPE_PAT = re.compile("|".join(escaped))
        PRE_PAT = re.compile(PAT)
        CACHED_SPECIAL_TOKENS = st

def vocab_init ():
    vocab = {idx: bytes([idx]) for idx in range(256)}
    return vocab


def process_chunk(file_path: str, start: int, end: int, special_token: list[str] = None)->Counter[str]:
    pattern_compile(special_token)
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    pos = 0
    count:Counter[str] = Counter()
    
    for match in SPE_PAT.finditer(chunk):
        segment = chunk[pos: match.start()]
        count.update(PRE_PAT.findall(segment))
        # count.update(m.group() for m in PRE_PAT.finditer(segment))
        # for m in PRE_PAT.finditer(segment):
        #     count[m.group()] += 1
        pos = match.end()
    count.update(PRE_PAT.findall(chunk[pos:]))
    # count.update(m.group() for m in PRE_PAT.finditer(chunk[pos:]))
    
    return count

def process_chunk_profiled(
    file_path: str,
    start: int,
    end: int,
    special_token: list[str] = None,
    profile: bool = False,
    prof_dir: str = "prof",
) -> Counter[str]:
    """
    多进程 worker 内 profile：
    - 每个子进程 dump 一个 .prof
    - tag 用来区分 chunk
    """
    if not profile:
        return process_chunk(file_path, start, end, special_token)

    Path(prof_dir).mkdir(parents=True, exist_ok=True)

    out_path = os.path.join(prof_dir, f"worker.prof")

    pr = cProfile.Profile()
    pr.enable()
    out = process_chunk(file_path, start, end, special_token)
    pr.disable()
    pr.dump_stats(out_path)
    return out

def pre_tokenize_parallel(file_path, special_token, profile_workers: bool = False, prof_dir: str = "prof")->Counter:
    num_processes = os.cpu_count()
    num_chunks = num_processes * 8
    pre_token_count:Counter[tuple[bytes, ...]] = Counter()

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        
        with ProcessPoolExecutor(num_processes) as ex:
            futures = [
                ex.submit(
                    process_chunk_profiled,
                    file_path,
                    start,
                    end,
                    special_token,
                    profile=(profile_workers and i == 0),
                    prof_dir=prof_dir,
                )
                    for i, (start, end) in enumerate(ranges)
            ]

            global_count:Counter[str] = Counter()
            for future in as_completed(futures):
                global_count.update(future.result())
        
        ## multiprocess Pool 的写法
        # tasks = [(s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]
        # ctx = mp.get_context("spawn")
        # with ctx.Pool(num_processes, initializer=pattern_compile, initargs=(file_path, special_token)) as pool:
        #     counters = pool.map(process_chunk, tasks)
        #     for counter in counters:
        #         global_count.update(counter)

        for key, value in global_count.items():
            b = key.encode('utf-8')
            # b_tuple = tuple(bytes([x]) for x in b)
            pre_token_count[tuple(b)] = value

    return pre_token_count

def pre_tokenize_serial(file_path: str, special_token: list[str], num_chunks: int = 8) -> Counter[str]:
    escaped = [re.escape(t) for t in special_token]
    spe_pat = re.compile("|".join(escaped))
    pre_pat = re.compile(PAT)

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        global_count: Counter[str] = Counter()
        pre_token_count = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            pos = 0
            for match in spe_pat.finditer(chunk):
                segment = chunk[pos:match.start()]
                # global_count.update(m.group(0) for m in pre_pat.finditer(segment))
                toks = pre_pat.findall(segment)
                global_count.update(toks)
                # for m in pre_pat.finditer(segment):
                #     global_count[m.group()] += 1
                pos = match.end()

            # for m in pre_pat.finditer(chunk[pos:]):
            #         global_count[m.group()] += 1
            global_count.update(pre_pat.findall(chunk[pos:]))
            # global_count.update(m.group(0) for m in pre_pat.finditer(chunk[pos:]))

    
    for key, value in global_count.items():
        b = key.encode('utf-8')
        # b_tuple = tuple(bytes([x]) for x in b)
        pre_token_count[tuple(b)] = value

    return pre_token_count


def get_pair(bucket:dict[int, set[tuple[int, int]]], vocab: dict[int, bytes], max_freq):
    while max_freq > 0 and (max_freq not in bucket or not bucket[max_freq]):
        max_freq -= 1
    
    if max_freq <= 0 :
        return None, 0
    
    pair = max(bucket[max_freq], key=lambda p: (vocab[p[0]], vocab[p[1]]))
    bucket[max_freq].remove(pair)

    if not bucket[max_freq]:
        del bucket[max_freq]
        
    return pair, max_freq

def merge(old_word:tuple[int, ...], old_word_freq, pair:tuple[int, int], new_id):
    new_word = []
    local_delta: Counter[tuple[int, int]] = Counter()
    n = len(old_word) 
    i = 0
    while i < n :
        if i < n - 1 and (old_word[i], old_word[i+1]) == pair:
            new_word.append(new_id)

            if i > 0 :
              old_p1 = (old_word[i - 1], old_word[i])
              local_delta[old_p1] -= old_word_freq
              
            old_p2 = (old_word[i], old_word[i + 1])
            local_delta[old_p2] -= old_word_freq

            if(len(new_word) > 1):
                new_p1 = (new_word[-2], new_word[-1])
                local_delta[new_p1] += old_word_freq

            i += 2
            if(i < n and old_word[i: i+2] != pair):
                old_p3 = (old_word[i - 1], old_word[i])
                local_delta[old_p3] -= old_word_freq
                new_p2 = (new_word[-1], old_word[i])
                local_delta[new_p2] += old_word_freq

        else:
            new_word.append(old_word[i])
            i += 1
    new_word = tuple(new_word)
        
     # === 差集更新 pairs_to_words 用 ===
    # 用 set（不是 Counter）即可：pairs_to_words 只需要“这个 word 是否包含该 pair”
    if len(old_word) >= 2:
        old_pairs_set: Set[tuple[int, int]] = set(zip(old_word[:-1], old_word[1:]))
    else:
        old_pairs_set = set()
    if len(new_word) >= 2:
        new_pairs_set: Set[tuple[int, int]] = set(zip(new_word[:-1], new_word[1:]))
    else:
        new_pairs_set = set()

    removed_pairs = old_pairs_set - new_pairs_set
    added_pairs = new_pairs_set - old_pairs_set

    return new_word, local_delta, removed_pairs, added_pairs



def compute_merge(pre_token_dict:Counter[tuple[int, ...]], nums_merge: int, vocab: dict[int, bytes]):

    pairs_counts: Counter[tuple[int, int]] = Counter()
    pairs_to_words:dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    bucket:dict[int, set[tuple[int, int]]] = defaultdict(set)

    for pre_token, freq in pre_token_dict.items():
        if len(pre_token) < 2:
            continue
        for pair in zip(pre_token[:-1], pre_token[1:]):
            pairs_to_words[pair].add(pre_token)
            pairs_counts[pair] += freq

    max_freq = -1
    for pair, freq in pairs_counts.items():
        max_freq = max(max_freq, freq)
        bucket[freq].add(pair)

    merges:list[tuple[bytes, bytes]] = []
    for step in range(nums_merge):
        new_id = step + 256
        pair, max_freq = get_pair(bucket, vocab, max_freq)
        if pair is None:
            break 
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merge_pair = (vocab[pair[0]], vocab[pair[1]])
        merges.append(merge_pair)

        global_delta:Counter[tuple[int, int]] = Counter()
        words = list(pairs_to_words[pair])
        for word in words:
            word_freq = pre_token_dict[word]
            new_word, local_delta, removed_pairs, added_pairs = merge(word, word_freq, pair, new_id)

            # 只更新发生变化的 pair
            for p in removed_pairs:
                pairs_to_words[p].discard(word)
            for p in added_pairs:
                pairs_to_words[p].add(new_word)

            global_delta.update(local_delta)
            pre_token_dict[new_word] = pre_token_dict.get(new_word, 0) + word_freq
            del pre_token_dict[word]
        
        for p, freq in global_delta.items():
            old = pairs_counts.get(p, 0)
            new = old + freq

            if old > 0:
                bucket[old].discard(p)
                if not bucket[old]:
                    del bucket[old]
            if new > 0:
                pairs_counts[p] = new
                bucket[new].add(p)
            else:
                # new <= 0：彻底删除
                if p in pairs_counts:
                    del pairs_counts[p]
        
        # pairs_counts.update(global_delta)

    return vocab, merges

def save_vocab(vocab: dict[int, bytes], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for idx, b in vocab.items():
            f.write(f"{idx}\t{b.hex()}\n")

def save_merges(merges: list[tuple[bytes, bytes]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#version: 1\n")
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None, 
              profile_workers: bool = False, prof_dir: str = "prof"):
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab = vocab_init()
    t0 = time.time()
    pre_token_dict = pre_tokenize_parallel(input_path, special_tokens, profile_workers, prof_dir)
    t1 = time.time() 
    print(f"[INFO] Pre-tokenization finished in {t1 - t0:.3f}s")
    vocab, merges = compute_merge(pre_token_dict, num_merges, vocab)
    t2 = time.time()
    print(f"[INFO] Merge computation finished in {t2 - t1:.3f}s")
    print(f"[INFO] Total training time: {t2 - t0:.3f}s")
    n = len(vocab)
    if special_tokens is not None:
        for s in special_tokens:
            vocab[n] = s.encode('utf-8')
            n += 1
    save_vocab(vocab, "../data/owt_vocab.txt")
    save_merges(merges, "../data/owt_merges.txt")

    return vocab, merges

def profile_main(fn, out_path="prof_main.prof", topn=40, sortby="cumtime"):

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    out = fn()
    pr.disable()
    pr.dump_stats(out_path)

    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby).print_stats(topn)
    print(s.getvalue())
    print(f"[INFO] main profile saved to {out_path}")
    return out

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-main", "-pm",action="store_true")
    parser.add_argument("--profile-workers","-pw", action="store_true")
    parser.add_argument("--prof-dir", type=str, default="prof")
    args = parser.parse_args()

    if args.profile_main or args.profile_workers:
        Path(args.prof_dir).mkdir(parents=True, exist_ok=True)

    input_path = "../data/owt_train.txt"
    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    def runner():
        return train_bpe(
            input_path,
            vocab_size,
            special_tokens,
            profile_workers=args.profile_workers,
            prof_dir=args.prof_dir,
        )

    if args.profile_main:
        vocab, merges = profile_main(
            runner,
            out_path=f"{args.prof_dir}/prof_main.prof",
            topn=50,
        )
    else:
        vocab, merges = runner()