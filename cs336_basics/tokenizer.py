import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
from collections import Counter, defaultdict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
import time
import heapq

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

def pre_tokenize_parallel(file_path, special_token)->Counter:
    num_processes = os.cpu_count()
    num_chunks = num_processes * 8
    pre_token_count:Counter[tuple[bytes, ...]] = Counter()
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        with ProcessPoolExecutor(num_processes) as ex:
            futures = [
                ex.submit(process_chunk, file_path, start, end, special_token) 
                for start, end in ranges
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
        if i < n - 1 and old_word[i: i+2] == pair:
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
        
    return new_word, local_delta


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
            new_word, local_delta = merge(word, word_freq, pair, new_id)

            for p in zip(word[:-1], word[1:]):
                pairs_to_words[p].discard(word)
            for p in zip(new_word[:-1], new_word[1:]):
                pairs_to_words[p].add(new_word)
            
            global_delta.update(local_delta)
            pre_token_dict[new_word] = pre_token_dict.get(new_word, 0) + word_freq
            del pre_token_dict[word]
        
        for p, freq in global_delta.items():
            old = pairs_counts.get(p, 0)
            new = old + freq

            if old > 0:
                bucket[old].discard(p)
            if new > 0:
                bucket[new].add(p)
        
        pairs_counts.update(global_delta)
        
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

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None):
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab = vocab_init()
    t0 = time.time()
    pre_token_dict = pre_tokenize_parallel(input_path, special_tokens)
    t1 = time.time() 
    print(f"[INFO] Pre-tokenization finished in {t1 - t0:.3f}s")
    vocab, merges = compute_merge(pre_token_dict, num_merges, vocab)
    t2 = time.time()
    print(f"[INFO] Merge computation finished in {t2 - t1:.3f}s")
    n = len(vocab)
    if special_tokens is not None:
        for s in special_tokens:
            vocab[n] = s.encode('utf-8')
            n += 1
    save_vocab(vocab, "vocab.txt")
    save_merges(merges, "merges.txt")
    return vocab, merges

class Tokenizer:

    def __init__ (self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None, cache_size: int = 50000):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PRE_PAT = re.compile(self.PAT)
        self.special_tokens = special_tokens
        if self.special_tokens:
            sts = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sts]
            self.SPE_PAT = re.compile("|".join(escaped))
        else:
            self.SPE_PAT = None

        self.vocab = vocab
        self.vocab_inv = {v: idx for idx, v in vocab.items()}
        self.merges = merges
        self.merge_dict = {m: idx for idx, m in enumerate(merges)}
        self.special_tokens = special_tokens
        self._encode_one_cached = lru_cache(maxsize=cache_size)(self._encode_one_nocache)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # --- vocab: id \t raw_bytes ---
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as f:
            for line in f:
                line = line.rstrip(b"\n")
                if not line:
                    continue
                idx_str, token_bytes = line.split(b"\t", 1)
                vocab[int(idx_str)] = token_bytes

        # --- merges: two tokens per line ---
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#version:"):
                    continue
                parts = line.rstrip("\n").split()
                if len(parts) != 2:
                    continue
                # default: interpret as utf-8 text tokens
                left = parts[0].encode("utf-8")
                right = parts[1].encode("utf-8")
                merges.append((left, right))

        # --- sanity check: merge endpoints must exist in vocab ---
        vocab_inv = {v: k for k, v in vocab.items()}
        bad = 0
        for a, b in merges[:2000]: 
            if a not in vocab_inv or b not in vocab_inv:
                bad += 1
                if bad >= 5:
                    break
        if bad:
            raise ValueError("Loaded merges contain tokens not found in vocab. Check file encoding/format.")

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str):
 
        if not self.special_tokens:
            return self.PRE_PAT.findall(text)

        tokens = []
        pos = 0
        for m in self.SPE_PAT.finditer(text):
            seg = text[pos: m.start()]
            seg_list = self.PRE_PAT.findall(seg)
            tokens.extend(seg_list)
            tokens.append(m.group())
            pos = m.end()
        
        tokens.extend(self.PRE_PAT.findall(text[pos:]))
         
        return tokens


    def _encode_one_nocache(self, token: str) -> tuple[int, ...]:
        # special token
        if self.special_tokens is not None and token in self.special_tokens:
            return (self.vocab_inv[token.encode("utf-8")],)

        tks = token.encode("utf-8")
        symbols = [bytes([b]) for b in tks]
        n = len(symbols)
        if n == 0:
            return ()
        if n == 1:
            return (self.vocab_inv[symbols[0]],)

        pre = [idx for idx in range(-1, n - 1)]
        nxt = [idx for idx in range(1, n)]
        nxt.append(-1)

        heap = []
        for i in range(n - 1):
            curr_pair = (symbols[i], symbols[i + 1])
            r = self.merge_dict.get(curr_pair)
            if r is not None:
                heapq.heappush(heap, (r, i))

        if not heap:
            return tuple(self.vocab_inv[s] for s in symbols)

        alive = [True] * n
        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1:
                continue

            curr_rank = self.merge_dict.get((symbols[i], symbols[j]))
            if curr_rank is None or curr_rank != rank:
                continue

            symbols[i] += symbols[j]
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                pre[nxt[j]] = i

            alive[j] = False
            nxt[j] = -1
            pre[j] = -1

            if pre[i] != -1:
                p1 = (symbols[pre[i]], symbols[i])
                r1 = self.merge_dict.get(p1)
                if r1 is not None:
                    heapq.heappush(heap, (r1, pre[i]))

            if nxt[i] != -1:
                p2 = (symbols[i], symbols[nxt[i]])
                r2 = self.merge_dict.get(p2)
                if r2 is not None:
                    heapq.heappush(heap, (r2, i))

        out = []
        i = 0
        while i != -1:
            if alive[i]:
                out.append(self.vocab_inv[symbols[i]])
            i = nxt[i]
        return tuple(out)
        
    def _apply_merge(self, tokens_list: list[str]) -> list[int]:
        token_ids: list[int] = []
        for token in tokens_list:
            token_ids.extend(self._encode_one_cached(token))
        return token_ids
            
            
    def _encode(self, text: str) -> list[int]:
        if(text == ""):
            return []
        tokens = self._pre_tokenize(text)
        return self._apply_merge(tokens)

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = []
        if ids is None:
            return ''

        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

        
# ① 没处理空输入 / 空 token
# ② special_tokens 为空时，SPE_PAT 是空正则
# ③ 没把 special token 做“最长优先匹配”
# ④ decode 在 token 粒度做 UTF-8 解码
# ⑤ decode([single_id]) 用 strict UTF-8 会抛异常

def main():
    special_tokens = ["<|endoftext|>"]
    file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000

    print("[INFO] Start BPE training")
    print(f"[INFO] input={file_path}, vocab_size={vocab_size}")

    t0 = time.perf_counter()
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    t1 = time.perf_counter()

    print("[INFO] BPE training finished")
    print(f"[INFO] time = {t1 - t0:.3f}s")
    print(f"[INFO] vocab size = {len(vocab)}")
    print(f"[INFO] num merges = {len(merges)}")

    assert len(vocab) == 256 + len(merges) + len(special_tokens), "vocab size mismatch"
    print("[INFO] sanity check passed")

if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()


# if __name__ == "__main__":
#     main()

