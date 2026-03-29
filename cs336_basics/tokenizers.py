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
import numpy as np
from array import array
from functools import partial
import shutil

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
        # --- vocab: id \t hex_bytes ---
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as f:
            for line in f:
                line = line.rstrip(b"\n")
                if not line:
                    continue
                idx_str, token_hex = line.split(b"\t", 1)
                vocab[int(idx_str)] = bytes.fromhex(token_hex.decode("utf-8"))

        # --- merges: two hex tokens per line ---
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#version:"):
                    continue
                parts = line.rstrip("\n").split()
                if len(parts) != 2:
                    continue

                left = bytes.fromhex(parts[0])
                right = bytes.fromhex(parts[1])
                merges.append((left, right))

        # --- sanity check ---
        vocab_inv = {v: k for k, v in vocab.items()}
        bad = 0
        for a, b in merges[:2000]:
            if a not in vocab_inv or b not in vocab_inv:
                bad += 1
                if bad >= 5:
                    break
        if bad:
            raise ValueError("Loaded merges contain tokens not found in vocab.")

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

def process_chunk(start: int, end: int, out_file:str, file_path: str, vocab_file, merges_file, special_tokens: list[str] = None):
    
    pattern_compile(special_tokens)
    buf = array("H")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens)
    with open(file_path, "rb") as f, open(out_file, "wb") as out:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
        pos = 0
        buf_tokens: int = 4_000_000
        encode = tokenizer.encode

        for m in SPE_PAT.finditer(chunk):
            segment = chunk[pos : m.start()]
            ids = encode(segment)
            buf.extend(ids)
            eot_id = tokenizer.vocab_inv[m.group().encode("utf-8")]
            buf.append(eot_id)
            if len(buf) >= buf_tokens:
                buf.tofile(out)
                buf = array("H")
            pos = m.end()

        tail = chunk[pos:]
        if tail:
            buf.extend(encode(tail))

        if buf:
            buf.tofile(out)
 
    return


def encode_parallel(file_path, vocab_file, merges_file, special_tokens,):
    num_process = os.cpu_count()
    num_chunks = num_process * 8

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        ranges = list(zip(boundaries[:-1], boundaries[1:]))

        with ProcessPoolExecutor(num_process) as ex:
            futures  = []
            process_fun = partial(process_chunk, 
                                  file_path=file_path,
                                  vocab_file = vocab_file , 
                                  merges_file = merges_file ,
                                  special_tokens = special_tokens )
            for i, (s, e) in enumerate(ranges):
                out_file = f"../data/val_{i:03d}.bin"
                futures.append(
                    ex.submit(process_fun, s, e, out_file)
                )

            for fut in as_completed(futures):
                fut.result()
        
        with open("../data/owt_val.bin", "wb") as out:
            for i in range(len(ranges)):
                tmp_file = f"../data/val_{i:03d}.bin"
                with open(tmp_file, "rb") as f:
                    shutil.copyfileobj(f, out)
                os.remove(tmp_file)


def main():
    special_tokens = ["<|endoftext|>"]
    input_file = "../data/owt_valid.txt"
    vocab_file = "../data/owt_vocab.txt"
    merge_file = "../data/owt_merges.txt"
    st = time.time()
    encode_parallel(input_file, vocab_file, merge_file,special_tokens)
    ed = time.time()
    print(f"Encoding completed in {ed - st:.2f} seconds.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()



