from torch.utils.data import Dataset, IterableDataset
import numpy as np
import torch


class Datasets(Dataset):
    def __init__(self, input_file, max_seq_len:int):
        super().__init__()
        self.ids = np.memmap(input_file, dtype=np.uint16, mode='r')
        self.max_len = max_seq_len
        self.N = self.ids.shape[0]
    
    def __len__(self):
        return self.N - self.max_len - 1

    def __getitem__(self, index):
        x = self.ids[index: index + self.max_len]
        y = self.ids[index + 1 : index + 1 + self.max_len]

        return x, y
    
class IterableDatasets(IterableDataset):
    def __init__(self, input_file, max_seq_len:int, batch_size:int, seed=42):
        super().__init__()
        self.token_ids = np.memmap(input_file, dtype=np.uint16, mode='r')
        self.max_len = max_seq_len
        self.N = self.token_ids.shape[0]
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        num_samples = self.N - self.max_len - 1
        indices = np.arange(worker_id, num_samples, num_workers, dtype=np.int64)

        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + worker_id)
        rng.shuffle(indices)
        
        offset = np.arange(self.max_len, dtype=np.int64)
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            idx = indices[i: i + self.batch_size]
            index = idx[:, None] + offset[None, :]
            x = self.token_ids[index]
            y = self.token_ids[index + 1]

            yield torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))