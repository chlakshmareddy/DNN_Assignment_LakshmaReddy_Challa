import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def build_vocab(captions, min_freq=1):
    from collections import Counter
    counter = Counter()

    for cap in captions:
        words = cap.lower().split()
        counter.update(words)

    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


class FlickrDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=30, feat_dir="./img_features/"):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len
        self.feat_dir = feat_dir

    def __len__(self):
        return len(self.data)

    def numericalize(self, text):
        words = text.lower().split()
        ids = [self.vocab["<bos>"]]

        for w in words:
            ids.append(self.vocab.get(w, self.vocab["<unk>"]))

        ids.append(self.vocab["<eos>"])

        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        feat_path = os.path.join(self.feat_dir, row["image"] + ".npy")
        img_feat = torch.tensor(np.load(feat_path)).float()  # (2048,)

        caption_ids = self.numericalize(row["caption"])

        return img_feat, caption_ids


def collate_fn(batch):
    feats, caps = zip(*batch)
    feats = torch.stack(feats)
    caps = torch.stack(caps)
    return feats, caps
