from torch.utils.data import Dataset
from itertools import product

import consts
import torch
import numpy as np


def generate_data(tokenizer):
    # keeping the raw dataset maybe to reproduce
    # the nice operation tables.
    data = []
    pairs = product(range(consts.P), repeat=2)
    for a, b in pairs:
        # I think we shold not divide by 0
        if b == 0:
            continue
        c = a
        a = (b * c) % consts.P
        a = tokenizer.decode(a)[0]
        b = tokenizer.decode(b)[0]
        c = tokenizer.decode(c)[0]
        data.append([consts.EOS, a, consts.DIV, b, consts.EQ, c, consts.EOS])
    return np.array(data)


class Tokenizer:

    def __init__(self):
        self.symbol_to_id = {symbol: i for i, symbol in enumerate(consts.SYMBOLS)}
        self.id_to_symbol = {i: symbol for symbol, i in self.symbol_to_id.items()}

    def encode(self, symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        return [self.symbol_to_id[s] for s in symbols]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self.id_to_symbol[id] for id in ids]

    @property
    def vocab_size(self):
        return len(self.symbol_to_id)


class BinaryDivisionModDataset(Dataset):

    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx])
        y = x.copy()  # Same as x for causal attention
        return torch.tensor(x), torch.tensor(y)
