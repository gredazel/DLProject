from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from torch import nn

# Set a random seed in a bunch of different places
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")


def kNucleos(k):
    #Generate a list of all the nucleotides (or, like you prefer, NITROGENOUS BASES) for a given k (fixed lenght, AS YOU ASKED)
    return [''.join(x) for x in product(['A', 'C', 'G', 'T'], repeat=k)]


def one_hot_encode(seq):
    #Given a DNA sequence, return its one-hot encoding
    # Make sure seq has only allowed bases
    allowed = set("ACTG")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGT): {invalid}")

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {'A': [1.0, 0.0, 0.0, 0.0],
             'C': [0.0, 1.0, 0.0, 0.0],
             'G': [0.0, 0.0, 1.0, 0.0],
             'T': [0.0, 0.0, 0.0, 1.0]}

    # Create array from nucleotide sequence
    vec = np.array([nuc_d[x] for x in seq])

    return vec

set_seed(15)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)  # tested on CPU, if runned over GPU will be 'CUDA'

#To train the deep network: We need to take the dataset and for each entry calculate the lenght K (dataset provided has different lenghts for each entry... unfortunately)
#Then for each of it we apply kNucleos as specified below:
seqs8 = kNucleos(8)
print('Total 8nucleotides:',len(seqs8))
#8 is K

#Our matrix will be represented as 4*K (4 rows, one for each possible type of nucleotide ACGT, and K columns as the size considered)
a8 = one_hot_encode("AAAAAAAA")
print("AAAAAAAA:\n",a8)

