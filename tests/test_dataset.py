"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Sanity Check for Dummy Dataset
"""
import sys
from pathlib import Path

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

import torch
from dataset import DummyDataset
from utils import dummy_dictionary


if __name__ == "__main__":
    dictionary = dummy_dictionary()
    dataset = DummyDataset(num_texts=100, seq_len=64, dictionary=dictionary)

    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    print(
        "dataset samples = {}, batches = {}".format(dataset.__len__(), len(dataloader))
    )

    sample_data = next(iter(dataloader))
    print(
        "Sample data: \nX: {}, seq_len: {}, \nY: {}, lenY: {}".format(
            sample_data[0], len(sample_data[0][0]), sample_data[1], len(sample_data[1])
        )
    )
