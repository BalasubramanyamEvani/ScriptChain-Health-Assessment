"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Dummy Dataset
"""
import random
import string
import torch
from typing import Any
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, num_texts, seq_len, dictionary) -> None:
        """
        Dummy dataset for text classification.

        Parameters:
        - num_texts (int): The number of texts in the dataset. 
            (Ideally should be words but in this case I am randomly picking ascii letter)
        - seq_len (int): The length of each text sequence.
        - dictionary (dict): The dictionary mapping characters to indices.
        """
        self.texts = []
        for _ in range(num_texts):
            self.texts.append(
                [random.choice(string.ascii_letters) for _ in range(seq_len)]
            )
        self.dictionary = dictionary
        self.length = len(self.texts)

    def tokenize(self, text):
        """
        Tokenize a text sequence by mapping characters to their corresponding indices.

        Parameters:
        - text (list): The input text sequence.

        Returns:
        - list: The tokenized text sequence.
        """

        return [self.dictionary[sym] for sym in text]

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        - int: The length of the dataset.
        """
        return self.length

    def __getitem__(self, index) -> Any:
        """
        Get a sample from the dummy dataset at the specified index.
        (Ideally the text sequences here would be not uniform and one would need
        to define a custom collate function but for simplicity all sequences are uniform
        in this dummy dataset)
        Parameters:
        - index (int): The index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing the text and label of the sample.
        """
        text = self.texts[index]
        text = torch.tensor(self.tokenize(text)).long()
        label = torch.tensor(random.choice([0, 1])).float()
        return text, label
