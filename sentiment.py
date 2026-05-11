import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np

#check for gpu availability and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = load_dataset("imdb")

train_data = dataset['train'] 
test_data = dataset['test']    


print(f"\nLabel: {'positive' if train_data[0]['label'] == 1 else 'negative'}")
print(f"Review: {train_data[0]['text'][:300]}")
print(f"\nTotal training examples: {len(train_data)}")
print(f"Total test examples: {len(test_data)}") 

from collections import Counter
import re
def tokenize(text):
    #change everyting to lowercase and remove html tags, punctions and numbers
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

counter = Counter()

for example in train_data:
    counter.update(tokenize(example['text']))

VOCAB_SIZE = 10000
vocab = ['<pad>','<unk>'] + [ word for word, _ in counter.most_common(VOCAB_SIZE - 2)]

word_to_idx = {word: idx for idx, word in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {vocab[2:12]}")
print(f"Index of 'great': {word_to_idx.get('great', 'not found')}")
print(f"Index of 'terrible': {word_to_idx.get('terrible', 'not found')}")



