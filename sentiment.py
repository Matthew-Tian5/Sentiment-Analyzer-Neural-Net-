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



MAX_LEN = 200

def encode(text):

    #toeknize and trim to 200 words, word 
    tokens = tokenize(text)[:MAX_LEN]

    #IMPORTANT
    #cinvert tokens to indices, use 1 for unknown words and pad with 0s if less than 200
    ids = [word_to_idx.get(t,1) for t in tokens]
    padded = ids +[0] * (MAX_LEN - len(ids))
    return padded

X_train = torch.tensor([encode(ex['text']) for ex in train_data], dtype = torch.long)
y_train = torch.tensor([ex['label'] for ex in train_data], dtype = torch.float)

X_test = torch.tensor([encode(ex['text']) for ex in test_data], dtype = torch.long)
y_test = torch.tensor([ex['label'] for ex in test_data], dtype = torch.float)


print(f"X_train shape: {X_train.shape}")  # should be [25000, 200]
print(f"y_train shape: {y_train.shape}")  # should be [25000]
print(f"\nFirst review encoded: {X_train[0][:20]}")  # first 20 numbers
print(f"First review label: {y_train[0]}")  # 0 = negative, 1 = positive


from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class SentimentNet(nn.Module):
    def __init__(self,vocab_size, embed_dim, hidden_dim):
        super().__init__()

        #converts each word Id into a 64 number vector that captures meaining
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0
                                      )
        #fully connceted layers that learn patterns from those vectors
        self.fcl = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        #activation functino that lets the network learn non linear patterns
        self.relu = nn.ReLU()

        #randomly turns off 30% of neurons during trainingg to prevent memorization
        self.dropout = nn.Dropout(0.3)
        
        #sigmoid squiahes the final outpit to between 0 and 1, which we can interpret as probability of being positive
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        out = self.fcl(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze()
    


#Create model
model = SentimentNet(vocab_size=10000, embed_dim=64, hidden_dim=128).to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")