import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Set the seed for random number generation
random.seed(42)
torch.manual_seed(42)

# Hyperparameters
BATCH_SIZE = 32     # how many independent sequences will we process in parallel?
BLOCK_SIZE = 8      # what is the maximum context length for predictions?
N_EMBD = 24         # the dimensionality of the character embedding vectors
N_HIDDEN = 128      # the number of neurons in the hidden layer of the MLP
MAX_ITERS = 1001
EVAL_INTERVAL = 200
LEARNING_RATE = 1e-2
EVAL_ITERS = 200

# Load and preprocess the data
with open('dataset/names.txt', 'r', encoding='utf-8') as f:
    text = f.read().splitlines()

random.shuffle(text)

# Create a mapping from characters to integers
chars = sorted(list(set(''.join(text))))
vocab_size = len(chars) + 1  # for special character
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = { ch:i for i,ch in stoi.items() }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


def build_dataset(data):
    """
    This function builds the dataset for the model.
    It takes a list of strings and returns input and target tensors.
    """
    X, Y = [], []
    for w in data:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# Split the data into training, validation, and test sets
n1 = int(0.8*len(text))
n2 = int(0.9*len(text))
Xtr,  Ytr  = build_dataset(text[:n1])     # 80%
Xdev, Ydev = build_dataset(text[n1:n2])   # 10%
Xte,  Yte  = build_dataset(text[n2:])     # 10%


@torch.no_grad()
def estimate_loss(split):
    """
    This function estimates the loss on the given split.
    It uses the model in evaluation mode and calculates the cross entropy loss.
    """
    model.eval()
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.train()
    return loss.item()


class FlattenConsecutive(nn.Module):
    """
    This class defines a module that flattens consecutive elements in a tensor.
    It is used to group data inside of a batch.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n  # the value we use for grouping data inside of batch

    def forward(self, x):
        """
        This function flattens consecutive elements in the input tensor.
        It handles 2D and 3D tensors.
        """
        if x.dim() == 2:
            B, T = x.shape
            x = x.view(B, T // self.n, -1)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            return x
        elif x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B, T // self.n, C * self.n)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            return x
        else:
            raise ValueError("Input tensor must be 2 or 3-dimensional.")


class CustomModel(nn.Module):
    def __init__(self, vocab_size, N_EMBD, N_HIDDEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, N_EMBD)
        self.flatten = FlattenConsecutive(2)  # wavenet grouping
        self.linear1 = nn.Linear(N_EMBD * 2, N_HIDDEN, bias=False)
        self.batchnorm = nn.BatchNorm1d(N_HIDDEN)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(N_HIDDEN * 2, N_HIDDEN, bias=False)
        self.linear3 = nn.Linear(N_HIDDEN * 2, N_HIDDEN, bias=False)
        self.output = nn.Linear(N_HIDDEN, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.linear1(x)
        B, T, C = x.shape
        x = self.batchnorm(x.view(B, C, T))  # reshape for batchnorm
        x = self.tanh(x.view(B, T, C))
        x = self.flatten(x)
        x = self.linear2(x)
        B1, T1, C1 = x.shape
        x = self.batchnorm(x.view(B1, C1, T1))  # reshape for batchnorm
        x = self.tanh(x.view(B1, T1, C1))
        x = self.flatten(x)
        x = self.linear3(x)
        # here we have only 2 dimensions so no reshape needed
        x = self.batchnorm(x)
        x = self.tanh(x)
        x = self.output(x)
        return x
    
    def init_weights(self):
        with torch.no_grad():
            # last layer make less confident to get uniform probability
            self.output.weight *= 0.1


model = CustomModel(vocab_size, N_EMBD, N_HIDDEN)
model.init_weights()

# the basic idea is that we are progressively fusing 2 char at first layer
# 2 two-chars at second layer and 2 four-chars at third layer and so on.

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def train_model():
    # Training
    for iter in range(MAX_ITERS):
        # every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            l1 = estimate_loss('train')
            l2 = estimate_loss('val')
            print(f"step {iter}: train loss {l1:.4f}, val loss {l2:.4f}")

        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        xb, yb = Xtr[ix], Ytr[ix]

        # evaluate the loss
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def generate_words(max_new_tokens=20):
    model.eval()
    # sample from the model
    for _ in range(max_new_tokens):

        out = []
        context = [0] * BLOCK_SIZE # initialize with all ...
        while True:
            # forward pass the neural net
            inpt = torch.tensor([context])
            logits = model(inpt)
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break

        print(''.join(itos[i] for i in out)) # decode and print the generated word


command = input("Enter your command: ")
if command.lower() == 'train':
    train_model()
elif command.lower() == 'generate':
    generate_words()
else:
    print("Invalid command. Please enter either 'train' or 'generate'.")