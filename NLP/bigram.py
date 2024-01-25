import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 2   # what is the maximum context length for predictions?
MAX_ITERS = 20000
EVAL_INTERVAL = 2000
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 2000
# ------------

torch.manual_seed(42)

with open('dataset/names.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    """
    This function generates a small batch of data of inputs x and targets y.
    The split parameter determines whether to use training data or validation data.
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    This function estimates the loss on the train and validation sets.
    It uses the model in evaluation mode and calculates the mean loss 
    over a number of iterations.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """
    This class defines a simple bigram language model using PyTorch's nn.Module.
    It uses an embedding table to map each token to the logits for the next token.
    """
    def __init__(self, vocab_size):
        """
        The constructor for the BigramLanguageModel class.
        It initializes the token embedding table, each token directly 
        reads off the logits for the next token from a lookup table.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        It calculates the logits and, if targets are provided, the loss.
        idx and targets are both (B,T) tensor of integers.
        """
        logits = self.token_embedding_table(idx) # (B,T,C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens based on the current context.
        idx is (B, T) array of indices in the current context.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)     # get predictions
            logits = logits[:, -1, :] # take last token of size (B, C)
            probs = F.softmax(logits, dim=-1) # create probability of size (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution of size (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for iter in range(MAX_ITERS):

    # every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, DEVICE=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))