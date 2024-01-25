import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 16
BLOCK_SIZE = 32
MAX_ITERS = 100000
EVAL_INTERVAL = 10000
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2

torch.manual_seed(42)

with open("dataset\input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a mapping from characters to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))
train_data = data[:n1]  # 80%
val_data = data[n1:n2]  # 10%
test_data = data[n2:]  # 10%


def get_batch(split):
    """
    Generates a small batch of data of inputs x and targets y.

    Args:
        split (str): The type of data split. Can be either 'train' or 'val'.

    Returns:
        x, y (torch.Tensor, torch.Tensor): The input and target tensors.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimates the loss for the model.

    Returns:
        out (dict): A dictionary with the mean loss for the 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    A head of self-attention.
    """

    def __init__(self, head_size):
        """
        Initializes the Head module.

        Args:
            head_size (int): The size of the head.
        """
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        # we are assigning this tril to become a parameter of
        # `Head` because we want to backprop effectively.
        self.DROPOUT = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Defines the forward pass for the Head module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            out (torch.Tensor): The output tensor.
        """
        # creating an attention matrix
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.DROPOUT(wei)
        # perform the weighted aggregation of the values
        # basically this wei Tensor is our
        # `ATTENTION` tensor which multiplied with value gives output vector
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, num_heads, head_size):
        """
        Initializes the MultiHeadAttention module.

        Args:
            num_heads (int): The number of heads.
            head_size (int): The size of each head.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # creating multiple heads and in each head we perform
        # similar operations (attentions) on different batches parallely
        # and concatenate these results
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.DROPOUT = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # concatenating over channel dimension
        out = self.DROPOUT(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, N_EMBD):
        """
        Initializes the FeedFoward module.

        Args:
            N_EMBD (int): The embedding dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),  # layers like these work better
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, N_EMBD, N_HEAD):
        """
        Initializes the Block module.

        Args:
            N_EMBD (int): The embedding dimension.
            N_HEAD (int): The number of heads.
        """
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedFoward(N_EMBD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer_text(nn.Module):
    """
    A Transformer module that uses only the decoder architecture.
    """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD=N_HEAD) for _ in range(N_LAYER)]
        )  # unpacking using list comprehension
        self.ln_f = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        """
        Defines the forward pass for the Transformer module.

        Args:
            idx (torch.Tensor): The input tensor.
            targets (torch.Tensor, optional): The target tensor.
            If provided, the loss will also be calculated.

        Returns:
            logits (torch.Tensor): The output logits.
            loss (torch.Tensor, optional): The calculated loss.
            Only returned if targets is not None.
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens.

        Args:
            idx (torch.Tensor): The input tensor.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            idx (torch.Tensor): The input tensor with the newly
            generated tokens appended.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Transformer_text()
m = model.to(DEVICE)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    # every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
