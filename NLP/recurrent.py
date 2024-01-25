import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
BLOCK_SIZE = 32  # length of the input sequences of integers
VOCAB_SIZE = None  # the input integers are in range [0 .. VOCAB_SIZE -1]
BATCH_SIZE = 16
MAX_ITERS = 1000
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
EVAL_ITERS = 100
# parameters below control the sizes of each model slightly differently
N_LAYER = 4
N_EMBD = 64
N_EMBD2 = 64
N_HEAD = 4

torch.manual_seed(42)

with open("dataset\input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a mapping from characters to integers
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
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
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
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


class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """

    def __init__(self):
        super().__init__()
        self.xh_to_h = nn.Linear(N_EMBD + N_EMBD2, N_EMBD2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)  # add rows horizontally
        ht = torch.tanh(self.xh_to_h(xh))
        return ht


class GRUCell(nn.Module):
    """
    A Gated Recurrent Unit (GRU) cell which is a type of recurrent
    neural network (RNN) cell. The GRU cell uses gating mechanisms
    to control and manage the flow of information between the current
    state and the previous state. This makes the GRU more expressive
    and easier to optimize compared to a standard RNN cell.
    """

    def __init__(self):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(N_EMBD + N_EMBD2, N_EMBD2)
        self.xh_to_r = nn.Linear(N_EMBD + N_EMBD2, N_EMBD2)
        self.xh_to_hbar = nn.Linear(N_EMBD + N_EMBD2, N_EMBD2)

    def forward(self, xt, hprev):
        """
        Defines the forward pass for the GRU cell.

        Args:
            xt: The input at the current time step.
            hprev: The hidden state at the previous time step.

        Returns:
            ht: The hidden state at the current time step.
        """
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = torch.sigmoid(self.xh_to_r(xh))  # reset gate - squashing to zero
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = torch.tanh(self.xh_to_hbar(xhr))  # new candidate hidden state
        # calculate the switch gate that determines if each channel should be updated at all
        z = torch.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht


class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) module.
    This module can use either a standard RNN cell
    or a Gated Recurrent Unit (GRU) cell.
    """

    def __init__(self, cell_type):
        """
        Initializes the RNN module.

        Args:
            cell_type (str): The type of cell to use in the RNN.
            Can be either 'rnn' for a standard RNN cell or 'gru' for a GRU cell.
        """
        super().__init__()
        self.BLOCK_SIZE = BLOCK_SIZE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.start = nn.Parameter(torch.zeros(1, N_EMBD2))  # the starting hidden state
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)  # token embeddings table
        if cell_type == "rnn":
            self.cell = RNNCell()
        elif cell_type == "gru":
            self.cell = GRUCell()
        self.lm_head = nn.Linear(N_EMBD2, self.VOCAB_SIZE)

    def forward(self, idx, targets=None):
        """
        Defines the forward pass for the RNN module.

        Args:
            idx (torch.Tensor): The input tensor.
            targets (torch.Tensor, optional): The target tensor.
            If provided, the loss will also be calculated.

        Returns:
            logits (torch.Tensor): The output logits.
            loss (torch.Tensor, optional): The calculated loss.
            Only returned if targets is not None.
        """
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx)  # (b, t, N_EMBD)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1))  # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]  # (b, N_EMBD)
            ht = self.cell(xt, hprev)  # (b, N_EMBD2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1)  # (b, t, N_EMBD2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad
    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens.

        Args:
            idx (torch.Tensor): The input tensor.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            idx (torch.Tensor): The input tensor with the newly generated
            tokens appended.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) -> sample
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# init model
model = RNN(cell_type="rnn")
# model = RNN( cell_type='gru')
print(f"model #params: {sum(p.numel() for p in model.parameters())}")

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
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
