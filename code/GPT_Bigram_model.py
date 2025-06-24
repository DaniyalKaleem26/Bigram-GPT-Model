# Importing the needed libraries
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
'''
Block Size is the context size of our model. Context length determines the window of input that the transformer considers to generate the next meaningful output. If set to small, 
it would not take into consideraton the dependable words that affect the output, and if set to big, it owuld take in words that have no contribution to 
meaning of the output. Actual GPTs have context limits of 3000 to 7000 words.
The batch size signifies how many batches of input and output pairs will we use in parallel. We need to work parallel since GPTs are trained on huge amounts of data. 
'''
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000 # Determine the number of iterations we need to run for the training process
eval_interval = 300 # Number used to give the loss of training and validation set every once in a while
learning_rate = 1e-2 # Sets the learning rate of adam optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Runs on GPU if available
eval_iters = 200 # Number of iterations for evaluating our model


torch.manual_seed(1337) # For reproducibility

# Reading the Shakespear poems text file. You can find the text file using the following link.
# wget https://raw.githubusercontent.com/DaniyalKaleem26/Bigram-GPT-Model/refs/heads/main/data/text
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Lets find all the unique characters that occur in this text file
chars = sorted(list(set(text)))
vocab_size = len(chars) # Total size of all the unique character present in our text file

# create a mapping from characters to integers
'''
We enumerate the each charcter in the text file and then store its index and character itself in i and ch respectively
Encode takes a string s and for each character c in the string it gives its index by using the dictionary stoi.
Decode does the reverse of it by taking the list of integers and giving the respective character c. 
The characters are then joined using join function with no spaces to form a string. 
''' 
stoi = { ch:i for i,ch in enumerate(chars) } 
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# We convert the whole text file into a 1D pytorch tensor of long int 64 type.
data = torch.tensor(encode(text), dtype=torch.long) # Encodes the whole text file into a multi-dimensional matrix
n = int(0.9*len(data)) # first 90% will be train set, rest validation set
train_data = data[:n] 
val_data = data[n:]

# data loading
def get_batch(split):
    '''
    The function generates a small batch of training or validation data depending upon the required split.
    The function will then take a randomized starting point i, but that starting point cant be such that i + block size(which is the context length) 
    exceeds the tensor length. Thus we subtract the block size from the length of data. The randint will give a 1d array of end position and number of elements.
    We now stack the data one by one using torch.stack to form the feature tensor, for the label tensor we just increment our starting and ending position by one.
    The label is for a word is the next word in the sentence. 
    '''
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # Moves the batches to GPU, if they are available
    return x, y

@torch.no_grad() # Prevents PyTorch from tracking computations for autograd (gradient calculations), since we're not training, just evaluating.
def estimate_loss():
    out = {}
    model.eval() # Switch to evaluation mode (important for dropout, batchnorm, etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # This gives the number of times we want to compute loss to get a good estimate. losses is a tensor that will store each loss value
        for k in range(eval_iters): 
            X, Y = get_batch(split) # Sample one batch of input/output pairs
            logits, loss = model(X, Y) # Run forward pass
            losses[k] = loss.item() # Store scalar loss value
        out[split] = losses.mean() # Average loss over eval_iters batches
    model.train() # Switch back to training mode
    return out # {'train': avg_train_loss, 'val': avg_val_loss}

# super simple bigram model
class BigramLanguageModel(nn.Module):
    '''
    This class defines the Bigram model(predicts the next word based on the current one, two words at play thus Bigram).
    The class inherits all the methods and properties from nn.Module.
    '''

    '''
    __init__ is our constructor, together with super() it calls the base class(i.e nn.module)
    We make a embedding lookup table matrix of size vocab_size by vocab_size. The embedding matrix contains vectors for the index of each character
    that are randomised initially. Each vector contains vocab_size number of values.
    '''

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    '''
    idx is of the for (B,T). B is the batch size and T is the context length. Targets are next token tensor that will be used for training, 
    they are optional. Each element in the idx is a index value of the character present inside the mini batch that we are using for a single iteration.
    PyTorch replaces each index with the corresponding row (vector) from the embedding matrix, which has shape (vocab_size, vocab_size).
    So for each token (integer) in idx, it fetches its associated vector of logits (length = vocab_size) from the embedding table.
    As such, this idx will now become a 3D tensor as each index value of a character is now replaced by its vector of probability values inside the array
    of batch size and context length. The shape of logits is (B,T,C) where C is the vocab_size. 

    To predict the loss we use Cross-entropy. To use cross entropy we need to collapse the column dimension and produce a 2D array for logits and 1D for targets.
    Now we have a tensor with 256 rows and 65 columns. Each row corresponds to a single token prediction, 
    and contains the 65 logits (scores) for each possible next character in the vocabulary.
    '''

 
    def forward(self, idx, targets=None):

        # Sidenote: Logits are logarithm of the odds of an event, representing the raw, unnormalized output of a classification model before any activation function is applied.
        logits = self.token_embedding_table(idx) # Shape of logits is (B,T,C), C being the vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context. There is no voacb_size present here.
        '''
        The logits and the loss are calculated for each iteration using the forward self function. The last logit is used to calculate the next character
        in our sequence since after passing through the transformer the whole of the sentence will have its meaning baked in the last character. 
        The targets are set to none, as such their is no reshaping of logits dimensions thus uses [:, -1, :] and outputs (B,C) since T is 1.
        Softmax is then applied to logits to convert those arrays of probabilistic numbers into normal distributive values between 0 and 1.
        Now we sample from this distribution to get the next token of our sequence using torch.multinomial and then concatenate it with idx to form the longer context.
        '''
        for _ in range(max_new_tokens):
            # We get the logits and loss from the forward self function.
            logits, loss = self(idx) 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size) # Creates an instance of our Bigram model. This helps create the embedding table of vocab_size by vocab_size.
# When we call the model next, we would need to specify our features and labels i.e x and y. 
m = model.to(device) # Switches to GPU if available 

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Sets the optimizer parameters.

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    # Useful to check for overfitting
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train') # Training dataset used

    # evaluate the loss
    logits, loss = model(xb, yb) # Logits and loss are calculated 
    optimizer.zero_grad(set_to_none=True) # Clears previous gradient
    loss.backward() # Backpropagates the loss to compute gradients.
    optimizer.step() # Updates model weights using gradients.
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Context is the first initial input to the model. It's just a seed to start generation.
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # Generates 500 tokens. Predict next token probabilities from last token. 
# Sample from those probabilities. Append the new token to the input sequence.
# Finally it prints the shakespeare like artificial text. 