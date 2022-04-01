# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:19:42 2022

@author: Leonid_PC
"""

import numpy as np
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F



"""
# define and print the net
n_hidden = 512
n_layers = 2
net = CharRNN(chars, n_hidden, n_layers)
print(net)
batch_size = 128
seq_length = 100
n_epochs =  10  # start small if you are just testing initial behavior
# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, 
      seq_length=seq_length, lr=0.001, print_every=10)
    
print(sample(net, 500, prime='christmas', top_k=2))
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode(text):
    """
    Encode text from string to integer
    """
    
    chars = list(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    
    encoded_text = np.array([char2int[ch] for ch in text])
    
    return encoded_text, chars, int2char, char2int

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def get_batches(arr, batch_size, seq_length):
    """
    Create a generator that returns batches of size
    batch_size x seq_length from arr.
       
    Arguments
    ---------
    arr: Array you want to make batches from
    batch_size: Batch size, the number of sequences per batch
    seq_length: Number of encoded chars in a sequence
    """
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        

def save_model(net):
    # change the name, for saving multiple files
    model_name = 'poem_4_epoch.net'
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}
    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)


class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=1e-3):
        
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.encoded_text, self.chars, self.int2char, self.char2int = \
            encode(tokens)
        
        self.lstm=nn.LSTM(len(self.chars), n_hidden,n_layers,   # LSTM layer 
                          dropout=drop_prob, batch_first=True)
        self.dropout=nn.Dropout(drop_prob)  # dropout layer
        self.fc=nn.Linear(n_hidden, len(self.chars))  # output layer
    
    def forward(self, x, hidden):
        """
        Forward pass through the network. 
        These inputs are x, and the hidden/cell state `hidden`. 
        """
        # Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        # pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        # put x through the fully-connected layer
        out = self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return hidden
    
    
def train(net, epochs=10, batch_size=10, seq_length=50, lr=0.001, 
          clip=5, val_frac=0.1, print_every=10):
    """ 
    Training a network 
    
        Arguments
        ---------        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    """
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=net.lr)
    criterion = nn.CrossEntropyLoss()
    
    data = net.encoded_text
    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    net.to(device)
    
    n_chars = len(net.chars)
    #for e in range(epochs):
    for e in trange(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for i, (x, y) in enumerate(get_batches(data, batch_size, seq_length)):
            
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if i % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    
                    x = one_hot_encode(x, n_chars)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    inputs, targets = inputs.to(device), targets.to(device)
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(i),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


def predict(net, char, h=None, top_k=None):
        """
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        """
        
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        inputs = inputs.to(device)
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)
        # get the character probabilities
        p = F.softmax(out, dim=1).data
        p = p.to(device)
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.cpu().numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.cpu().numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
        
    net.to(device)
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [i for i in prime]
    h = net.init_hidden(1)
    for i in prime:
        char, h = predict(net, i, h, top_k=top_k)
    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for i in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)




