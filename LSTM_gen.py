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
        
def get_batch(sequence, batch_size, seq_len):
    trains = []
    targets = []
    for _ in range(batch_size):
        batch_start = np.random.randint(0, len(sequence) - seq_len)
        chunk = sequence[batch_start: batch_start + seq_len]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

def save_model(net):
    # change the name, for saving multiple files
    model_name = 'poem_4_epoch.net'
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}
    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)



class TextRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    
    def init_hidden(self, batch_size=1):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size, 
                              requires_grad=True).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size, 
                              requires_grad=True).to(device))
        return hidden
    
class TextGenerator_LSTM:
    def __init__(self, layers_sizes, train_text, abc, n_layers):
        self.abc = abc
        self.text = train_text
        self.encoded_text, _, self.idx_to_char, self.char_to_idx = encode(self.text)
        
        self.model = TextRNN(input_size=len(self.idx_to_char), 
                             hidden_size=layers_sizes['hidden_size'], 
                             embedding_size=layers_sizes['embedding_size'], 
                             n_layers=n_layers)
        self.model.to(device)
    
    def train(self, batch_size, seq_len, n_epochs=10000, lr=1e-3):
        
        model = self.model
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=5, 
            verbose=True, 
            factor=0.5)
        
        loss_avg = []

        for epoch in range(n_epochs):
            model.train()
            train, target = get_batch(self.encoded_text, batch_size, seq_len)
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            hidden = model.init_hidden(batch_size)

            output, hidden = model(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')
                scheduler.step(mean_loss)
                loss_avg = []
                predicted_text = self.evaluate(model)
                print(predicted_text)
                
    def evaluate(self, model, start_text=' ', prediction_len=200, temp=0.3):
        
        model = self.model
        char_to_idx, idx_to_char = self.char_to_idx, self.idx_to_char
        
        model.eval()
        
        hidden = model.init_hidden()
        idx_input = [char_to_idx[char] for char in start_text]
        train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
        predicted_text = start_text
        
        _, hidden = model(train, hidden)
            
        inp = train[-1].view(-1, 1, 1)
        
        for i in range(prediction_len):
            output, hidden = model(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_char = idx_to_char[top_index]
            predicted_text += predicted_char
        
        return predicted_text




