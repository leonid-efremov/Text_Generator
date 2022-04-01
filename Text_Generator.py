# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Генератор теста
main файл:
    - простая грамматическая модель
    - интерфейс для вызова LSTM модели
    - основная программа
"""

import os 
import json
import pickle
from random import choice
from Lang_gen import trigramm_model, lang_preprocess
from LSTM_gen import CharRNN, train, predict, sample, encode



class TextGenerator:

    def __init__(self, model_type='LSTM'):
        """
        Class for generating text
        """
        
        self.model_type = model_type

        self.train_text = None        
        self.init_word = None
        self.num_words = 15
        
        a = ord('а')  # add all letters which needed - Russian alphabet (abc)
        self.abc = ''.join([chr(i) for i in range(a, a + 32)] 
                                + [' '] + ['\n'] + [chr(a + 33)])
        

    def load_text(self, source='WarAndPeace1.txt'):
        """
        Train model: for words in 'source' find next words
        Save into pickle binary format
        """

        file_name, file_extension = os.path.splitext(source)
        
        if file_extension == '.txt':
            with open(source, 'r') as f:  # read text from file
                self.train_text = f.read()  
                
        elif file_extension == '.json':
            with open(source, 'r', encoding="utf-8") as f:
                t = json.load(f)  # loading chat from telegram
                self.train_text = str()
                for i in t['messages']:
                    self.train_text += str(i['text']) + ' '  

        #with open('DontRename&PlaceInOneFolder.pkl', 'wb') as w:
        with open('words.pkl', 'wb') as w:  # save data to file
            pickle.dump(self.train_text, w)
    
    
    def prepare_model(self, data=None):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of texts.
        If data is None load saved file
        """
        
        # loading data
        if data is not None:  
            for i in data:
                self.load_text(source=i)
        else:
            # DontRename&PlaceInOneFolder
            with open('DontRename&PlaceInOneFolder.pkl', 'rb') as w:
                self.train_text = pickle.load(w)
        
        # setup and train model
        self.corpus, _, _ = lang_preprocess(self.train_text, self.abc)
        
        if self.model_type == 'Lang':
            _, self.words2, self.words3 = lang_preprocess(
                self.train_text, self.abc) 
        
        elif self.model_type == 'LSTM':
            # net parameters
            n_hidden = 512
            n_layers = 2
            tokens = self.train_text.lower()  # convert to lowercase  
            tokens = ''.join([i for i in tokens if i in self.abc])
            self.LSTM_model = CharRNN(self.train_text, n_hidden, n_layers)
            
            
    def generate(self):
        "Generation of phrase for initial word"
        
        if self.model_type == 'Lang':
            phrase = trigramm_model(self.init_word, self.words2, self.words3, 
                                    num_words=self.num_words)
            res = ' '.join(phrase)  # save and print resulting phrase
            
        elif self.model_type == 'LSTM':
            # train parameters
            batch_size = 128
            seq_length = 100
            n_epochs =  10
            
            encoded, _, _, _ = encode(self.train_text)
            train(self.LSTM_model, epochs=n_epochs, batch_size=batch_size, 
                  seq_length=seq_length, lr=0.001, print_every=10)
            
            prime = self.init_word if self.init_word else choice(self.corpus)
            res = sample(self.LSTM_model, self.num_words*10, prime=prime, top_k=2)
            
        print(res)
        return res
            
    
    
                   
if __name__ == '__main__':  
    
    gen = TextGenerator(model_type='LSTM')  # initate model
    
    train_texts = ['WarAndPeace1.txt', 'HarryPotter.txt']  # ['result.json']
    gen.prepare_model(data=train_texts)  # data=train_texts   
    #start_word = str(input('Enter your word:'))  # generate phrase from word
    #gen.init_word = 'утро доброе'  # start_word
    p = gen.generate()    
    
    
    
    
