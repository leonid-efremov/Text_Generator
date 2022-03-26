# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Генератор теста
"""

import os 
import json
import pickle
import pandas as pd
from random import choice
import torch
from torch import nn
import torch.nn.functional as F



class TextGenerator:

    def __init__(self):
        """
        Class for generating text
        """

        self.train_text = None
        self.words2 = dict()  # collecting all words and next words
        self.words3 = dict()
        
        self.init_word = None
        self.num_words = 15
        
        a = ord('а')  # add all letters which needed - russian alphabet (abc)
        self.abc = ''.join([chr(i) for i in range(a, a + 32)] 
                                + [' '] + ['\n'] + [chr(a + 33)])
        
    def encode(self):
        """
        Encode text from string to integer
        """
        chars = list(set(self.train_text))
        pass
        

    def fit(self, source='WarAndPeace1.txt'):
        "Train model: for words in 'source' find next words"

        file_name, file_extension = os.path.splitext(source)
        if file_extension == '.txt':
            with open(source, 'r') as f:  # read text from file
                self.train_text = f.read()        
        elif file_extension == '.json':
            with open(source, 'r', encoding="utf-8") as f:
                t = json.load(f)  # loading ghat from telegram
                self.train_text = str()
                for i in t['messages']:
                    self.train_text += str(i['text']) + ' '  

        #with open('DontRename&PlaceInOneFolder', 'wb') as w:
        with open('words', 'wb') as w:  # save data to file
            pickle.dump(self.train_text, w)
            
        
    def _preprocess(self, text):
        """
        Create preprocessed data for model from text
        """
        # remove all sybols which are not letters        
        self.corpus = self.train_text.lower()  # convert to lowercase  
        self.corpus = ''.join([i for i in self.corpus if i in self.abc])           
        self.corpus = self.corpus.split()  # split text in single words

        # find unique words and next words for each in bigramm model
        for prev, curr in zip(self.corpus, self.corpus[1:]):
            if prev not in self.words2.keys():
                self.words2[prev] = []
            self.words2[prev].append(curr)
        # same for trigramm model    
        for prev, curr, n in zip(self.corpus,self.corpus[1:],self.corpus[2:]):
            if (prev, curr) not in self.words3.keys():
                self.words3[(prev, curr)] = []
            self.words3[(prev, curr)].append(n)
            
        # remove empty elements
        self.words2 = dict([(k, v) for k,v in self.words2.items() if len(v)>0])
        self.words3 = dict([(k, v) for k,v in self.words3.items() if len(v)>0])
        
        return self.corpus, self.words2, self.words3
        
        
     
    # Language n-gram models
    def bigramm_model(self):
        "Generate phrase in bigramm model"
        
        phrase = list()
        words = [item for sublist in list(self.words2.values()) \
                 for item in sublist]
        
        # set initial word or pick random
        if self.init_word is not None:
            self.init_word = self.init_word.lower()
            self.init_word = self.init_word.split()
            for i in self.init_word:
                phrase.append(i) 
        else:
            phrase.append(choice(words))         
        li = len(phrase)
        
        for _ in range(self.num_words - li):
            # if word appear in train data pick random next word
            if phrase[-1] in list(self.words2.keys()):
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        
        return phrase
    


    def trigramm_model(self):
        "Generate phrase in trigramm model"
        
        phrase = list()
        words = [item for sublist in list(self.words3.values()) \
                 for item in sublist]
        
        # set two initial words or pick random
        if self.init_word is not None:
            self.init_word = self.init_word.lower()                  
            self.init_word = self.init_word.split()
            for i in self.init_word:
                phrase.append(i)
        else:
            phrase.append(choice(words))  
        # if word appear in train data pick random next word
        if len(phrase) < 2:
            if phrase[-1] in list(self.words2.keys()):
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        li = len(phrase)
            
        for i in range(self.num_words - li):
            # if word appear in train data pick random next word
            if (phrase[-2], phrase[-1]) in list(self.words3.keys()):
                candidates = self.words3[(phrase[-2], phrase[-1])]
                phrase.append(choice(candidates))
            # elif pick word from bigramm model
            elif phrase[-1] in list(self.words2.keys()):  
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        
        return phrase
    
        
        
    def generate(self, model='trigram'):
        "Generation of phrase for initial word"
        
        if model == 'trigram':
            phrase = self.trigramm_model()        
        res = ' '.join(phrase)  # save and print resulting phrase
        print(res)
        return res
    
    
    def prepare_model(self, data=None):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of texts.
        If data is None load saved file
        """
        
        self.words2 = dict()
        self.words3 = dict()
        
        if data is not None:  # train model if necessary data specified
            for i in data:
                self.fit(source=i)
        else:
            # DontRename&PlaceInOneFolder
            with open('DontRename&PlaceInOneFolder', 'rb') as w:  # load data
                self.train_text = pickle.load(w)
        
        self.corpus, self.words2, self.words3 = \
            self._preprocess(self.train_text)
        
        

def main():
    "Main function"
    
    gen = TextGenerator()  # initate model
    
    train_texts = ['WarAndPeace1.txt', 'HarryPotter.txt']  # ['result.json']
    gen.prepare_model(data=train_texts)  # data=train_texts   
    #start_word = str(input('Enter your word:'))  # generate phrase from word
    #gen.init_word = 'утро доброе'
    p = gen.generate()    
    return gen, p
                   
if __name__ == '__main__':    
    gen, p = main()

    
    
    
    
