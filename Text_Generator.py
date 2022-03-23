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
        
        a = ord('а')  # add all letters which needed a.e. all russian alphabet
        self.alphabet = ''.join([chr(i) for i in range(a, a + 32)] 
                                + [' '] + ['\n'] + [chr(a + 33)])
        

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

         
        # remove all sybols which are not letters        
        train_text = self.train_text.lower()  # convert to lowercase  
        train_text = ''.join([i for i in train_text if i in self.alphabet])           
        train_text = train_text.split()  # split text in single words

        # find unique words and next words for each in bigramm model
        for prev, curr in zip(train_text, train_text[1:]):
            if prev not in self.words2.keys():
                self.words2[prev] = []
            self.words2[prev].append(curr)
        # trigramm model    
        for prev, curr, n in zip(train_text, train_text[1:], train_text[2:]):
            if (prev, curr) not in self.words3.keys():
                self.words3[(prev, curr)] = []
            self.words3[(prev, curr)].append(n)
            
        # remove empty elements
        self.words2 = dict([(k, v) for k,v in self.words2.items() if len(v)>0])
        self.words3 = dict([(k, v) for k,v in self.words3.items() if len(v)>0])
        
        # DontRename&PlaceInOneFolder
        save_data = [self.words2, self.words3]
        with open('words', 'wb') as w:  # save data to file
            pickle.dump(len(save_data), w)
            for d in save_data:
                pickle.dump(d, w)
        
   
        
    def bigramm_model(self):
        "Generate phrase in bigramm model"
        
        phrase = list()
        words = [item for sublist in list(self.words2.values()) for item in sublist]
        
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
        words = [item for sublist in list(self.words3.values()) for item in sublist]
        
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
    
        
        
    def generate(self):
        "Generation of phrase for initial word"
        
        phrase = self.trigramm_model()        
        res = ' '.join(phrase)  # save and print resulting phrase
        print(res)
        return res
    
    
    def prepare_model(self, data=None):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of text names.
        If data is None load saved file
        """
        
        self.words2 = dict()
        self.words3 = dict()
        
        if data is not None:  # train model if necessary data specified
            for i in data:
                self.fit(source=i)
        else:
            # DontRename&PlaceInOneFolder
            save_data = []
            with open('DontRename&PlaceInOneFolder', 'rb') as w:  # load data
                for _ in range(pickle.load(w)):
                    save_data.append(pickle.load(w))
            self.words2 = save_data[0]
            self.words3 = save_data[1]
        
        

def main():
    "Main function"
    
    gen = TextGenerator()  # initate model
    
    train_texts = ['WarAndPeace1.txt', 'HarryPotter.txt']  #['result.json']
    gen.prepare_model(data=train_texts)    
    #start_word = str(input('Enter your word:'))  # generate phrase from word
    #gen.init_word = 'утро доброе'
    p = gen.generate()    
    return gen, p
                   
if __name__ == '__main__':    
    gen, p = main()

    
    
    
    
