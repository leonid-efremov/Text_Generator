# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:28:21 2022

@author: Leonid_PC
"""

from random import choice


def preprocess(text, abc, for_lang=False):
    """
    Create preprocessed data for model from text
    """
    words2 = dict()
    words3 = dict()
    
    # remove all sybols which are not letters        
    corpus = text.lower()  # convert to lowercase  
    corpus = ''.join([i for i in corpus if i in abc])       
    
    if for_lang:
        # find unique words and next words for each in bigramm model
        for prev, curr in zip(corpus, corpus[1:]):
            if prev not in words2.keys():
                words2[prev] = []
            words2[prev].append(curr)
            # same for trigramm model    
        for prev, curr, n in zip(corpus, corpus[1:], corpus[2:]):
            if (prev, curr) not in words3.keys():
                words3[(prev, curr)] = []
            words3[(prev, curr)].append(n)
            
        # remove empty elements
        words2 = dict([(k, v) for k,v in words2.items() if len(v) > 0])
        words3 = dict([(k, v) for k,v in words3.items() if len(v) > 0])
        
    return corpus, words2, words3


# Language n-gram models
def bigramm_model(init_word, words2, num_words=15):
    "Generate phrase in bigramm model"
    
    phrase = list()
    words = [item for sublist in list(words2.values()) for item in sublist]
    
    # set initial word or pick random
    if init_word:
        init_word = init_word.lower()
        init_word = init_word.split()
        for i in init_word:
            phrase.append(i) 
    else:
        phrase.append(choice(words))         
    li = len(phrase)
    
    for _ in range(num_words - li):
        # if word appear in train data pick random next word
        if phrase[-1] in list(words2.keys()):
            candidates = words2[phrase[-1]]
            phrase.append(choice(candidates))
        # else pick just random word
        else:
            phrase.append(choice(words))
    
    return phrase



def trigramm_model(init_word, words2, words3, num_words=15):
    "Generate phrase in trigramm model"
    
    phrase = list()
    words = [item for sublist in list(words3.values()) for item in sublist]
    
    # set two initial words or pick random
    if init_word:
        init_word = init_word.lower()                  
        init_word = init_word.split()
        for i in init_word:
            phrase.append(i)
    else:
        phrase.append(choice(words))  
    # if word appear in train data pick random next word
    if len(phrase) < 2:
        if phrase[-1] in list(words2.keys()):
            candidates = words2[phrase[-1]]
            phrase.append(choice(candidates))
        # else pick just random word
        else:
            phrase.append(choice(words))
    li = len(phrase)
        
    for i in range(num_words - li):
        # if word appear in train data pick random next word
        if (phrase[-2], phrase[-1]) in list(words3.keys()):
            candidates = words3[(phrase[-2], phrase[-1])]
            phrase.append(choice(candidates))
        # elif pick word from bigramm model
        elif phrase[-1] in list(words2.keys()):  
            candidates = words2[phrase[-1]]
            phrase.append(choice(candidates))
        # else pick just random word
        else:
            phrase.append(choice(words))
    
    return phrase

    
