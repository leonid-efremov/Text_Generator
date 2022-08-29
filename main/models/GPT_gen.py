# -*- coding: utf-8 -*-
"""
Created on Wen Aug 18 13:34:21 2022

@author: Leonid_PC
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
logging.disable(level=logging.CRITICAL)

MODELS_PATH = 'main/models/'
MODELS_PATH_RESERVE = 'models/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TextGenerator_GPT:

    def __init__(self, model_type):
        """Load pretrained model from model_path"""
        try:
            self.md = MODELS_PATH + model_type + "_gpt/essays"
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.md)
            self.model = GPT2LMHeadModel.from_pretrained(self.md)
        except OSError:
            self.md = MODELS_PATH_RESERVE + model_type + "_gpt/essays"
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.md)
            self.model = GPT2LMHeadModel.from_pretrained(self.md)

        self.model.to(device)

    def gen_from_pretrained(self, context=" ", max_length=50):
        """Generate phrase from content"""
        input = self.tokenizer.encode(context, return_tensors="pt")

        output = self.model.generate(input.to(device), max_length=max_length, 
                                     repetition_penalty=5.0, do_sample=True, 
                                     top_k=5, top_p=0.95, temperature=1)

        phrase = self.tokenizer.decode(output[0])
        return phrase

