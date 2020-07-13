#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import codecs
import math
import nltk
from nltk import bigrams
from nltk import FreqDist
from prettytable import PrettyTable

###############################
###### Order Dictionary #######
###############################
def orderDic(dic):
    return sorted(dic.items(), key=lambda x: x[1], reverse=True)


###############################
###### Remove punctuations####3
###############################
def removePunc(tokens):
    noPunc = []
    for token in tokens:
        if not token in [",",".",";",":","-","_","[","]","{","}","'","?","(",")",'"',"''","``","!","$","#"]:
            noPunc.append(token)

    return noPunc






######################
# The Main Function  #
######################
def main(file1,file2):

    #open files
    inputFile1= codecs.open(file1, 'r', 'utf-8')
    inputFile2= codecs.open(file2, 'r', 'utf-8')
    #read files
    rawFile1 = inputFile1.read()
    rawFile2 = inputFile2.read()

    #call english.pickle
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #tokenize in sentences
    sentences1 = sent_tokenizer.tokenize(rawFile1)
    sentences2 = sent_tokenizer.tokenize(rawFile2)

    #tokenize in words
    tokens1 = nltk.word_tokenize(rawFile1)
    tokens2 = nltk.word_tokenize(rawFile2)



main(sys.argv[1], sys.argv[2])
