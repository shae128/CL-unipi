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
    return sorted(dic, key=lambda x: (dic[x]['count']), reverse=True)


########################################
########### Name Entity finder #########
########################################
def topTenNE(senteces):

    # to hold each name entity and it's count and Sentences
    NE_CS = {}

    # PoS tagging tokes
    for sentence in senteces:
        tokens = nltk.word_tokenize(sentence)
        PosTags = nltk.pos_tag(tokens)
        ne_chunks = nltk.ne_chunk(PosTags)

        for node in ne_chunks:
            try:
                node.label()
            except AttributeError:
                pass
            else:
                if node.label() == "PERSON":
                    for leave in node.leaves():
                        if leave[0] not in NE_CS.keys():
                            NE_CS[leave[0]] = { 'count' : 1, 'senteces' : [] }
                            NE_CS[leave[0]]['senteces'].append(sentence)
                        else:
                            NE_CS[leave[0]]['count'] += 1
                            NE_CS[leave[0]]['senteces'].append(sentence)
    

    # sort the NE_CS dictionary
    NE_CS_Sorted = orderDic(NE_CS)

    # Create a list to hold top ten Name entities and their related info
    topTen_NE = []
    for i in range(0,10):
        tempDic = {}
        the_name = NE_CS_Sorted[i]
        tempDic['Name'] = the_name
        tempDic['count'] = NE_CS[the_name]['count']
        tempDic['senteces'] = NE_CS[NE_CS_Sorted[i]]['senteces']
        topTen_NE.append(tempDic)


    return topTen_NE



######################
# The Main Functi n  #
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
#    tokens1 = nltk.word_tokenize(rawFile1)
#    tokens2 = nltk.word_tokenize(rawFile2)
#
    # Find Person names
    topTenPerson1 = topTenNE(sentences1)
    topTenPerson2 = topTenNE(sentences2)

main(sys.argv[1], sys.argv[2])
