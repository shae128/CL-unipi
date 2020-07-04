#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import codecs
import nltk

def main(file1):

    # Declare function scope variables

    # Sentences length list
    sentenceLens = []
    # Sentences token length sum
    sumST = 0

    # Tokens length list
    tokensLens = []
    # Tokens characters length sum
    sumTL = 0

    #open the first file
    inputFile= codecs.open(file1, 'r', 'utf-8')
    #read the first file
    rawFile = inputFile.read()
    #call english.pickle
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #tokenize in sentences
    sentences = sent_tokenizer.tokenize(rawFile)
    #tokenize in words
    tokens = nltk.word_tokenize(rawFile)

    # Create a list to store Sentences length
    for sentence in sentences:
        #tokenize Sentences in words
        sentenceTokens = nltk.word_tokenize(sentence)
        # add sentence token numbers to function scope list
        sentenceLens.append(len(sentenceTokens))

    # sum all Sentences length together 
    for length in sentenceLens:
        sumST += length
        
    # Sentences average length
    sentenceAvgLen = sumST / len(sentenceLens)



    # Create a list to store tokens length
    for token in tokens:
        tokensLens.append(len(token))
 
    # sum all tokens length together 
    for length in tokensLens:
        sumTL += length
               
    # Tokens average length
    tokensAvgLen = sumTL / len(tokensLens)

#
#    for token in tokens:
#        print(token.encode('utf-8'))
#        print("------------------")
#        print("------------------")
#
#    print(sentences)
#    print(tokens)
#    print ("sentences lenght: ", sentenceLens, len(sentenceLens))

    print ("Sentences: ", len(sentences))
    print ("Sentences average length: ", sentenceAvgLen)
    print ("Tokens: ", len(tokens))
    print ("Tokens average length: ", tokensAvgLen)

main(sys.argv[1])
