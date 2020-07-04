#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import codecs
import nltk

def main(file1):

    # Declare function scope variables
    sentenceLens = []
    # Sentences token length sum
    sumST = 0

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

    for sentence in sentences:
    #    print(sentence.encode('utf-8'))
        #tokenize Sentences in words
        sentenceTokens = nltk.word_tokenize(sentence)
    #    print(sentenceTokens)
    #    print("------------------")
        # add sentence token numbers to function scope list
        sentenceLens.append(len(sentenceTokens))

    for length in sentenceLens:
        sumST += length
        
    # Sentences average length
    sentenceAvgLen = sumST / len(sentenceLens)

        
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
    print ("Tokens: ", len(tokens))
    print ("sentences average length: ", sentenceAvgLen)


main(sys.argv[1])
