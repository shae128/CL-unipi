#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import codecs
import nltk

# Calculate Sentences' length
def sentencesLenCal(sentences):

    ######################################
    # Declare function's scope variables #
    ######################################

    # Sentences length list
    sentenceLens = []
    # Sentences token length sum
    sumST = 0

    # Fill the list the Sentences length
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

    return sentenceAvgLen


# Calculate tokens' length
def tokensLenCal(tokens):

    ######################################
    # Declare function's scope variables #
    ######################################

    # Tokens length list
    tokensLens = []
    # Tokens characters length sum
    sumTL = 0

    # Fill the list of tokens length
    for token in tokens:
        tokensLens.append(len(token))
 
    # sum all tokens length together 
    for length in tokensLens:
        sumTL += length
               
    # Tokens average length
    return sumTL / len(tokensLens)



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


    sentenceAvgLen1 = sentencesLenCal(sentences1)
    tokensAvgLen1 = tokensLenCal(tokens1)
#
    sentenceAvgLen2 = sentencesLenCal(sentences2)
    tokensAvgLen2 = tokensLenCal(tokens2)
#    for token in tokens:
#        print(token.encode('utf-8'))
#        print("------------------")
#        print("------------------")
#
#    print(sentences)
#    print(tokens)
#    print ("sentences lenght: ", sentenceLens, len(sentenceLens))

    print ("The Adventures Of Sherlock Holmes: ")
    print ("Sentences: ", len(sentences1))
    print ("Sentences average length: ", sentenceAvgLen1)
    print ("Tokens: ", len(tokens1))
    print ("Tokens average length: ", tokensAvgLen1)
    print ("#############################################")
    print ("The Time Machine: ")
    print ("Sentences: ", len(sentences2))
    print ("Sentences average length: ", sentenceAvgLen2)
    print ("Tokens: ", len(tokens2))
    print ("Tokens average length: ", tokensAvgLen2)


main(sys.argv[1], sys.argv[2])
