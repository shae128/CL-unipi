#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import codecs
import nltk
from prettytable import PrettyTable



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


    # Initializing pretty table
    basicT = PrettyTable()
    
    # Adding table's columns
    basicT.field_names = ["Metrics", "The Adventures Of Sherlock Holmes", "The Time Machine"]

    # Filling table's rows
    basicT.add_row(["Tokens", len(tokens1), len(tokens2)])
    basicT.add_row(["Sentences", len(sentences1), len(sentences2)])
    basicT.add_row(["Tokens average length", "{:.2f}".format(tokensAvgLen1), "{:.2f}".format(tokensAvgLen2)])
    basicT.add_row(["Sentences average length", "{:.2f}".format(sentenceAvgLen1), "{:.2f}".format(sentenceAvgLen2)])

    # Print out the basic table
    print(basicT.get_string(title=" B A S I C      A N A L Y S I S"))
#
#    print ("The Adventures Of Sherlock Holmes: ")
#    print ("Sentences: ", len(sentences1))
#    print ("Sentences average length: ", sentenceAvgLen1)
#    print ("Tokens: ", len(tokens1))
#    print ("Tokens average length: ", tokensAvgLen1)
#    print ("#############################################")
#    print ("The Time Machine: ")
#    print ("Sentences: ", len(sentences2))
#    print ("Sentences average length: ", sentenceAvgLen2)
#    print ("Tokens: ", len(tokens2))
#    print ("Tokens average length: ", tokensAvgLen2)
#

main(sys.argv[1], sys.argv[2])
