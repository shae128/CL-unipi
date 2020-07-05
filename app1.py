#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import codecs
import nltk
from prettytable import PrettyTable



###############################
# Calculate Sentences' length #
###############################
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


############################
# Calculate tokens' length #
############################
def tokensLenCal(tokens):


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

###############################################
# Calculate Vocabulary based on Token Numbers #
###############################################
def vocabCal(tokens, interval):

    # List of Vocabulary sizes
    vocabSize = []
    # Tokens size
    tokenSize = []

    lastPart = 0

    # Iterating in tokens' list by interval
    for tokenNum in range (interval, len(tokens), interval):
        vocabList = set(tokens[:tokenNum])
        tokenSize.append(tokenNum)
        vocabSize.append(len(vocabList))
        lastPart += tokenNum

    # If tokens' number is not dividable by interval 
    # Then add also the last part which
#    if lastPart != len(tokens):
#        vocabList = set(tokens)
#        tokenSize.append(len(tokens))
#        vocabSize.append(len(vocabList))

    finaleList = [tokenSize, vocabSize]

    return finaleList

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


    sentenceAvgLen1 = sentencesLenCal(sentences1)
    tokensAvgLen1 = tokensLenCal(tokens1)


    sentenceAvgLen2 = sentencesLenCal(sentences2)
    tokensAvgLen2 = tokensLenCal(tokens2)


    # Initializing pretty table
    basicT = PrettyTable()
    
    # Adding table's columns
    basicT.field_names = ["Metrics", file1[6:-4], file2[6:-4]]

    # Filling table's rows
    basicT.add_row(["Tokens", len(tokens1), len(tokens2)])
    basicT.add_row(["Sentences", len(sentences1), len(sentences2)])
    basicT.add_row(["Tokens average length", "{:.2f}".format(tokensAvgLen1), "{:.2f}".format(tokensAvgLen2)])
    basicT.add_row(["Sentences average length", "{:.2f}".format(sentenceAvgLen1), "{:.2f}".format(sentenceAvgLen2)])

    # Print out the basic table
    print(basicT.get_string(title=" B A S I C      A N A L Y S I S"))


    # Calculating vocabulary size
    vocabSize1 = vocabCal(tokens1, 5000)
    vocabSize2 = vocabCal(tokens2, 5000)


    # Initializing pretty table
    vocabT = PrettyTable()
    
    # Adding table's columns
    vocabT.field_names = ["Tokens", file1[6:-4], file2[6:-4]]

    # Find the bigger text size
    intervalSize1 = len(vocabSize1[0]) 
    intervalSize2 = len(vocabSize2[0]) 

    # synchronizing lists to show in a uniq table
    if intervalSize1 > intervalSize2:
        for i in range (intervalSize2, intervalSize1):
            vocabSize2[0].append(vocabSize1[0][i])
            vocabSize2[1].append("-")
            
    else:
        for i in range (intervalSize1, intervalSize2):
                vocabSize1[0].append(vocabSize2[0][i])
                vocabSize1[1].append("-")

    # Filling table's rows
    for i in range (len(vocabSize1[0])):
        vocabT.add_row([vocabSize1[0][i], vocabSize1[1][i], vocabSize2[1][i]])

    # Print out the basic table
    print(vocabT.get_string(title=" V O C A B U L A R Y    S I Z E"))



main(sys.argv[1], sys.argv[2])
