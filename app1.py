#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import codecs
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


###############################
# Calculate Sentences' length #
###############################
def sentencesLenCal(sentences):

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


##########################################
# Calculate HAPAX based on Token Numbers #
##########################################
def hapaxCal(tokens, interval):

    # Where to start each iteration
    startPoint = 0

    # Tokens size
    tokenSize = []

    # List of HAPAX
    hapax = []

    # List of HAPAX Length
    hapaxLen = []

    # Iterating in tokens' list by interval
    for tokenNum in range (interval, len(tokens), interval):

        tokenPortion = tokens[startPoint:tokenNum]
        vocabList = set(tokenPortion)
        tokenSize.append(tokenNum)

        startPoint = tokenNum + 1

        for token in vocabList:
            tokenFreq = tokenPortion.count(token)
            if tokenFreq == 1:
                hapax.append(token)

        # Calculate distribution |V1|/|C|
        hapaxLen.append("{:.5f}".format(len(hapax)/len(tokens)))

    finaleList = [tokenSize, hapaxLen]

    return finaleList


########################################
# The relation between nouns and verbs #
########################################
def NounVerb(tokens):

    Nouns = []
    Verbs = []

    # POS-Tagging all tokens of the test  
    PosTags = nltk.pos_tag(tokens)
    
    # Finding nouns and verbs
    for token in PosTags:
        if token[1] in ['NN','NNS','NP','NPS']:
            Nouns.append(token)
        elif token[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            Verbs.append(token)

    relation = "{:.5f}".format(len(Nouns)/len(Verbs))
    finaleList = [len(Nouns), len(Verbs), relation]

    return finaleList


########################################
############## PoS tagger ##############
########################################
def PosTagger(tokens, justTags=False):

    # remove punctuations
    noPunctTokens = removePunc(tokens)

    # PoS Tagging 
    PosTags = nltk.pos_tag(noPunctTokens)

    # to hold PoS Tags itselves instead of tuples
    justTags = []


    for tags in PosTags:
        justTags.append(tags[1])

    if justTags:
        return justTags
    else:
        return PosTags

########################################
############## Top 10 PoS ##############
########################################
def topTenPoS(tokens):

    # to hold PoS itselves instead of tuples
    justPoS = PosTagger(tokens, True)

    # To hold freq of any PoS
    PosFreq = {}

    # Sorted PoS Dictionary
    sortedPoS = {}

    # remove conflicts
    UniqPoS = set(justPoS)

    # Calculate each PoS freq
    for PoS in UniqPoS:
        posCount = justPoS.count(PoS)
        PosFreq[PoS] = posCount

    # Sort PoS Dictionary based on Values
    sortedPoS = orderDic(PosFreq)

    return sortedPoS 

########################################
############## Top 10 PoS ##############
########################################
def bigramsProb(tokens):

    # Dictionary to hold bigram's probability
    bigramProb = {}
    # To hold PoS-tagging Analysis
    PosTagged = PosTagger(tokens)
#    print(PosTagged[1000:1200])
#    print("++++++++++++++++++++++++++++++++++")   
#    print("++++++++++++++++++++++++++++++++++")   
    PosTagFreq = FreqDist(PosTagged)
#    print("++++++++++++++++++++++++++++++++++")   
#    print("Most common Tags")   
#    print(type(PosTagFreq['NN']))

    # to hold PoS itselves instead of tuples
    justPoS = PosTagger(tokens, True)

    # remove conflicts
    UniqPoS = set(justPoS)

    # To hold PoS-Bigrams
    PosBigrams = list(bigrams(PosTagged))
#    print(PosBigrams[1000:1200])   
#    print("++++++++++++++++++++++++++++++++++")   
#    print("++++++++++++++++++++++++++++++++++")   

    PosBigramsFreq = FreqDist(PosBigrams)

    # Find PoS-Bigrams
    for tag in UniqPoS:
        for bigram in PosBigramsFreq:
            if tag == bigram[0]:
                # Calculate bigram's probability 
                bigramProb[bigram] = "{:.5f}".format(PosBigramsFreq[bigram]/PosTagFreq[tag])

    sortedBigrams = orderDic(bigramProb)

    return sortedBigrams[:10]


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



    ############################################
    ############## Basic Analysis ##############
    ############################################

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

    #------------------------------------------#


    ############################################
    ############## Vocabulary Analysis #########
    ############################################

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

    #------------------------------------------#


    ############################################
    ############## Hapax Analysis ##############
    ############################################

    # Calculating hapax distribution
    hapaxDest1 = hapaxCal(tokens1, 5000)
    hapaxDest2 = hapaxCal(tokens2, 5000)

    # Initializing pretty table
    hapaxT = PrettyTable()
    
    # Adding table's columns
    hapaxT.field_names = ["Tokens", file1[6:-4], file2[6:-4]]


    # Find the bigger text size
    hapaxSize1 = len(hapaxDest1[0])
    hapaxSize2 = len(hapaxDest2[0])

    # synchronizing lists to show in a uniq table
    if hapaxSize1 > hapaxSize2:
        for i in range (hapaxSize2, hapaxSize1):
            hapaxDest2[0].append(hapaxDest1[0][i])
            hapaxDest2[1].append("-")
    else:
        for i in range (hapaxSize1, hapaxSize2):
                hapaxDest1[0].append(hapaxDest2[0][i])
                hapaxDest1[1].append("-")

    # Filling table's rows
    for i in range (len(hapaxDest1[0])):
        hapaxT.add_row([hapaxDest1[0][i], hapaxDest1[1][i], hapaxDest2[1][i]])

    # Print out the basic table
    print(hapaxT.get_string(title="H A P A X    DISTRIBUTION"))

    #------------------------------------------#



    ############################################
    ######## Nouns and Verbs relation ##########
    ############################################

    # Calculating relation for each text
    NV1 = NounVerb(tokens1)
    NV2 = NounVerb(tokens2)

    # Initializing pretty table
    NVT = PrettyTable()
    
    # Adding table's columns
    NVT.field_names = ["File", "Nouns", "Verbs", "Relation"]

    # Filling table's rows
    NVT.add_row([file1[6:-4], NV1[0], NV1[1], NV1[2]])
    NVT.add_row([len(file1[6:-4])*"-", "", "", ""])
    NVT.add_row([file2[6:-4], NV2[0], NV2[1], NV2[2]])

    # Print out the basic table
    print(NVT.get_string(title="NOUNS AND VERBS RELATION"))

    #------------------------------------------#



    ############################################
    ################ Top Ten PoS ###############
    ############################################


    # Find all PoS sorted by Freq 
    PoS1= topTenPoS(tokens1)
    PoS2= topTenPoS(tokens2)

    # Initializing pretty table
    PoST = PrettyTable()
    
    # Adding table's columns
    PoST.field_names = ["Rate", file1[6:-4], file2[6:-4]]

    # Filling table's rows
    for i in range(0,10):
        temp1 = PoS1[i][0] + ":   " + str(PoS1[i][1])
        temp2 = PoS2[i][0] + ":   " + str(PoS2[i][1])
        PoST.add_row([i+1, temp1, temp2])

    # Print out the basic table
    print(PoST.get_string(title="TOP TEN   P o S "))
    #------------------------------------------#



    ################################################################
    ################ Top Ten PoS-Bigrams probability ###############
    ################################################################


    # Find all PoS sorted by Freq 
    bigramsProb1 = bigramsProb(tokens1)
    bigramsProb2 = bigramsProb(tokens2)

    # Initializing pretty table
    BigramT = PrettyTable()
    
    # Adding table's columns
    BigramT.field_names = ["Rate", file1[6:-4], file2[6:-4]]

    # Filling table's rows
    for i in range(0,10):
        temp1 = str(bigramsProb1[i][0]) + ":   " + str(bigramsProb1[i][1])
        temp2 = str(bigramsProb2[i][0]) + ":   " + str(bigramsProb2[i][1])
        BigramT.add_row([i+1, temp1, temp2])

    # Print out the basic table
    print(BigramT.get_string(title="TOP TEN   PoS-Bigrams"))


main(sys.argv[1], sys.argv[2])
