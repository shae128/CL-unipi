#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import codecs
import math
import nltk
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



######################################################
########### Calculate sentences' probability #########
########### based on a Markov(0) model       #########
######################################################
def cal_markov0(corpusLen, freqDest, sentences):
    probability = 1.0

    for token in sentences:
        tokenProbability = (freqDest[token]*1.0/corpusLen*1.0)
        probability *= tokenProbability

    return probability



####################################################
####### Per Sentence Analysis for each NE ##########
####################################################
def per_sent_analisys(dic, title, corpusSent):

    NE_details = {}
    date_pattern = "([0-9]{2}[-/][0-9]{2}[/-][0-9]{4})"
    days_monthes = ['Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday',
                    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                    'September', 'October', 'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 
                    'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    # tokenizing al corpus
    Total_tokens = []
    for sent in corpusSent:
        toks = nltk.word_tokenize(sent)
        Total_tokens += toks

    # corpus Length
    corpusLen = len(Total_tokens)
    tokensFreqDist = FreqDist(Total_tokens)

    
    print(3*"\n")
    for i in range(8, 40, 4):
        print(i*"*")
    print()
    print("Book Title:", title )
    print()
    for i in range(36, 4, -4):
        print(i*"*")


    print(2*"\n")
    print(44*"=")
    print(u'\u2193'*14, "     ", "Top Ten Person Name Entities with highest frequency".upper(),"     ", u'\u2193'*14)
    print(44*"=")
    print(2*"\n")


    order_number = 1

    for NE in dic:

        places = []
        person = []
        nouns = []
        verbs = []
        dates = []
        sentencesProb = []

        minLen = NE['senteces'][0]
        maxLen = NE['senteces'][0]

        for sentence in NE['senteces']:

            # to find longest and shortes Sentenceis
            if len(sentence) > len(maxLen):
                maxLen = sentence
            if len(sentence) < len(minLen):
                minLen = sentence


            # Analysis for each sentence of each NE
            tokens = nltk.word_tokenize(sentence)
            PosTags = nltk.pos_tag(tokens)
            ne_chunks = nltk.ne_chunk(PosTags)
    
            # finding dates
            date_numbers = re.findall(r"([0-9]{2}[-/][0-9]{2}[/-][0-9]{4})", sentence)
            if len(date_numbers) is not 0:
                dates.append(date_numbers)
            
            # finding days and monthes
            for token in tokens:
                if token in days_monthes:
                    dates.append(token)

            # Finding nouns and verbs
            for token in PosTags:
                if token[1] in ['NN','NNS','NP','NPS']:
                    nouns.append(token[0])
                elif token[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                    verbs.append(token[0])


            # Findig places and person names
            for node in ne_chunks:
                try:
                    node.label()
                except AttributeError:
                    pass
                else:
                    if node.label() == "GPE":
                        for leave in node.leaves():
                            places.append(leave[0])
                    if node.label() == "PERSON":
                        for leave in node.leaves():
                            person.append(leave[0])
 
            places_freq = FreqDist(places)
            person_freq = FreqDist(person)
            nouns_freq = FreqDist(nouns)
            verbs_freq = FreqDist(verbs)


            # Markov(0) 
            if len(sentence) >= 8 and len(sentence) <= 12:
                tempList = []
                tempList.append(sentence)
                tempList.append(cal_markov0(corpusLen, tokensFreqDist, tokens))
                sentencesProb.append(tempList)

        if len(dates) is not 0:
            dates = set(dates)

        if len(sentencesProb) is not 0:
            sentencesProbSorted = sorted(sentencesProb, key=lambda x: x[1])
        else:
            sentencesProbSorted = []
        
        NE_details[NE['Name']] = {
            'Places': places_freq.most_common(10),
            'Person': person_freq.most_common(10),
            'Nouns': nouns_freq.most_common(10),
            'Verbs': verbs_freq.most_common(10),
            'Dates' : dates,
            'Markov' : sentencesProbSorted
        }

        print()
        print(25*"=")
        tempSTR = str(order_number) + ". " + NE['Name'] + " ===> " + str(NE['count']) 
        print(tempSTR)
        print("")

        print(u'\u2193'*3, "     ", "The shortes sentence which contains <".upper(), NE['Name'], ">     ", u'\u2193'*3)
        print("")
        print(minLen)
        print(2*"\n")

        print(u'\u2193'*3, "     ", "The longest sentence which contains <".upper(), NE['Name'], ">     ", u'\u2193'*3)
        print()
        print(4*" ", maxLen)
        print(2*"\n")

        print(u'\u2193'*3, "     ", "Top ten Places in the same sentence with <".upper(), NE['Name'], ">     ",  u'\u2193'*3)
        for pl in NE_details[NE['Name']]['Places']:
            print()
            print(8*" ", pl[0], "===>", pl[1])
        print(2*"\n")
        print(u'\u2193'*3, "     ", "Top ten Person in the same sentence with <".upper(), NE['Name'], ">     ",  u'\u2193'*3)
        for pl in NE_details[NE['Name']]['Person']:
            print()
            print(8*" ", pl[0], "===>", pl[1])
        print(2*"\n")

        print(u'\u2193'*3, "     ", "Top ten Nouns in the same sentence with <".upper(), NE['Name'], ">     ",  u'\u2193'*3)
        for pl in NE_details[NE['Name']]['Nouns']:
            print()
            print(8*" ", pl[0], "===>", pl[1])
        print(2*"\n")

        print(u'\u2193'*3, "     ", "Top ten Verbs in the same sentence with <".upper(), NE['Name'], ">     ",  u'\u2193'*3)
        for pl in NE_details[NE['Name']]['Verbs']:
            print()
            print(8*" ", pl[0], "===>", pl[1])
        print(2*"\n")

        print(u'\u2193'*3, "     ", "Top ten Dates in the same sentence with <".upper(), NE['Name'], ">     ",  u'\u2193'*3)
        for pl in NE_details[NE['Name']]['Dates']:
            print()
            print(8*" ", pl)
        print(2*"\n")

        print(u'\u2193'*3, "     ", " Markov(0) sentence with highest probability ".upper(), "     ",  u'\u2193'*3)
        if len(NE_details[NE['Name']]['Markov']) is not 0:
            print()
            print(8*" ", "Sentence:", NE_details[NE['Name']]['Markov'][0][0])
            print(8*" ", "probability:", "===>", NE_details[NE['Name']]['Markov'][0][1])
        print(2*"\n")
        print(25*"=")


        order_number += 1


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

    #tokenizing sentences
    sentences1 = sent_tokenizer.tokenize(rawFile1)
    sentences2 = sent_tokenizer.tokenize(rawFile2)

    # Find Person names
    topTenPerson1 = topTenNE(sentences1)
    topTenPerson2 = topTenNE(sentences2)

    # Analysis of Name Entities and their Sentences
    per_sent_analisys(topTenPerson1, file1[:-4], sentences1)
    per_sent_analisys(topTenPerson2, file2[:-4], sentences2)


main(sys.argv[1], sys.argv[2])
