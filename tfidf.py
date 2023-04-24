import numpy as np
import math
import pandas as pd
import re
from collections import defaultdict
import math

class TFIDF:
    def __init__(self, dataFrame):
        self.fit(dataFrame)
        self.dfRows = self.transformDF(dataFrame)       
        

    def fit(self, dataFrame):
        print('=========== Fitting dataset ===========')
        #total word vocab
        self.words = set()
        #total number of documents
        self.total_documents = dataFrame.shape[0]
        #dict that contains total number of documents words appear in
        self.tdf_dict = defaultdict(int)

        for index, row in dataFrame.iterrows():
            text = row['EmailText']
            #make sentence lowercase
            text = text.lower()
            #find all special characters used in string
            special_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
            #remove all special characters used in string
            text = re.sub('[^a-zA-Z0-9\s]', '', text)
            #find all numbers with words and numbers
            numbers = re.findall(r'\b(?=\w*\d)(?=\d*\w)\w+\b', text)
            #remove all numbers 
            text = re.sub(r'\b(?=\w*\d)(?=\d*\w)\w+\b', '', text)
            #split string into words
            text = text.split(' ')
            
            idf_set = set()

            for word in text:
                if word.isdigit():
                    self.words.add('numberVal') 
                    idf_set.add('numberVal')
                else:
                    self.words.add(word)
                    idf_set.add(word)
            
            for char in special_chars:
                self.words.add(char)
                idf_set.add(char)

            for number in numbers:
                if number.isdigit():
                    self.words.add('numberVal')
                    idf_set.add('numberVal')
                else:
                    self.words.add('wordNumber')
                    idf_set.add('wordNumber')
            
            for word in idf_set:
                self.tdf_dict[word] += 1


    def transformDF(self, dataFrame):
        print('============= Extracting features from dataset ==============')
        # Create an empty dataframe with columns for each word in the vocabulary
        self.vocab_cols = list(self.words)
        self.dfRows = []
        
        for index, row in dataFrame.iterrows():
            text = row['EmailText']
            label = row['Label']

            # Initialize a dictionary to hold the word counts for this row
            word_freq = {word: 0 for word in self.vocab_cols}

            #make sentence lowercase
            text = text.lower()

            #find all special characters used in string
            special_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
            for char in special_chars:
                word_freq[char] += 1
            #remove all special characters used in string
            text = re.sub('[^a-zA-Z0-9\s]', '', text)

            #find all numbers with words and numbers
            numbers = re.findall(r'\b(?=\w*\d)(?=\d*\w)\w+\b', text)
            for number in numbers:
                if number.isdigit():
                    word_freq['numberVal'] += 1
                else:
                    word_freq['wordNumber'] += 1
            #remove all numbers 
            text = re.sub(r'\b(?=\w*\d)(?=\d*\w)\w+\b', '', text)

            #split string into words
            text = text.split(' ')
            for word in text:
                if word.isdigit():
                    word_freq['numberVal'] += 1
                else:
                    word_freq[word] += 1

            for key in word_freq:
                word_freq[key] = word_freq[key] * math.log((self.total_documents + 1) / (self.tdf_dict[key] + 1))
            
            # Add the label value to the word count dictionary
            word_freq['Label'] = label
            self.dfRows.append(word_freq)

        return self.dfRows


    def generateDF(self):
        print('=========== Generating new dataframe with extracted features =================')
        new_df = pd.DataFrame(self.dfRows, columns=['Label'] + [col for col in self.vocab_cols if col != 'Label'])
        new_df = new_df.drop(columns=[''])
        return new_df









