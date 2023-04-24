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











"""
    def transform(self, data):
        
        Transforms the data passed as input into a tdf-idf vector/matrix, depending on the input.
        Arguments
        ---------
        data: list of string or string.
            The data to fit the featurizer.
        AttributeError
            Related to the vocabulary lenght. Happens if fit with empty data or not fit.

        if isinstance(data, list):
            return self._transform_document(data)
        elif isinstance(data, str):
            return self._transform_sentence(data)
    
    def _transform_document(self, data):
        #This method is just used for simple batch transforming. 
        to_transform = data
        sentence_arrays = []
        for sentence in data:
            sentence_arrays.append(self._transform_sentence(sentence))
        return np.matrix(sentence_arrays)

    def _transform_sentence(self, data):
        tokens = [token.lower() if self.lower_case else token for token in data.split()]
        # Initializes array with the size of vocabulary.
        word_array = np.zeros(len(self.word_indexes))
        sentence_tf_idf = self._compute_sentence_tf_idf(data)
        # Runs over every token in sentence
        for token in tokens:
            if token in self.word_indexes:
                token_index = self.word_indexes[token]
                # Add the tfidf value for each token in sentence to its position in vocabulary array.
                word_array[token_index] = sentence_tf_idf[token]
        return word_array

    def _compute_sentence_tf_idf(self, sentence):

            #Computes the tf_idf for a single sentence(document).

            sentence_tf_idf = {}
            # Gets the document frequency by using the helper method
            document_frequency = term_frequency(sentence, self.ignore_tokens, self.lower_case)
            # Gets the total number of words in sentence
            total_words = sum(document_frequency.values())
            # Find individual term frequency value averaged by total number of words.
            averaged_frequency = {k:(float(v)/total_words) for k,v in document_frequency.items()}
            for term, tf in averaged_frequency.items():
                # Out of vocabulary words are simply zeroed. They are going to be removed later either way.
                # Computes the tfidf for each word by using word tf times the term idf
                sentence_tf_idf[term] = tf*self.idf_dict.get(term, 0)
            return sentence_tf_idf

"""