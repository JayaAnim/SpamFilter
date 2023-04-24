
import pandas as pd  # for reading data from csv 
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
import numpy as np  # for handling multi-dimensional array operation
from bow import BoW
from utils import remove_low_variance_features, select_k_best_features, perform_pca
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf import TFIDF

class SVM:
    # Set regularization strength and learning rate
    regularization_strength = 10000
    learning_rate = 0.000001

    def __init__(self, featureParams):
        data = pd.read_csv('spam.csv')

        #featureExtractor = BoW(data)
        #newDF = featureExtractor.generateDF()
        #featureExtractor = TFIDF(data)
        #newDF = featureExtractor.generateDF()

        newDF = self.extractDF(data, featureParams)
        

        #prepare data sets for training and testing
        X_train, X_test, y_train, y_test = self.prepareSets(newDF, featureParams)

        #get weights for SVM
        weights = self.trainSVM(X_train, y_train)

        #test the SVM
        self.testSVM(weights, X_test, y_test)

    #extracts data using user specified method (bag of words or tf-idf)
    def extractDF(self, data, featureParams):
        if featureParams.extract == 1:
            print('========= Extracting dataset using BoW =========')
            featureExtractor = BoW(data)
            return featureExtractor.generateDF()
        elif featureParams.extract == 2:
            print('=========== Extracting dataset using TD-IDF =============')
            featureExtractor = TFIDF(data)
            return featureExtractor.generateDF()
        else:
            print('=========== Extractor error: no extractor specified ============')
            return None
        
        
    #Prepares the training and testing datasets after features have been extracted
    def prepareSets(self, data, featureParams):

        # Enumerate labels
        label_map = {'ham':1, 'spam':-1}
        data['Label'] = data['Label'].map(label_map)

        # Split the data into features (X) and output (Y)
        Y = data.loc[:, 'Label'] 
        X = data.iloc[:, 1:]

        
        #select features using userspecified method
        X = self.selectFeatures(X, Y, featureParams)

        # Normalize the features... NORMALIZE! NORMALIZE! NORMALIZE!
        X_normalized = MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(X_normalized)

        # Add intercept column
        X.insert(loc=len(X.columns), column='intercept', value=1)

        # Split the data into training and testing sets
        print("============Splitting dataset into train and test sets===========")
        X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=19)
        return X_train, X_test, y_train, y_test
    
    #Selects features using userspecified method (remove_low_variance_features, select_k_best_features, perform_pca)
    def selectFeatures(self, X, Y, featureParams):
        if featureParams.select == 1:
            return remove_low_variance_features(X, featureParams.threshold)
        elif featureParams.select == 2:
            return select_k_best_features(X, Y, featureParams.k)
        elif featureParams.select == 3:
            return perform_pca(X, featureParams.n)
        elif featureParams.select == 4:
            return X
        else:
            print('============== Feature selection error: no selection method specified =============')
            return None

        

    
    #traings the svm model use the stochastic gradient descent function
    def trainSVM(self, X_train, y_train):
        # Train the model using stochastic gradient descent (SGD)
        print("=======Training started======")
        weights = self.sgd(X_train.to_numpy(), y_train.to_numpy())
        print("=======Training finished======")
        return weights
    

    #uses stochastic gradient descent to minimize function and calculate weights
    def sgd(self, features, outputs):
        #number of iterations to minimize function
        max_epochs = 2000

        weights = np.zeros(features.shape[1])

        # Stochastic Gradient Descent
        for epoch in range(1, max_epochs): 
            # Shuffle to prevent repeating update cycles
            features, outputs = shuffle(features, outputs)
            for index, feature in enumerate(features):
                ascent = self.calcGradient(weights, feature, outputs[index])
                weights = weights - (SVM.learning_rate * ascent)
                
        return weights
    

    #calculates gradient for stochastic gradient descent function
    def calcGradient(self, weights, feature_batch, output_batch):
        # If only datapoint is passed make it iterable
        if type(output_batch) == np.int64:
            output_batch = np.array([output_batch])
            feature_batch = np.array([feature_batch])

        distance = 1 - (output_batch * np.dot(feature_batch, weights))
        
        # verify datapoints are iterable (has prevented errors)
        if type(distance) == np.float64:
            distance = np.array([distance])
        if type(output_batch) == np.int64:
            output_batch = np.array([output_batch])
        if type(feature_batch) == np.int64:
            feature_batch = np.array([feature_batch])
        
        # Calculate cost gradient
        gradient = np.zeros(len(weights))
        for index, d in enumerate(distance):
            if max(0, d) == 0:
                di = weights
            else:
                di = weights - (SVM.regularization_strength * output_batch[index] * feature_batch[index])
            gradient += di

        gradient = gradient/len(output_batch)  # average
        return gradient

    #tests trained svm against testing set
    def testSVM(self, weights, X_test, y_test):
        # Make predictions using test set
        y_predictions = np.array([])
        for i in range(X_test.shape[0]):
            # Predict the output for a single data point
            y_pred = np.sign(np.dot(weights, X_test.to_numpy()[i]))
            y_predictions = np.append(y_predictions, y_pred)
        
        # Calculate accuracy of the model on the test set
        accuracy = accuracy_score(y_test.to_numpy(), y_predictions)
        print(f'SVM prediction accuracy: {accuracy}')


    