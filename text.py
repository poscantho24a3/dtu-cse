import os
import json
import datetime
import numpy as np
import glob
import time
import re
import threading

from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer

class NNClassification():
    def __init__(self, path_training=""):
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?']
        self.training = []
        self.output = []
        self.output_empty = []
        self.w_0 = np.array([])
        self.w_1 = np.array([])
        self.w_file = 'ws.json' 
        self.ERROR_THRESHOLD = 0.4
        self.PATH = path_training
        self.get_data_for_training()
        pass
    
    def load_w(self):
        if os.path.isfile(self.w_file):
            with open(self.w_file) as data_file: 
                w = json.load(data_file) 
                self.w_0 = np.asarray(w['w0']) 
                self.w_1 = np.asarray(w['w1'])

    def get_data_for_training(self):
        with open(self.PATH, encoding='utf-8') as json_file:  
            training_data = json.load(json_file)
            for pattern in training_data:
                wTV = ViPosTagger.postagging(ViTokenizer.tokenize((pattern['sentence'])))
                self.words.extend(wTV[0])
                self.documents.append((wTV[0], pattern['class']))
                if pattern['class'] not in self.classes:
                    self.classes.append(pattern['class'])

        self.output_empty = [0] * len(self.classes)
        self.words = [w.lower() for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        # prepare for predicting
        self.load_w()

    # compute sigmoid nonlinearity
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def clean_up_sentence(self, sentence):
        sentence_words = ViPosTagger.postagging(ViTokenizer.tokenize((sentence)))[0]
        sentence_words = [word.lower() for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def predict(self, sentence, show_details=False):
        x = self.bow(sentence.lower(), self.words, show_details)
        if show_details:
            print ("sentence:", sentence, "\n bow:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = self.sigmoid(np.dot(l0, self.w_0))
        # output layer
        l2 = self.sigmoid(np.dot(l1, self.w_1))

        return l2

    def train(self, hidden_neurons=10, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2):
        # alpha: learning rate.
        np.random.seed(1)
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [word.lower() for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            self.training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(self.output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)


        X = np.array(self.training)
        y = np.array(self.output)
        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(self.classes)) )
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        w_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        w_1 = 2 * np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_w_0_weight_update = np.zeros_like(w_0)
        prev_w_1_weight_update = np.zeros_like(w_1)

        w_0_direction_count = np.zeros_like(w_0)
        w_1_direction_count = np.zeros_like(w_1)
            
        for j in iter(range(epochs+1)):

            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, w_0))
                    
            if dropout:
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, w_1))

            layer_2_error = y - layer_2
            if (j % 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break
                    
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(w_1.T)

            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)
            
            w_1_weight_update = (layer_1.T.dot(layer_2_delta))
            w_0_weight_update = (layer_0.T.dot(layer_1_delta))
            
            if(j > 0):
                w_0_direction_count += np.abs(((w_0_weight_update > 0)+0) - ((prev_w_0_weight_update > 0) + 0))
                w_1_direction_count += np.abs(((w_1_weight_update > 0)+0) - ((prev_w_1_weight_update > 0) + 0))        
            
            w_1 += alpha * w_1_weight_update
            w_0 += alpha * w_0_weight_update
            
            prev_w_0_weight_update = w_0_weight_update
            prev_w_1_weight_update = w_1_weight_update

        now = datetime.datetime.now()

        # persist ws
        w = {
            'w0': w_0.tolist(), 
            'w1': w_1.tolist(),
            'datetime': now.strftime("%Y-%m-%d %H:%M"),
            'words': self.words,
            'classes': self.classes
        }
        w_file = "ws.json"

        with open(w_file, 'w') as outfile:
            json.dump(w, outfile, indent=4, sort_keys=True)
        print("saved ws to:", w_file)
        
        self.w_0 = w_0
        self.w_1 = w_1

    def classify(self, sentence, show_details=False):
        results = self.predict(sentence, show_details)
        print(results)
        results = [[i,r] for i,r in enumerate(results) if r > self.ERROR_THRESHOLD ] 
        results.sort(key=lambda x: x[1], reverse=True) 
        return_results = [self.classes[r[0]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

# if __name__ == "__main__":
    # nn = NNClassification("data/complains.json")
    # nn.train()
    # nn.classify('cà phê sao dở vậy?')
