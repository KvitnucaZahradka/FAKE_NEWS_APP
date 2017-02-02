import pickle
import re
from collections import Counter
from textblob import TextBlob
import math
from flaskapp import nlp_optimized
from flaskapp import predict

word = re.compile(r'\w+')


# opening the file from the function
def safely_open(name_of_saved_file, pick):
    try:
        if pick:
            with open(name_of_saved_file + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        else:
            with open(name_of_saved_file, 'rb') as handle:
                return pickle.load(handle)
    except FileNotFoundError:
        print("File " + name_of_saved_file + " was not found ")


# this function turns the text into a vector
def text_to_vector(text):
    text = relevant_words(text)
    words = word.findall(text)
    return Counter(words)


def relevant_words(text):
    blob = TextBlob(text)
    tags = blob.tags
    return " ".join([t[0] for t in tags if ((t[1] == "NN") or (t[1] == "JJ"))])


# sigmoid
def sigmoid(x, alpha=0.3, shift=2000):
    return (1 / (1 + math.exp(-alpha*(x - shift))))
