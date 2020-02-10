from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.datasets import imdb
import numpy as np

word_idx = imdb.get_word_index()

data = ['Indian cricket team will wins World Cup, says Capt. Virat Kohli','We will win next  Lok Sabha Elections, says confident Indian PM','The nobel laurate won the hearts of the people','The movie Raazi is an exciting Indian Spy thriller based upon a real story']

def tss(sent):
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sent) # Now we have the tokenized sentence
    
    # Now we will perform stopword elimination
    sw = set(stopwords.words('english'))
    useful_words = [w for w in words if w.lower() not in sw]
    
    # Now we will do stemming using snowball stemmer
    #ss = SnowballStemmer('english')
    
    #useful_words = [ss.stem(w) for w in useful_words]
    useful_words = set(useful_words)
    useful_words = list(useful_words)
    
    return useful_words

def vectorize_sentence(sentence, dims=15000):
    
    r = [word_idx.get(word, "#") for word in sentence]
    r = np.array(r)
    r_ = np.delete(r, np.where(r == '#'))
    r_ = np.array([int(num) for num in r_])
    r_ = np.delete(r_, np.where(r_ > 15000))
    
    
    output = np.zeros((dims,))
    output[r_] = 1
    output = output.astype('int32')
    output = output.reshape((1,dims)) # We need to resize it because to the model we have to insert into matrix of form each review in each row/ vector. if we have to give just one review the matrix to be given is of shape (1,15000) ==> 1 review vectorized to 15000 words.    
    return output

