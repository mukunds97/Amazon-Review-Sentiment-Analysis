import os
import re
import sys
import pickle
import pandas as pd
from gensim.models import Word2Vec

def read_file(file_path):
    with open(file_path) as f:
        file_content = f.read().split('\n')[:-1]
    return file_content

class vectorizer():
    def __init__(self, reviews):
        self.dataset = reviews
        self.tokens = None
        self.model = None
        self.model_path = os.path.join(os.getcwd(),'data','w2v.model')

    def pre_process(self):
        ''' Removing of Special Characters '''
        pattern = """[!"#$%&().,/?*+/:;<=>@[\\]^`{|}~\t\n-]+""" 
        self.dataset = [re.sub(pattern, ' ',line).lower() for line in self.dataset]
        data = []
        for line in self.dataset:
            data.append(line.split())
        self.tokens = data

    def vectorize(self):
        self.model = Word2Vec(self.tokens, window=10)

    def save_model(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))

    def read_model(self):
        return self.model
    
    def pred_similar_words(self, text, num):
        ''' Finds n number of similar words from vocabulary when given a word'''
        similar_words = self.model.wv.most_similar(text.lower(), topn=num)
        similar_words_df = pd.DataFrame(similar_words)
        similar_words_df.columns = ['Word', 'Similarity Score']
        return(similar_words_df)



def main():
    review_directory = 'data/'
    pos_reviews = read_file(os.path.join(review_directory,'pos.txt'))
    neg_reviews = read_file(os.path.join(review_directory,'neg.txt'))
    review_corpus = pos_reviews + neg_reviews

    word_vectorizer = vectorizer(review_corpus)
    word_vectorizer.pre_process()
    word_vectorizer.vectorize()
    word_vectorizer.save_model()
    print(word_vectorizer.pred_similar_words('Good', 20))
    print(word_vectorizer.pred_similar_words('Bad', 20))

if __name__ == '__main__':
    main()