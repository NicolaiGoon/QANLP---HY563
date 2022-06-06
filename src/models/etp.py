import pickle

from torch import embedding
import wordEmbeddings

class etp:
    '''
  Entity Type Prediction
    '''
    model = None
    def __init__(self):
        self.model = pickle.load(open('./sklearn/etp_model.sav','rb'))

    def predict(self,question):
        embeddings =  wordEmbeddings.getWordEmbeddings(question)
        return self.model.predict([embeddings])[0]