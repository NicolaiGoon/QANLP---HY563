import pickle
import wordEmbeddings

class qtp:
    '''
  Question Type Prediction
    '''
    model = None
    def __init__(self):
        self.model = pickle.load(open('./sklearn/qtp_model.sav','rb'))

    def predict(self,question):
        embeddings =  wordEmbeddings.getWordEmbeddings(question)
        return self.model.predict([embeddings])[0]
