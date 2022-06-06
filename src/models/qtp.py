import pickle

class qtp:
    '''
  Question Type Prediction
    '''
    model = None
    def __init__(self):
        self.model = pickle.load(open('./sklearn/qtp_model.sav','rb'))

    def predict(self,embeddings):
      return self.model.predict(embeddings)
