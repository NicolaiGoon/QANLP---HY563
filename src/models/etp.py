import pickle

class etp:
    '''
  Entity Type Prediction
    '''
    model = None
    def __init__(self):
        self.model = pickle.load(open('./sklearn/etp_model.sav','rb'))

    def predict(self,embeddings):
      return self.model.predict(embeddings)
