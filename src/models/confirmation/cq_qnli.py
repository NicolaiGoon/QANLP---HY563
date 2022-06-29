from sentence_transformers import CrossEncoder

class cq_qlni:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/qnli-distilroberta-base')

    def getSimilarity(self,text1,text2):
        scores = self.model.predict([(text1,text2)])
        return scores[0]

    def getSimilarities(self,text1,texts):
        scores = self.model.predict([(text1,x) for x in texts])
        return sorted({x:i for x,i in zip(texts,scores)}.items(),key=lambda x: x[1],reverse=True)