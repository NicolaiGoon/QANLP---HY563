from sentence_transformers import SentenceTransformer,util

'''
https://huggingface.co/tasks/sentence-similarity
'''
class cq_minilm:
    def __init__(self):
        self.model =  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def getSimilarity(self,sentence1,sentence2):
        return util.pytorch_cos_sim(
            self.model.encode(sentence1,convert_to_tensor=True),
            self.model.encode(sentence2,convert_to_tensor=True)
            )

        