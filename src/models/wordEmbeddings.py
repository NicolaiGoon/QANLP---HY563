from sentence_transformers import SentenceTransformer

'''
responsible for creating word embeddings.
We use Sentence-Transformers
'''
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def getWordEmbeddings(sentence):
    return model.encode(sentence)