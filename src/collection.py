import requests
import random
from metrics import metrics

class collcetion:

    answerTypes = set(["factoid","definition","confirmation"])
    factoidTypes = set()
    topics = None

    def __init__(self):
        '''
        fetches collection from URL
        '''
        collection  = requests.get("http://users.ics.forth.gr/~mountant/files/project_collection.json")
        collection = collection.json()


        for topic in collection:
            for q in topic["qa"]:
                if (q["type"] == "factoid"): 
                    self.factoidTypes.add(q["entity"])

        self.topics = collection

    def getRandomAnswers(self):
        '''
        returns the prediction for each answer randomly , along with the correct answers
        '''
        results = []
        gold = []
        for topic in self.topics:
            text = topic["text"].split(" ")
            for q in topic["qa"]:
                randomAnswer = random.choice(text)
                results.append(randomAnswer)
                gold.append(q["answer"])
        return results,gold

            

c = collcetion()
pred,gold = c.getRandomAnswers()
print(metrics.exactMatch(pred,gold))