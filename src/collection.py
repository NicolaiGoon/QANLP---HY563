import requests
import random
from metrics import metrics
from models.wordEmbeddings import getWordEmbeddings
from WekaApi import createWeka
import pandas as pd

class collection:

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
                    entity = q["entity"]
                    # reduce multiple classes
                    if(entity == "GPE" or entity == "ORG" or entity == "ORG, GPE"):
                       self.factoidTypes.add("ORG-GPE")
                       q["entity"] = "ORG-GPE"
                    elif(entity == "TIME" or entity == "DATE" or entity == "DATE, TIME"):
                       self.factoidTypes.add("DATE-TIME")
                       q["entity"] = "DATE-TIME"
                    else:
                        self.factoidTypes.add(entity)

        self.topics = collection
        print(self.answerTypes)
        print(self.factoidTypes)

    def getRandomAnswers(self):
        '''
        returns the prediction for each answer randomly , along with the correct answers
        '''
        results = {'factoid':[],'definition':[],'confirmation':[]}
        gold = {'factoid':[],'definition':[],'confirmation':[]}
        for topic in self.topics:
            text = topic["text"].split(" ")
            for q in topic["qa"]:
                randomAnswer = random.choice(text)
                results[q["type"]].append(randomAnswer)
                gold[q["type"]].append(q["answer"])
        return results,gold

    def getFeaturesAndLabels(self):
        '''
        returns the features and the labels for:
        Question Type Prediction
        Entity Type Prediction
        '''
        Xquestion = []
        Yquestion = []
        Xentity = []
        Yentity = []
        typeClasses = ",".join([x for x in self.answerTypes])
        entityClasses = ",".join([x for x in self.factoidTypes])
        for t in self.topics:
            for qa in t['qa']:
                qEmbeddings = getWordEmbeddings(qa['question'])
                Xquestion.append(qEmbeddings)
                Yquestion.append(qa['type'])
                # only for factoid
                if(qa['type'] == 'factoid'):
                    Xentity.append(qEmbeddings)
                    Yentity.append(qa['entity'])
        return Xquestion,Yquestion,Xentity,Yentity

    def exportAsWeka(self):
        '''
        exports the collection as two Weka files for:
        Question Type Prediction
        Entity Type Prediction
        '''
        Xquestion,Yquestion,Xentity,Yentity = self.getFeaturesAndLabels()
        typeClasses = ",".join([x for x in self.answerTypes])
        entityClasses = ",".join([x for x in self.factoidTypes])
        createWeka('./Weka/qtypes',X=Xquestion,Y=Yquestion,classes=typeClasses)
        createWeka('./Weka/etypes',X=Xentity,Y=Yentity,classes=entityClasses)

    def exportDataCsv(self):
        '''
        exports the collection as CSV for Machine Learning:
        Question Type Prediction
        Entity Type Prediction
        '''
        Xquestion,Yquestion,Xentity,Yentity = self.getFeaturesAndLabels()
        
        Qdf = []
        for x,y in zip(Xquestion,Yquestion):
            row = list(x)
            row.append(y)
            Qdf.append(row)
        Qdf = pd.DataFrame(Qdf)
        Qdf = Qdf.set_axis([*Qdf.columns[:-1],"class"],axis=1,inplace=False)

        edf = []
        for x,y in zip(Xentity,Yentity):
            row = list(x)
            row.append(y)
            edf.append(row)
        edf = pd.DataFrame(edf)
        edf = edf.set_axis([*edf.columns[:-1],"class"],axis=1,inplace=False)

        Qdf.to_csv('./sklearn/questions.csv',index=None)
        edf.to_csv('./sklearn/entities.csv',index=None)
