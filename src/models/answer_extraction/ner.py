from html import entities
import requests


'''
Named entity Recognision from text using DBpedia Spotlight
'''

def getEntities(text):
    headers = {"Accept": "application/json; charset=utf-8"}
    response = requests.get("https://api.dbpedia-spotlight.org/en/annotate?text="+text,headers=headers).json()
    entities = []
    # take the first entity
    if("Resources" in response):
        for e in response['Resources']:
            entities.append(e['@URI'])
            break
    return entities
