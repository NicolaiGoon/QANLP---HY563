import requests
import json

def sparql_query(query_string):
    '''
    code from: https://github.com/cnikas/elas4rdf_qa/blob/main/entity_expansion.py
    '''
    # Execute a query on a SPARQL endpoint
    url = "http://dbpedia.org/sparql"
    payload = {
        "query": query_string,
        "default-graph-uri": "http://dbpedia.org"
    }
    headers = {"Accept":"application/json"}
    response = requests.get(url,params=payload,headers=headers)
    try:
        response_json =  response.json()
        keys = response_json['head']['vars']
        results = []
        for b in response_json['results']['bindings']:
            result = {}
            for k in keys:
                result[k] = b[k]['value']
            results.append(result)
    except json.decoder.JSONDecodeError:
        print("sparql error")
        results = [] 
    return results

def getComment(entity_uri):
    query = ("select ?comment where {{"
                "<{}> rdfs:comment ?comment "
                "filter(lang(?comment) = 'en')}}").format(entity_uri)
    response = sparql_query(query)
    if len(response) != 0:
        return response[0]['comment']
    else:
        return None