import requests
import pprint
import json
#solution 1
url ='http://127.0.0.1:9050/upload-cv'
files = {'file': open('/Users/swarupdas/Downloads/SwarupDas-ML-April.pdf', 'rb')}
r = requests.post(url, files=files)
pprint.pprint(json.loads(r.text))

#solution 2
# url ='http://127.0.0.1:9050/upload-cv-sota'
# files = {'file': open("/Users/swarupdas/Downloads/MRR.pdf", 'rb')}
# r = requests.post(url, files=files)
# pprint.pprint(json.loads(r.text))
# z=requests.get('http://127.0.0.1:9050/query-cv-rag', params={"question": 'who has good experience in Python ?'})

#solution 3
# url ='http://127.0.0.1:9050/query-cv-rag'
# z=requests.get('http://127.0.0.1:9050/query-cv-rag', params={"question": 'give me the best machine learning engineer profile'})
# pprint.pprint(json.loads(z.text))

while True:
    x=input('''
    ********** Welcome to Manatal Resume Parser Api **********
    
    Please select from the following options
    
    1. CV parsing using Regex,NER & Spacy
    2. CV parsing using Gen Ai (RAG)
    3. QnA Parsing using Gen Ai (RAG)
    4. Exit
    ************************************************************
    ''')
    try:
        option=int(x)

        if option==4:
            break
        if option==1:
            url = 'http://127.0.0.1:9050/upload-cv'
            path=input("SPACY PARSER : enter the url of the resume in pdf format ... ")
            #/Users/swarupdas/Downloads/SwarupDas-ML-April.pdf
            files = {'file': open(path, 'rb')}
            r = requests.post(url, files=files)
            pprint.pprint(json.loads(r.text))

        if option==2:
            url ='http://127.0.0.1:9050/upload-cv-sota'
            path = input("Gen Ai RAG PARSER : enter the url of the resume in pdf format ... ")
            files = {'file': open(path, 'rb')}
            r = requests.post(url, files=files)
            pprint.pprint(json.loads(r.text))

        if option==3:
            url = 'http://127.0.0.1:9050/query-cv-rag'
            question = input("Gen Ai RAG QnA : enter the question ")
            z=requests.get('http://127.0.0.1:9050/query-cv-rag', params={"question":question})
            pprint.pprint(json.loads(z.text))
    except:
        raise Exception("invalid input")
