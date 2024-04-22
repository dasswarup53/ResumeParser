import requests
import pprint
import json
url ='http://127.0.0.1:9050/upload-cv-sota'
files = {'file': open('/Users/swarupdas/Downloads/Shivam_Resume_.pdf', 'rb')}
# r = requests.post(url, files=files)
# pprint.pprint(json.loads(r.text))


z=requests.get('http://127.0.0.1:9050/query-cv-rag', params={"question": 'who has good experience in Python ?'})

pprint.pprint(json.loads(z.text))
# pprint.pprint(json.loads(z.text))