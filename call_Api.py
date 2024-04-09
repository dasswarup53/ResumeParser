import requests
import pprint
import json
url ='http://127.0.0.1:9050/upload-cv'
files = {'file': open('/Users/swarupdas/Downloads/SwarupDas-ML-March (1).pdf', 'rb')}
r = requests.post(url, files=files)
pprint.pprint(json.loads(r.text))