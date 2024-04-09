# ResumeParser
### This is the resume parser-api that accepts in PDF and returns the extracted fields  in JSON.

### Currently it can extract the following fields 
### sample output :

```json
{'contact_info': {'email': 'dasswarup53@gmail.com',
                  'phone': '9503554884',
                  'websites': ['https://www.linkedin.com/in/swarupd/']},
 'education': {'college': ['Liverpool John Moores University'],
               'degree': ['MSc Data Science',
                          'Bachelor of Engineering Computer (Year 2018']},
 'metadata': {'job_pk': 12,
              'language-code': 'en',
              'language-confidence': 0.9999960436887447,
              'remark': 'success',
              'status': 'succ'},
 'personal': {'location': 'Pune, India', 'name': 'Swarup Das'},
 'skills': ['Matplotlib',
            'Postgresql',
            'Xgboost',
            'Metrics',
            'Segmentation',
            'Pandas',
            'Api',
            'Startup',
            'System',
            'Governance',
            'Analysis',
            'Keras',
            'Python',
            'Java',
            'Engineering',
            'Engagement',
            'Mysql',
            'Migration',
            'Forecasting',
            'Updates',
            'Flask',
            'Sql',
            'Analytics',
            'Aws',
            'Cloud',
            'Seaborn',
            'Sanic',
            'Warehouse',
            'Tensorflow',
            'Docker',
            'Etl',
            'Hive',
            'Algorithms',
            'Numpy',
            'Pyspark',
            'Auditing',
            'Queries',
            'Testing',
            'Technical',
            'Spark',
            'Hypothesis'],
 'work_exp': [{'designation': 'Lead Machine Learning Engineer (Fintech)'},
              {'designation': 'Data Scientist'},
              {'designation': 'Software Engineer Trainee'}]}

```
### This parser employs Regex , NER Tagging  and a tuned Spacy Model to extract the fields

### Deliverables

The deliverable should include the following:

1. Definition of the accuracy of the resume parser → How do we evaluate success?
->Parsing resumes is typically a classification problem , to be more precise it is a multiclass classification problem.
we can use metrics like F1 Score, Precision,Recall, ROC- AUC to evaluate the performance . While training our model, the best performing model is the one which had the highest F1 Score.
```
E    #       LOSS TRANS...  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
---  ------  -------------  --------  ------  ------  ------  ------
  0       0        9238.97   1544.76    0.20    0.10    3.45    0.00
  3     200      199317.57  66695.30   32.15   39.21   27.24    0.32
  6     400       58940.22  23063.40   55.83   59.72   52.41    0.56
 10     600       11210.37  20986.48   54.71   52.11   57.59    0.55
 13     800        8272.94  16835.60   44.18   70.99   32.07    0.44
 16    1000       11448.79  16229.70   53.76   48.03   61.03    0.54
 20    1200        6256.43  16764.19   55.08   64.43   48.10    0.55
 23    1400        6140.25  14505.28   56.56   61.41   52.41    0.57
 26    1600        6545.44  14061.78   60.71   70.55   53.28    0.61 ---- Best Model
 30    1800         586.02  14062.91   58.19   58.80   57.59    0.58
 33    2000        4867.82  12252.47   54.72   69.79   45.00    0.55
 36    2200       44495.01  12424.94   60.04   65.84   55.17    0.60
 40    2400         831.39  12326.80   60.22   63.95   56.90    0.60
 43    2600        8369.49  10585.22   58.05   68.54   50.34    0.58
 46    2800         731.13  10233.91   56.37   67.39   48.45    0.56
 50    3000         254.50  10085.33   58.71   61.24   56.38    0.59
 53    3200       15952.13   8614.01   56.12   66.35   48.62    0.56
```
2. The architecture of the Resume Parser (high-level) - How does it work?
(e.g. holistic approach, deep learning models, etc.)
   1. The Parser first tries to convert pdf to text 
   2. then  it tries to detect the langugae of the doc.
   3. After that it performs extraction/prediction using Regex, NER  and custom spacy model
   4. Post that the results are saved into the SQLite DB
3. Milestones to complete the project with an estimated timeline 

    This is an initial implementation , would refine  more depending on the feedback
5. Basic implementation of the Resume Parser
    1. Simple API service (e.g. FastAPI service)
    2. Resume Parser (e.g. Python)

### Gen Ai Solution:
#### Using ChatGPT, Lang Chain & RAG I am developing a solution, would upload it before our discussion
