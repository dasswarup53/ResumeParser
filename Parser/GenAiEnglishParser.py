import os
import json
import openai
# import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

directory = 'cv'

def load_docs(directory):
  loader = DirectoryLoader(directory,show_progress=True) #unstructuredloader by default has used this auto identify file type and load it, mode="single", strategy='fast'(other option is strategy='hi_res' that use yolo varient if mode is elements)
  documents = loader.load()
  return documents

def split_docs(documents, chunk_size=3000, chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# documents = load_docs(directory)
# len(documents)

def parse_return(txt):

    key=''

    # docs = split_docs(documents)
    # print(len(docs))

    os.environ["OPENAI_API_KEY"] = key

    embeddings = OpenAIEmbeddings(openai_api_key=key)


    persist_directory = 'db'

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    embedding = OpenAIEmbeddings()

    # vectordb = Chroma.from_documents(documents=docs,
    #                                  embedding=embedding,
    #                                  persist_directory=persist_directory)
    # persiste the db to disk
    # vectordb.persist()
    vectordb = None

    # Now we can load the persisted database from disk, and use it as normal.
    # vectordb = Chroma(persist_directory=persist_directory,
    #                   embedding_function=embedding)
    #
    # retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # by default search_type="similarity_score_threshold"

    # docs = retriever.invoke("how much years of experience Rohit has.")
    # print(docs[0].page_content)

    # turbo_llm = ChatOpenAI(
    #     temperature=0,
    #     model_name='gpt-3.5-turbo'
    # )

    # create the chain to answer questions
    # qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
    #                                   chain_type="stuff",
    #                                   retriever=retriever,
    #                                   return_source_documents=True)

    # query="what are the educational qualifications of Swarup."
    # llm_response = qa_chain(query)
    # process_llm_response(llm_response)

    # warning = "If you don't know the answer, just say that you don't know, don't try to make up an answer"
    # job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
    # Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
    # Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
    # Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
    # Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
    # Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
    # Strong communication and collaboration skills"
    # question = warning+job_description + " Based on the given job description"
    # warning = "If you don't know the answer, just say that you don't know, don't try to make up an answer"
    # job_description = "MS or PhD in Data science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
    # Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
    # Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
    # Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
    # Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
    # Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
    # Strong communication and collaboration skills"
    # question = warning+job_description + " Based on the given job description"
    # query = question + "retrive the full document information of a resume which is good fit based on skills,education and work experience mwntioned in it? "
    # query="retrieve the full text for Swarup"
    # resume_doc = retriever.invoke(query)


    review_template = """\
    For the following text, extract the following information:
    
    Skills: what are the skills? \
    Answer them as a comma separated Python list.
    
    Education: What are the educational qualifications of the candidate?\
    Answer them as a comma separated Python list.
    
    
    Work experience: Extract all organisation name where he/she has worked and also extract the designation, date of tenure and a short summary of the work they did over there\
    Answer them as a JSON with keys like designation, company name, summary, start-date and end-date.
    
    Format the output as JSON with the following keys:
    Skills
    Education
    Work experience
    
    text: {text}
    """



    prompt_template = ChatPromptTemplate.from_template(review_template)


    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "Hi"},
                        {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    memory.load_memory_variables({})

    turbo_llm_memory = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo'
    )


    memory_llm_conversation = ConversationChain(
        llm=turbo_llm_memory,
        memory = memory,
        verbose=True
    )

    # txt=''' Swarup Das
    # Team Lead | MSc. Data Science | Machine Learning | MLOps | Data Engineering
    # Pune, India • +91 9503554884 • dasswarup53@gmail.com • https://www.linkedin.com/in/swarupd/
    #
    # Data and ML lead with six years experience in building and productionizing machine learning models and ETL
    # pipelines at scale. My forte resides in building scalable & machine learning powered end to end products.
    #
    # WORK EXPERIENCE
    # _______________________________________________________________________________________________________________
    #
    # Lead Machine Learning Engineer (Fintech)
    # OneCard by FPL Technologies, Pune, India
    # Built amazing products like OneScore (20 Million + Users) & OneCard (~ 2 Million Users). Startup gained its unicorn
    # tag in just 3 years. Led the team of engineers to create Data Lake, ML Workbenches & Analytics Platform.
    #
    # March 2020 – Present
    #
    # ●
    #
    # Trained and containerised a real time fraud detection system using top notch regression strategies that
    # mitigated frauds by 98%
    #
    # ● Built a dynamic offer recommendation engine using location based intelligence and real time data
    #
    # ingestion pipelines.
    #
    # ● Tuned BERT , LLAMA2 (entity extraction & sentiment analysis) to improve CX experience and boost ﬁrst
    #
    # call resolution by 45%. (currently exploring other LLMs as well)
    #
    # ● Reduced signup drop-offs from 65% to 15% and increased user engagement by 40%, through a
    #
    # combination of hypothesis testing, segmentation analysis and A/B testing.
    #
    # ● Developed and deployed models using Ensemble Learning & Feature Engineering methods that powers
    #
    # Risk, Demand Forecasting, Onboarding and Recommendation Engines
    #
    # ● Penned highly eﬃcient time series models using SARIMAX, RNNs for spends projection, payment defaults
    #
    # and anomaly/outlier detection.
    #
    # ● Engineered an ML Platform that was featured by AWS .(Click Here).
    # ● Developed POCs for inhouse projects using Deep Learning, Clustering , Natural Language Processing and
    #
    # Computer Vision (YoloV8, DeepFace).
    #
    # ● Penned & Owned AWS MLOps pipelines using Docker, AirFlow, MlFlow, GitOps etc. that automatically
    #
    # trains, tunes and deploys 120+ risk, credit scoring, churn and LTV models.
    #
    # ● Programmed highly scalable and secure Flask Api pipelines for taking ML models to production
    #
    # environments.
    #
    # ● Crafted several dashboards for model performance tracking , governance and drift monitoring using
    #
    # business critical and statistical metrics along with tools like Prometheus and Grafana.
    #
    # ● Mastered complex SQL query development and performance tuning/optimization using partitions
    # ● Solved the resource starvation problem by implementing On Demand Containerization.
    # ● Leading the migration from AWS Data Lake to Delta Lake using databricks (125s+ TBs of Data).
    # ● Automated the model maintenance and monitoring using AWS Sagemaker to tackle data distribution drift.
    # Led the ﬁrst major effort to build the serverless data lake that served as the base of the analytics platform
    # ●
    # and it also improved the operational eﬃciency by 35%
    #
    # ● Crafted highly scalable AWS Glue ETL pipelines using Hive,Hudi & Spark that updates 1200+ SQL tables on
    #
    # a daily basis which revolutionized the entire data warehouse stack.
    #
    # ● Performed R&D on data storage systems and compression techniques using Delta, Hudi, Iceberg formats
    #
    # to optimize the storage and querying costs that overall reduced the operational cost by 45%.
    #
    # ● Built the governance and auditing module that is compliant with partner banks, on top of our existing data
    #
    # infrastructure to ensure data consistency, reliability and availability.
    #
    # ● Collaborated closely with CXO’s, data analysts and domain experts to build highly scalable & secure data
    #
    # products to tackle challenging business problems.
    #
    # ● Mentoring & grooming data analysts for technical queries and architecturing complex ML pipelines.
    #
    #
    #  Data Scientist
    # Rule 14 LLC, Pune, India
    # First hire of the data science team and was also responsible for team building and heading initial client projects
    #
    # Jan 2019 – Feb 2020
    #
    # ● Worked on Text Summarisation using Encoder Decoder models, forecasting using ARIMA/SARIMAX
    # ● Developed a variety of in house projects using PCA, XGboost, Random Forest, Text Rank algorithms
    #
    # Software Engineer Trainee
    # Searce LLC, Pune, India
    #
    # July 2018 – Dec 2018
    #
    # ● Worked on Image captioning (using VGG16), Model Blending and learnt best practices like CI/CD, Canary
    #
    # Deployment, Test Driven Development etc
    #
    # EDUCATION
    # _______________________________________________________________________________________________________________
    #
    # ● MSc Data Science (Year 2021), Liverpool John Moores University, Grade : Distinction
    # ● Bachelor of Engineering Computer (Year 2018), Pune University, Grade : Distinction
    #
    # SKILLS & OTHER
    # _______________________________________________________________________________________________________________
    #
    # ML Algorithms: RandomForest, XGBoost, K-Means, SVM, Naive Bayes, PCA, Linear & Logistic Regression,
    # Libraries: Keras, PyTorch,Tensorﬂow, Airﬂow, Sklearn, Seaborn, Matplotlib, Numpy, Pandas, Sanic, Flask,FastAPI,
    # Pyspark, Git
    # Big Data/DataBases: Hive, Hudi, Spark, IceBerg, Kafka, MySql, PostgreSql, BigQuery, DataBricks
    # Cloud: AWS EC2, RDS,DMS,Lambda, Glue,Studio, Event Bridge, Sagemaker, Athena, Kinesis, S3, ECR, Docker,
    # Kubernetes
    # Languages: Python, SQL, Java'''
    messages = prompt_template.format_messages(text=txt)
    # chat = ChatOpenAI(temperature=0.0, model=turbo_llm_memory)
    response = memory_llm_conversation(messages)
    print("++++")
    print(response['response'])
    print("++++")

    from langchain.output_parsers import ResponseSchema
    from langchain.output_parsers import StructuredOutputParser

    skills_schema = ResponseSchema(name="Skills",
                                   description="what are the  skills? \
    Answer output them as a comma separated Python list.")

    Projects_schema = ResponseSchema(name="Education",
                                     description="What are the educational qualifications of the candidate?\
    Answer output them as a comma separated Python list.")

    Work_experience_schema = ResponseSchema(name="Work experience",
                                            description="Extract all organisation name where he/she has worked and also extract designation\
    Answer output them as a comma separated Python list.")

    response_schemas = [skills_schema,

                        Projects_schema,

                        Work_experience_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    # print(format_instructions)

    prompt = ChatPromptTemplate.from_template(template=review_template)

    messages = prompt.format_messages(text=txt,
                                    format_instructions=format_instructions)

    response2 = turbo_llm_memory(messages)

    return json.loads(response2.content)