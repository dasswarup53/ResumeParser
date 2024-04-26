from fastapi import Request
from timeit import default_timer as timer
from datetime import timedelta
import os
import spacy
import pandas as pd
import warnings
from db import engine
from Models import models as models
from Models.models import  Parser_Record
from sqlalchemy.orm import sessionmaker
import uvicorn
from fastapi import FastAPI, UploadFile

from Parser.EnglishResumeParser import EnglishResume
from logging.config import dictConfig
import logging
from utils.log_config import log_config
from Parser.GenAiEnglishParser import llm_parse_return

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils.service_helper import *
#####
# RAG init

documents = load_docs('cv-directory')

docs = split_docs(documents)
# print(len(docs))

embeddings = OpenAIEmbeddings(openai_api_key=key)

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=docs,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # by default search_type="similarity_score_threshold"

# docs = retriever.invoke("how much years of experience Rohit has.")
# print(docs[0].page_content)

turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                       chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)

#####
warnings.filterwarnings("ignore")

dictConfig(log_config)
app = FastAPI(title="Sample FastAPI Application",
    description="Sample FastAPI Application with Swagger and Sqlalchemy",
    version="1.0.0",debug=True)

models.Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()
nlp = spacy.load('en_core_web_sm')
skill_list = pd.read_csv("skills.csv").columns.values
custom_nlp = spacy.load('model-best-v2', disable=["parser"])

logger = logging.getLogger("uvicorn.error")


#post api to
@app.post("/upload-cv")
def upload_cv(file: UploadFile):
    logger.info(f"processing the resume {file.filename}")
    text=pdf_2_txt(file)
    lang,prob=detect_language_with_langdetect(text[:100])
    logger.info(f"Detected language code {lang} with confidence {prob}")
    # text=remove_non_alphanumeric_lines(text)
    # print(text)
    start = timer()
    if lang=='en':
        logger.info(f"Parsing the resume....")
        resume_obj=EnglishResume(nlp, custom_nlp, skill_list, text)
        data=resume_obj.parse()

        data['metadata']={
            'job_pk':22,
            'remark':'success',
            'status':'200 OK',
            'language-code':lang,
            'language-confidence':prob
        }
        end = timer()
        logger.info(f"parsed resume in  {timedelta(seconds=end - start)} seconds")

        parser_record = Parser_Record(remark='success', status="succ", resume_pk=22, candidate_pk=22,
                                      language_code='en', language_confidence=prob, api_type='STANDARD_PARSER', response=data)
        session.add(parser_record)
        session.commit()
        session.close()
        return data
    else:
        return {"status": "Currently we only support English resume"}


@app.post("/upload-cv-sota")
def upload_cv(file: UploadFile):
    logger.info(f"processing the resume {file.filename}")
    text=pdf_2_txt(file)
    lang,prob=detect_language_with_langdetect(text[:100])
    logger.info(f"Detected language code {lang} with confidence {prob}")
    # text=remove_non_alphanumeric_lines(text)
    # print(text)
    start = timer()
    if lang=='en':
        logger.info(f"Parsing the resume....")
        data=llm_parse_return(text,key)

        end = timer()
        logger.info(f"parsed resume in  {timedelta(seconds=end - start)} seconds")
        # session.close()
        return data
    else:
        return {"status": "Currently we only support English resume"}

@app.get("/query-cv-rag/")
def query_cv(request: Request):
    params = request.query_params
    print(params.get("question"))
    print("------------------")
    query=params.get("question")
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    return {"status": "Done"}

if __name__ == "__main__":

    ##RAG init

    uvicorn.run("main:app", port=9050, reload=False,workers=1)