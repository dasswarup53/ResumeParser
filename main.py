from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from timeit import default_timer as timer
from datetime import timedelta
import fitz
import os
import spacy
import spacy_transformers
import pandas as pd
import warnings
from db import engine
import models as models
from models import  Parser_Record
import logging
import re
from sqlalchemy.orm import sessionmaker
import uvicorn
from typing import List,Optional
from fastapi.encoders import jsonable_encoder
import io
from fastapi import FastAPI, UploadFile, File
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
import logging
import traceback
from Parser.EnglishResumeParser import EnglishResume
from logging.config import dictConfig
import logging
from log_config import log_config
from langdetect import detect_langs
from Parser.GenAiEnglishParser import parse_return
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

#####
#adding RAG
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

documents = load_docs('cv-directory')

key = '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'

# docs = split_docs(documents)
# print(len(docs))

os.environ["OPENAI_API_KEY"] = key

embeddings = OpenAIEmbeddings(openai_api_key=key)

persist_directory = 'db'

embedding = OpenAIEmbeddings()

# vectordb = Chroma.from_documents(documents=docs,
#                                  embedding=embedding,
#                                  persist_directory=persist_directory)

# vectordb.persist()
# vectordb = None

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




def detect_language_with_langdetect(line):

    try:
        langs = detect_langs(line)
        for item in langs:
            # The first one returned is usually the one that has the highest probability
            return item.lang, item.prob
    except: return "err", 0.0

#post api to
@app.post("/upload-cv")
def upload_cv(file: UploadFile):
    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    logger.info(f"processing the resume {file.filename}")
    text=""
    doc = fitz.open(file.filename)  # open a document
    for page in doc:  # iterate the document pages
        text += page.get_text()  # get plain text encoded as UTF-8
    print(text)
    lang,prob=detect_language_with_langdetect(text[:100])
    logger.info(f"Detected language code {lang} with confidence {prob}")
    # text=remove_non_alphanumeric_lines(text)
    # print(text)
    start = timer()
    if lang=='en':
        logger.info(f"Parsing the resume....")
        resume_obj=EnglishResume(nlp, custom_nlp, skill_list, text)
        data=resume_obj.parse()

        parser_record=Parser_Record(remark='success',status="succ",resume_pk=22,candidate_pk=22,language_code='en',language_confidence=prob)
        session.add(parser_record)
        session.commit()
        data['metadata']={
            'job_pk':parser_record.job_pk,
            'remark':parser_record.remark,
            'status':parser_record.status,
            'language-code':parser_record.language_code,
            'language-confidence':parser_record.language_confidence
        }
        end = timer()
        logger.info(f"parsed resume in  {timedelta(seconds=end - start)} seconds")
        session.close()
        return data
    else:
        return {"status": "Currently we only support English resume"}


@app.post("/upload-cv-sota")
def upload_cv(file: UploadFile):
    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    logger.info(f"processing the resume {file.filename}")
    text=""
    doc = fitz.open(file.filename)  # open a document
    for page in doc:  # iterate the document pages
        text += page.get_text()  # get plain text encoded as UTF-8
    print(text)
    lang,prob=detect_language_with_langdetect(text[:100])
    logger.info(f"Detected language code {lang} with confidence {prob}")
    # text=remove_non_alphanumeric_lines(text)
    # print(text)
    start = timer()
    if lang=='en':
        logger.info(f"Parsing the resume....")
        data=parse_return(text)

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
    query=params.get("question")
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    return {"status": "Done"}

if __name__ == "__main__":

    ##RAG init

    uvicorn.run("main:app", port=9050, reload=False,workers=1)