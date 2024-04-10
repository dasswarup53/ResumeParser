from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from timeit import default_timer as timer
from datetime import timedelta
import fitz
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

if __name__ == "__main__":

    uvicorn.run("main:app", port=9050, reload=True)