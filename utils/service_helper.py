from langdetect import detect_langs
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz

def detect_language_with_langdetect(line):

    try:
        langs = detect_langs(line)
        for item in langs:
            # The first one returned is usually the one that has the highest probability
            return item.lang, item.prob
    except: return "err", 0.0

def pdf_2_txt(file):
    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    text = ""
    doc = fitz.open(file.filename)  # open a document
    for page in doc:  # iterate the document pages
        text += page.get_text()  # get plain text encoded as UTF-8
    # print(text)
    return text

### RAGG Helper Functions
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