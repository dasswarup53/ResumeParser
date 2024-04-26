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

def llm_parse_return(txt,key):


    # docs = split_docs(documents)
    # print(len(docs))
    embeddings = OpenAIEmbeddings(openai_api_key=key)



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


    # memory_llm_conversation = ConversationChain(
    #     llm=turbo_llm_memory,
    #     memory = memory,
    #     verbose=True
    # )
    # messages = prompt_template.format_messages(text=txt)
    # chat = ChatOpenAI(temperature=0.0, model=turbo_llm_memory)
    # response = memory_llm_conversation(messages)
    # print("++++")
    # print(response['response'])
    # print("++++")

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