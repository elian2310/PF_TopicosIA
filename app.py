import os
import openai
import sys
import streamlit as st
import rdflib
import logging
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from chromadb.api.types import Embeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from SPARQLWrapper import SPARQLWrapper, JSON

app = FastAPI()

logging.basicConfig(filename='app.log', level=logging.INFO)

anzoURL = "http://localhost:80/sparql"
username = "admin"
password = "Passw0rd1"

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

os.environ["OPENAI_API_KEY"] = ""
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# CONVERSATIONAL CHAIN
db_directory = './docs/chroma'
embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vectordb = Chroma(persist_directory=db_directory, embedding_function=embedding)

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
  retriever = vectordb.as_retriever()
)

@app.get('/chat/{prompt}')
async def chat(prompt: str):
  response = chain({"question": prompt,
                    "chat_history": []})
  insertToKG(prompt)
  insertToKG(response["answer"])

  return (response["answer"])

def insertToKG(base, model="gpt-3.5-turbo"):
    prompt = f"Convert the following text delimited by triple parentheses into a SPARQL insert query without any prefix to a graph called music ((({base})))"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    gptResponse = response.choices[0].message["content"]
    logging.info(gptResponse)
    
    try:
       sparql = SPARQLWrapper(anzoURL)
       sparql.method = 'POST'
       sparql.setCredentials(username, password)
       sparql.setQuery(f"""{gptResponse}""".encode("utf-8"))
       sparql.setReturnFormat(JSON)
       
       results = sparql.query().convert()

       if results and "results" in results:
          logging.info("Successfully connected to anzograph")
       else:
          logging.info("Failed to connect to anzograph")
    except Exception as e:
       logging.info(f"An error ocurred: {e}")
