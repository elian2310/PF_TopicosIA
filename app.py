import os
import openai
import sys
import streamlit as st
import rdflib
import rdflib.plugins.stores.sparqlstore as store
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

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

os.environ["OPENAI_API_KEY"] = "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"
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# CONVERSATIONAL CHAIN
db_directory = './docs/chroma'
embedding = OpenAIEmbeddings(penai_api_key=os.environ['OPENAI_API_KEY'])
vectordb = Chroma(persist_directory=db_directory, embedding_function=embedding)

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
  retriever = vectordb.as_retriever()
)

@app.get('/chat/{prompt}')
async def chat(prompt: str):
  response = chain({"question": prompt,
                    "chat_history": []})
  return response["answer"]
