import os
import openai
import sys
import streamlit as st
import rdflib
import rdflib.plugins.stores.sparqlstore as store
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, DirectoryLoader, CSVLoader
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from chromadb.api.types import Embeddings
from fastapi import FastAPI


os.environ["OPENAI_API_KEY"] = "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"
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# LOAD DATA

pdfLoader = PyPDFDirectoryLoader("./data/PDF/")
csvLoader = DirectoryLoader('./data/CSV/', glob='**/*.csv', loader_cls=CSVLoader, loader_kwargs={'encoding': 'latin-1'})

loadedPdf = pdfLoader.load()
loadedCsv = csvLoader.load()

# SPLIT DATA

pdfSplitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 30, separators = "\n")
csvSplitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 0)

pdfSplits = pdfSplitter.split_documents(loadedPdf)
csvSplits = csvSplitter.split_documents(loadedCsv)

# VECTOR STORAGE

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

db_directory = './docs/chroma'

vectordb = Chroma(embedding_function=embeddings, persist_directory=db_directory)

batch_size = 5000

print(len(pdfSplits))
print(len(csvSplits))

for i in range(0, 10000, batch_size):
    batch = pdfSplits[i:i+batch_size]  
    vectordb.add_documents(documents=batch)

vectordb.add_documents(pdfSplits[15000:])

for i in range(0, 40000, batch_size):
    batch = csvSplits[i:i+batch_size]
    vectordb.add_documents(documents=batch)

vectordb.add_documents(csvSplits[45000:])

vectordb.persist()
