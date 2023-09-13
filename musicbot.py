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


os.environ["OPENAI_API_KEY"] = "openIAKEY"
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# LOAD DATA

csvs = 10
pdfs = 10

csvLoaders = []
pdfLoaders = []

for i in range(csvs):
  csvLoaders.append(CSVLoader(file_path=f"./data/CSV/csv{i}.csv", encoding="utf-8", csv_args={'delimiter': ','}))

for i in range(pdfs):
  pdfLoaders.append(PyPDFLoader(f"./data/PDF/pdf{i}.pdf"))

loadedCsv = []
loadedPdf = []

for i in range(len(csvLoaders)):
  loadedCsv.append(csvLoaders[i].load())

for i in range(len(pdfLoaders)):
  loadedPdf.append(pdfLoaders[i].load())

# SPLIT DATA

splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 30, separators = "\n")

pdfSplits = splitter.split_documents(loadedPdf)

# VECTOR STORAGE

embeddings = OpenAIEmbeddings()

db_directory = 'docs/chroma'

vectordb = Chroma(embeddings=embeddings, persist_directory=db_directory)
vectordb.add_documents(documents=pdfSplits)
vectordb.add_documents(documents=loadedCsv)

vectordb.persist()
