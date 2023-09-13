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

# CONVERSATIONAL CHAIN
db_directory = 'docs/chroma'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=db_directory, embedding_function=embedding)

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
  retriever = vectordb.as_retriever()
)

def chat(prompt):
  response = chain({"question": prompt,
                    "chat_history": st.session_state['history']})
  st.session_state['history'].append((prompt, response["answer"]))
  return response["answer"]