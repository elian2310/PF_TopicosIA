import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from chromadb.api.types import Embeddings

os.environ["OPENAI_API_KEY"] = "openIAKEY"
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# LOAD DATA

csvs = 10
pdfs = 10

csvLoaders = []
pdfLoaders = []

for i in range(csvs):
  csvLoaders.append(CSVLoader(file_path=f"csv{i}.csv"))

for i in range(pdfs):
  pdfLoaders.append(PyPDFLoader(f"pdf{i}.pdf"))

loadedCsv = []
loadedPdf = []

for i in range(len(csvLoaders)):
  loadedCsv.append(csvLoaders[i].load())

for i in range(len(pdfLoaders)):
  loadedPdf.append(pdfLoaders[i].load())

# SPLIT DATA

chunkSize = 300
overlap = 30

splitter = RecursiveCharacterTextSplitter(chunk_size = chunkSize, chunk_overlap = overlap)

pdfSplits = splitter.split_documents(loadedPdf)

# VECTOR STORAGE

embeddings = OpenAIEmbeddings()

sentence1 = "a"
embedding1 = embeddings.embed_query(sentence1)

db_directory = 'docs/chroma'
vectordb = Chroma.from_documents(documents=pdfSplits, embedding=embeddings, persist_directory=db_directory)
vectordb.persist()
