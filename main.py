import streamlit
from langchain.indexes import VectorstoreIndexCreator
import os
from langchain import OpenAI
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import tiktoken
from langchain.chains import RetrievalQA
import time
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = ""

directory_path = 'docs/'

streamlit.title("Testing GPT")
prompt = streamlit.text_input("Type here...")

def load_docs(directory_path):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents

documents = load_docs(directory_path)

def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key="",
    environment=""
)
pinecone_index_name = ""
index = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)

def get_similar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
        time.sleep(60)
    else:
        similar_docs = index.similarity_search(query, k=k)

    return similar_docs

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.9)
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

query = "What is Gnan Vidhi?"
answer = get_answer(query)

if prompt:
    response = chain.run(topic=prompt)
    # response = llm(prompt)
    streamlit.write(response)
    # streamlit.write(docs[0].page_content)
