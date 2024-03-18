import os

import streamlit as st
from bs4 import BeautifulSoup as Soup
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings, JinaEmbeddings
from recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

os.environ['GOOGLE_API_KEY'] = st.secrets['PALM_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = OllamaEmbeddings()  # model='mistral'
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")


# embeddings = JinaEmbeddings(
#     jina_api_key='jina_78e0899aa68f400d9e8570e8dc5e823779amu6-0kF3iw2KzLc0cJ8GwKD0W',
#     model_name="jina-embeddings-v2-base-en"
# )


def prepare_docs():
    """
    Loads, splits, and embeds documents from a given URL.

    :return: A tuple containing the loaded documents and the embeddings.
    :rtype: tuple
    """
    url = "https://www.shepherd.edu/student-handbook/"

    # I modified the Recursive URL Loader to include the root url in the extraction
    loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = loader.load_and_split(text_splitter)
    return docs


def get_vector_db():
    """
    Retrieve the vector database.

    This method attempts to open the file "faiss_store_openai.pkl" and return the contents.
    If the file is not found, it prepares the necessary documents and embeddings using the
    `prepare_docs` method. It then creates a FAISS vector store using the documents and embeddings.
    The vector store is persisted to disk in the "faiss_store_openai.pkl" file.

    :return: The vector database.
    """
    try:
        vectors = FAISS.load_local("faiss_store_openai", embeddings)
        # vectors = FAISS.load_local("faiss_store_ollama", embeddings)
        # vectors = FAISS.load_local("faiss_store_google", embeddings)
        # vectors = FAISS.load_local("faiss_store_jina", embeddings)
        return vectors
        # with open("faiss_store_openai.pkl", "rb") as f:
        #     vectors = pickle.load(f)
        #     return vectors
    except RuntimeError:
        docs = prepare_docs()
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_store_openai")
        # vector_store.save_local("faiss_store_ollama")
        # vector_store.save_local("faiss_store_google")
        # vector_store.save_local("faiss_store_jina")
        return vector_store
