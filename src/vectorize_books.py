import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from torch import embedding

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
data_dir = f"{parent_dir}/data"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"


embedding = HuggingFaceEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

def vectorize_book_and_store_to_db(class_subject_name, vector_db_name):
    book_dir = f"{data_dir}/{class_subject_name}"
    vector_db_path = f"{vector_db_dir}/{vector_db_name}"
    loader = DirectoryLoader(path=book_dir, glob="*./pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=vector_db_path)
    print(f"{class_subject_name} saved to vector db: {vector_db_name}")


def vectorize_chapters(class_subject_name):
    pass