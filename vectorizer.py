import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except:
    from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings


def retrieve():
    loader = PyPDFLoader("./pdfDatabase/Academic-Catalog.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    itech_db = "itech_vectorStore"
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    if os.path.exists(itech_db): #check whether the vector database exists
        db = Chroma(persist_directory=itech_db,
                            embedding_function=embedding_function)
    else: # create a new database with a pseudo entry
        db = Chroma.from_documents(
            texts,
            embedding=embedding_function,
            persist_directory=itech_db
        )
    retriever = db.as_retriever()
    return retriever

