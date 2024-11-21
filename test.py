import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except:
    from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import SystemMessagePromptTemplate
import json
from groq import Groq
import dotenv

dotenv.load_dotenv()
groq_api_key= os.getenv("GROQ_API_KEY")
model = "llama3-8b-8192"
client = Groq(api_key=groq_api_key)

DEFAULT_SYSTEM_TEMPLATE_FOR_MEMORY = """
    You are a helpful assistant with the ability to retreive information from a vector store containing information specific to Indiana Tech
    from the users prompt, here are the relevant pieces of information from the vector store

    {relevant_pieces}

    

    """

itech_db = "itech_vectorStore"
embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
db = Chroma(persist_directory=itech_db,
                        embedding_function=embedding_function)
    


retreiver = db.as_retriever(search_kwargs = {"k":20})

def chat(prompt):
    docs = retreiver.invoke(prompt)
    docs = [doc.page_content for doc in docs]
    docs
    memory_message = SystemMessagePromptTemplate.from_template(
                DEFAULT_SYSTEM_TEMPLATE_FOR_MEMORY
    ).format(
        relevant_pieces = json.dumps(docs)
    )
    formatted_prompt = {"role": "user", "content": prompt}
    messages = [
        {"role": "system", "content": memory_message.content},
        formatted_prompt,
    ]
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model
    )
    
    response = chat_completion.choices[0].message.content
    return(response)

while True:
    i = input(">>> ")
    res = chat(i)
    print(res)
