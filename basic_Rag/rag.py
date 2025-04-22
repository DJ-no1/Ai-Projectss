from pathlib import Path
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
# print("Docs: ", docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)
# print("Split texts: ", text_splitter)
# Split texts:  <langchain_text_splitters.character.RecursiveCharacterTextSplitter object at 0x000002707F075730>

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=dotenv.get_key(dotenv.find_dotenv(), "GOOGLE_API_KEY"),
    model="models/text-embedding-004"
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder,
# )
# vector_store.add_documents(documents=split_docs)
# print("Injection Done")

retrival = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder,
)

userQuery = input("write your question?")

search_results = retrival.similarity_search(
    query=userQuery
)
#print("Relevant chunks: ", search_results)

# system prompt
system_prompt = f"""
    You are an helpful AI assistant that is used to answer questions.

    context: {search_results}
"""
client = OpenAI(
    api_key=dotenv.get_key(dotenv.find_dotenv(), "GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# user prompt
response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": userQuery}
        ]
    )
                
            
print(response.choices[0].message.content)