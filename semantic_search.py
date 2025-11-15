import getpass
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


file_path = "Naveen_resume.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(docs)
# print(f"{docs[1].page_content[:]}\n")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# print((all_splits[0]))  # Print the first chunk's content

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
# os.environ["OPENAI_API_KEY"] = ""

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

embeddings = DeterministicFakeEmbedding(size=4096)
# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# print(all_splits[4].page_content)
# vector_1 = embeddings.embed_query(
#     "Naveen Veeramreddy is a software engineer with experience in Python and machine learning.")

# print(len(vector_1))
# print(len(all_splits[0].page_content))


# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     # Where to save data locally, remove if not necessary
#     persist_directory="./chroma_langchain_db",
# )


vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "Naveen Veeramreddy is a software engineer with experience in Python and machine learning.", k=1
# )

# print(results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score(
    "A backend application written in Python that simulates a car showroom experience, allowing users to select a car brand, model, and calculates the final price including taxes and insurance.")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)
