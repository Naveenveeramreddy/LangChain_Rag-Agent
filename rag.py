from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader


# from langchain_ollama import ChatOllama
# file_path = "./test3.pdf"
# loader = PyPDFLoader(file_path)
# docs = loader.load()

import bs4
from langchain_community.document_loaders import WebBaseLoader


# bs4_strainer = bs4.SoupStrainer(
#     class_=("post-title", "post-header", "post-content"))
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs={"parse_only": bs4_strainer},
# )
# docs = loader.load()

file_path = "Naveen_resume.pdf","Pan_card.pdf","Aadhar_Card.pdf" 
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(f"Split blog post into {len(all_splits)} sub-documents.")

embeddings = DeterministicFakeEmbedding(size=4096)

vector_store = InMemoryVectorStore(embeddings)

document_ids = vector_store.add_documents(documents=all_splits)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]

# model = ChatOpenAI(
#     model="gemma-3-4b",
#     temperature=0.1,
#     max_tokens=1000,
#     timeout=30,
#     base_url="http://localhost:1234/v1",
#     api_key="not-needed",
# )

model = ChatOllama(
    model="smollm2:135m",
    temperature=0,
    # other params...
)

prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

agent = create_agent(
    model,
    # tools,
    # system_prompt=prompt
)

query = (
    # "What is the standard method for Task Decomposition?\n\n"
    # "Once you get the answer, look up common extensions of that method."
    "can you explain about Google AI-ML Student Intern - Virtua?  "
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
