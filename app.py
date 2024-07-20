import streamlit as st
import subprocess
import sys

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "langchain",
    "langchain_community",
    "sentence-transformers",
    "python-dotenv",
    "pinecone-client",
    "huggingface-hub",
]

# Install required packages if not already installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

login(token='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')
# Load environment variables
load_dotenv('load.env')

# Initialize Pinecone and other resources only once
if 'initialized' not in st.session_state:
    # Initialize Pinecone
  # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = 'new1'
    index = pinecone.Index(index_name=index_name, api_key='bcbe1f88-c72b-4642-b743-2195654f3202', host='https://new1-tib9ci6.svc.aped-4627-b74a.pinecone.io')  #

    # Initialize the retriever
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model, text_key='text')
    retriever = vector_store.as_retriever()

    # Define the prompt template
    template = """
    You are a customer service agent. Customers will ask you questions about their inquiries.
    Use the following piece of context to answer the question.
    If you don't know the answer, just say you don't know.
    Keep the answer within 2 sentences and concise.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize the LLM
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.8, "top_k": 50},
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    # Define the RAG chain
    ragchain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Store everything in session state
    st.session_state.initialized = True
    st.session_state.ragchain = ragchain

st.title("Customer Service Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your message"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the user input and get the chatbot response
    try:
        result = st.session_state.ragchain.invoke(prompt)
        # Display chatbot response in chat message container
        with st.chat_message("bot"):
            # Only show the answer from the result
            answer = result.split("Answer:")[1].strip()
            st.markdown(answer)
        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "bot", "content": answer})
    except Exception as e:
        st.error(f"Error: {str(e)}")
