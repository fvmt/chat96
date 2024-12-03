import os
from langchain_text_splitters import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()


from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore():
    # sentence - transformers / multi - qa - MiniLM - L6 - cos - v1
    # HuggingFaceEmbeddings have to be downloaded locally because there are a lot of problems to use it from online
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv('HUGGINGFACE_EMBEDDING_PATH')) #, huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    client = QdrantClient(os.getenv('QDRANT_URL'))

    # Create a new collection if it is not already existing
    try:
        client.create_collection(
            os.getenv('COLLECTION_NAME'),
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    except UnexpectedResponse:
        pass

    # This content_payload_key="text" parameter took quite a while for me to figure out
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.getenv('COLLECTION_NAME'),
        embedding=embeddings,
        content_payload_key="text"
    )
    return vector_store

def get_conversation_chain(vectorstore):
    # Local LLM
    llm = OllamaLLM(model='Llama3.2')


    # Prompt settings where stolen from on of the examples on the web
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain