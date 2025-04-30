# Import necessary modules for document loading, vector storage, retrieval, and query processing
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import (SimpleDirectoryReader, StorageContext, Settings,
                              VectorStoreIndex, get_response_synthesizer)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore

# Set Ollama as the embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

# 1. Load Data from the 'data' directory
# Reads all documents in the 'data' folder and prepares them for indexing
documents = SimpleDirectoryReader("data").load_data()

# 2. Index Data using ChromaDB
# Set up a persistent ChromaDB client and collection for storing document vectors (embeddings)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("demo_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# Create a storage context that wraps the vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Create the index
# Build a VectorStoreIndex from the loaded documents and storage context
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context)

# 4. Create a query engine for retrieval-augmented generation (RAG)
# Set up a retriever to fetch the top 3 most similar documents for a query
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
# Set up a response synthesizer to combine retrieved information into a final answer
response_synthesizer = get_response_synthesizer()
# Create a query engine that uses the retriever, synthesizer, and a similarity postprocessor
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_threshold=0.5)])

# 5. Run a sample query and print the response
# The query engine retrieves relevant documents and synthesizes an answer
response = query_engine.query("What is Cloud Club?")
print(response)
