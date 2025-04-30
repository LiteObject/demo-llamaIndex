# Demo llamaIndex

## What is Llama Index?

>LlamaIndex (formerly known as GPT Index) is an advanced data framework designed to facilitate the connection between large language models (LLMs) and external data sources. It serves as a central interface, enabling the integration of custom data with LLMs to create powerful, data-aware applications. 

## Project

This project demonstrates the use of LlamaIndex for document indexing and querying using AI models.

## Description

This application uses LlamaIndex to create a vector store index from documents in a `data` directory. It then uses this index to answer queries about the content of these documents. The project showcases the integration of Hugging Face embeddings and the Ollama LLM.

## Features

- Document indexing using LlamaIndex
- Vector store querying
- Integration with Hugging Face embeddings (BAAI/bge-base-en-v1.5)
- Use of Ollama LLM (llama3 model)

## Setup

1. Ensure you have Python installed on your system.
2. Install the required dependencies (you may want to use a virtual environment):
   ```
   pip install llama-index huggingface_hub ollama
   ```
3. Make sure you have the Ollama service running with the llama3 model available.

## Usage

1. Place your documents in the `data` directory. The example uses a file named `story.txt`.
2. Run the application:
   ```
   python app.py
   ```
3. The application will index the documents and then perform a sample query: "Who is Byte?"

## File Structure

- `app.py`: Main application file
- `data/`: Directory containing documents to be indexed
  - `story.txt`: Sample story about Byte, an AI robot
- `.pylintrc`: Pylint configuration file

## Configuration

The project uses the following configurations:

- Embedding model: BAAI/bge-base-en-v1.5
- LLM: Ollama (llama3 model)
- LLM request timeout: 360.0 seconds

---
## Q&A
**Q: What are the differences betwen `as_chat_engine` and `as_query_engine`?**

**A:** The `as_chat_engine` method is used to create a chat engine that can handle conversational interactions, while the `as_query_engine` method is used to create a query engine that can handle specific queries. The chat engine is designed for interactive conversations, while the query engine is more focused on retrieving information based on specific queries.

- **Use case** for `as_query_engine`: Fact-based Q&A, search, retrieval.
- **Use case** for `as_chat_engine`: Chatbots, assistants, or any scenario where context from previous turns matters.

---

## Links
- [Starter Example (local)](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)
- [Starter Example (OpenAI)](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)
- [GitHub/Llama-Index](https://github.com/run-llama/llama_index)
- [LlamaHub](https://llamahub.ai/)
