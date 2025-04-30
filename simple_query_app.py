# Import core LlamaIndex classes for document loading, indexing, and settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# Import Ollama embedding and LLM classes for local model inference
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# Import spinner for user feedback during long operations
from halo import Halo

# Import custom color class for colored console output
from color import Color


def console_print(message: str, color_name: str = Color.WHITE) -> None:
    # Print a message to the console in the specified color
    print(color_name + message + Color.reset())


def load_document(folder: str = ""):
    # Load and index documents from the specified folder
    if not folder:
        raise ValueError("The 'folder' parameter cannot be null or empty.")

    # Start the spinner to indicate processing
    spinner.start()

    # Set up the embedding and LLM using local Ollama models
    # Embedding model: nomic-embed-text (via Ollama)
    # LLM: llama3.2 (via Ollama)
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest")
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

    # Load all text files from the folder into Document objects
    documents = SimpleDirectoryReader(folder).load_data()
    # Create a vector index from the loaded documents
    index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)
    # Create a query engine for answering questions
    base_query_engine = index.as_query_engine()
    # Stop the spinner after processing
    spinner.stop()
    # Notify the user that the database is ready
    console_print("The database is up to date.", Color.LIGHT_GRAY)
    return base_query_engine


if __name__ == "__main__":
    # Main entry point for the script
    # Create a spinner for visual feedback
    spinner = Halo(text='Loading', spinner='dots')

    try:
        # Load and index documents from the 'data' folder
        query_engine = load_document("data")

        while True:
            # Prompt the user for a question
            user_question = input(
                Color.LIGHT_BLUE +
                "Please enter your question (or type 'quit' to exit):"
                + Color.reset() + "\n")

            # Exit loop if the user wants to quit
            if user_question.lower() in ['quit', 'qq', 'q', 'bye', 'exit']:
                break

            # Start spinner while processing the query
            spinner.start()

            # Query the index with the user's question
            response = query_engine.query(user_question)

            # Stop spinner after getting the response
            spinner.stop()

            # Print the response in cyan
            console_print(str(response) + "\n", Color.CYAN)

        # Thank the user after exiting the loop
        console_print("Thank you for using the query engine!\n",
                      Color.LIGHT_GRAY)

    except (ValueError, TypeError) as e:
        # Handle errors and notify the user
        spinner.fail()
        console_print(f"Task failed: {e}", Color.RED)
