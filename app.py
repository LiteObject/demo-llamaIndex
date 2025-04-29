from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from halo import Halo

from color import Color


def console_print(message: str, color_name: str = Color.WHITE) -> None:
    print(color_name + message + Color.reset())


def load_document(folder: str = ""):
    if not folder:
        raise ValueError("The 'folder' parameter cannot be null or empty.")

    # Start the spinner
    spinner.start()

    # bge-base embedding model
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest")
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

    # Load all the text files from the specified "data" directory into a list of Document objects.
    documents = SimpleDirectoryReader(folder).load_data()
    index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)
    base_query_engine = index.as_query_engine()
    spinner.stop()
    console_print("The database is up to date.", Color.LIGHT_GRAY)
    return base_query_engine


if __name__ == "__main__":
    # Create a spinner
    spinner = Halo(text='Loading', spinner='dots')

    try:

        query_engine = load_document("data")

        while True:
            # Get the question from the user
            user_question = input(
                Color.LIGHT_BLUE + "Please enter your question (or type 'quit' to exit):" + Color.reset() + "\n")

            # Check if the user wants to quit
            if user_question.lower() in ['quit', 'qq', 'q', 'bye', 'exit']:
                break

            spinner.start()

            # Use the user's question in the query
            response = query_engine.query(user_question)

            spinner.stop()

            # Print the response
            console_print(str(response) + "\n", Color.CYAN)

        console_print("Thank you for using the query engine!\n",
                      Color.LIGHT_GRAY)

    except (ValueError, TypeError) as e:
        spinner.fail()
        console_print(f"Task failed: {e}", Color.RED)
