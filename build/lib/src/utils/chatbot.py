import gradio as gr
import time
from openai import OpenAI
import os
from langchain_chroma import Chroma
from typing import List, Tuple
import re
import ast
import html
from src.utils.load_config import LoadConfig
from dotenv import load_dotenv

# os.makedirs('/Users/jcarhart/Desktop/code_personal_use/LLM-Zero-to-Hundred/RAG-GPT/data/vectordb/uploaded/chroma', exist_ok=True)

load_dotenv(".env")  # This will load the .env file

client = OpenAI()
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

APPCFG = LoadConfig()
URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"
hyperlink = f"[RAG-GPT user guideline]({URL})"


class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    cleaning references from retrieved documents.
    """

    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:
        """
        Generate a response to a user query using document retrieval and language model completion.

        Parameters:
            chatbot (List): List representing the chatbot's conversation history.
            message (str): The user's query.
            data_type (str): Type of data used for document retrieval ("Preprocessed doc" or "Upload doc: Process for RAG").
            temperature (float): Temperature parameter for language model completion.

        Returns:
            Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
        """

        # Handle Preprocessed doc case
        if data_type == "Preprocessed doc":
            # Check if the vector database exists
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=APPCFG.embedding_model)
                print(f"Number of documents in the vector store: {vectordb._collection.count()}")
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module."))
                return "", chatbot, None

        # Handle Upload doc: Process for RAG case
        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_persist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append(
                    (message, "No file was uploaded. Please first upload your files using the 'upload' button."))
                return "", chatbot, None

        # Perform similarity search on the vector database
        docs = vectordb.similarity_search(message, k=APPCFG.k)
        print("Retrieved documents: ", docs)

        # Prepare question and chat history
        question = "# User new question:\n" + message
        retrieved_content = ChatBot.clean_references(docs)

        # Build chat history and the full prompt
        chat_history = f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        prompt = f"{chat_history}{retrieved_content}{question}"

        print("========================")
        print("Constructed prompt before sending to OpenAI:")
        print(prompt)
        print("========================")

        # Generate the response from OpenAI API
        response = client.chat.completions.create(
            model=APPCFG.llm_engine,
            messages=[
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )

        # Append the new message and OpenAI's response to the chat history
        chatbot.append((message, response.choices[0].message.content))

        # Return the required 3 values: clear input, updated chatbot history, and retrieved content
        return "", chatbot, retrieved_content

    @staticmethod
    def clean_references(documents: List) -> str:
        """
        Clean and format references from retrieved documents.

        Parameters:
            documents (List): List of retrieved documents.

        Returns:
            str: A string containing cleaned and formatted references.
        """
        server_url = "http://localhost:8000"
        markdown_documents = ""
        counter = 1

        # Iterate over the Document objects without converting them to strings
        for doc in documents:
            print(f"Document {counter} content before cleaning:\n", doc.page_content)  # Access the document content

            # Directly access content and metadata
            content = doc.page_content
            metadata = doc.metadata

            # Clean up the content as needed (e.g., decode escape sequences)
            content = bytes(content, "utf-8").decode("unicode_escape")
            content = re.sub(r'\\n', '\n', content)
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()

            # Decode HTML entities
            content = html.unescape(content)

            # Prepare the PDF URL using metadata
            pdf_url = f"{server_url}/{os.path.basename(metadata['source'])}"

            # Format content and metadata into markdown string
            markdown_documents += (
                    f"# Retrieved content {counter}:\n"
                    + content
                    + "\n\n"
                    + f"Source: {os.path.basename(metadata['source'])}"
                    + " | "
                    + f"Page number: {str(metadata['page'])}"
                    + " | "
                    + f"[View PDF]({pdf_url})\n\n"
            )
            counter += 1

        return markdown_documents
