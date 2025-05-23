import shutil
import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
# import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pypdf.errors import PdfStreamError  # Import the error to handle corrupted PDFs

os.makedirs('/Users/jcarhart/Desktop/code-personal/RAG-GPT/data/vectordb/uploaded/chroma', exist_ok=True)
load_dotenv()  # This will load the .env file


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using OpenAI embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with OpenAI embeddings. It provides methods to prepare and save the VectorDB.

    Parameters:
        data_directory (str or List[str]): The directory or list of directories containing the documents.
        persist_directory (str): The directory to save the VectorDB.
        embedding_model_engine (str): The engine for OpenAI embeddings.
        chunk_size (int): The size of the chunks for document processing.
        chunk_overlap (int): The overlap between chunks.
        metadata_file (str, optional): Path to the metadata JSON file. Defaults to None.
    """

    def __init__(
            self,
            data_directory: str,
            persist_directory: str,
            embedding_model_engine: str,
            chunk_size: int,
            chunk_overlap: int,
            metadata_file: str = None
    ) -> None:
        """
        Initialize the PrepareVectorDB instance.

        Parameters:
            data_directory (str or List[str]): The directory or list of directories containing the documents.
            persist_directory (str): The directory to save the VectorDB.
            embedding_model_engine (str): The engine for OpenAI embeddings.
            chunk_size (int): The size of the chunks for document processing.
            chunk_overlap (int): The overlap between chunks.
            metadata_file (str, optional): Path to the metadata JSON file. Defaults to None.
        """

        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        """Other options: CharacterTextSplitter, TokenTextSplitter, etc."""
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()
        self.metadata_file = metadata_file
        self.metadata_map = self.__load_metadata() if metadata_file else {}

    def __load_metadata(self):
        """
        Load metadata from JSON file and create a mapping by filename.

        Returns:
            dict: A mapping of PDF filenames to metadata.
        """
        try:
            with open(self.metadata_file, 'r') as f:
                metadata_list = json.load(f)

            # Create a mapping of PDF filenames to metadata
            return {item["pdf"]: item for item in metadata_list}
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            return {}

    def __load_all_documents(self) -> List:
        """
        Load all documents from the specified directory or directories.

        Returns:
            List: A list of loaded documents.
        """
        doc_counter = 0
        docs = []
        if isinstance(self.data_directory, list):
            print("Loading the uploaded documents...")
            for doc_dir in self.data_directory:
                try:
                    loaded_docs = PyPDFLoader(doc_dir).load()

                    # Add metadata to each document
                    filename = os.path.basename(doc_dir)
                    if filename in self.metadata_map:
                        for doc in loaded_docs:
                            doc.metadata.update(self.metadata_map[filename])

                    docs.extend(loaded_docs)
                    doc_counter += 1
                except PdfStreamError as e:
                    print(f"Error loading PDF from {doc_dir}: {e}")
                except Exception as e:
                    print(f"Unexpected error loading PDF from {doc_dir}: {e}")

            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")
        else:
            print("Loading documents manually...")
            document_list = os.listdir(self.data_directory)
            for doc_name in document_list:
                if not doc_name.endswith('.pdf'):
                    continue

                doc_path = os.path.join(self.data_directory, doc_name)
                try:
                    loaded_docs = PyPDFLoader(doc_path).load()

                    # Add metadata to each document
                    if doc_name in self.metadata_map:
                        for doc in loaded_docs:
                            doc.metadata.update(self.metadata_map[doc_name])

                    docs.extend(loaded_docs)
                    doc_counter += 1
                except PdfStreamError as e:
                    print(f"Error loading PDF {doc_name}: {e}")
                except Exception as e:
                    print(f"Unexpected error loading PDF {doc_name}: {e}")

            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")

        return docs

    def __chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded documents using the specified text splitter.

        Parameters:
            docs (List): The list of loaded documents.

        Returns:
            List: A list of chunked documents.

        """
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        """
        Load, chunk, and create a VectorDB with OpenAI embeddings, and save it.

        Returns:
            Chroma: The created VectorDB.
        """

        # Remove the existing vector database if it exists
        if os.path.exists(self.persist_directory):
            print(f"Removing existing vector database at {self.persist_directory}...")
            shutil.rmtree(self.persist_directory)

        docs = self.__load_all_documents()

        print(f"Documents to chunk: {len(docs)}")
        for doc in docs:
            print(f"Document content: {doc.page_content[:200]}")  # Print first 200 characters of each doc
            if doc.metadata and self.metadata_map:
                print(f"Document metadata: {doc.metadata}")  # Print metadata if available

        chunked_documents = self.__chunk_documents(docs)
        print("Preparing vectordb...")

        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        print("VectorDB is created and saved.")
        print(f"Number of vectors in vectordb: {vectordb._collection.count()}")
        print("Number of vectors in vectordb:",
              vectordb._collection.count(), "\n\n")
        return vectordb