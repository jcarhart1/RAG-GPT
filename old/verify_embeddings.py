from langchain_community.vectorstores import Chroma
from utils.load_config import LoadConfig

# Initialize APPCFG
APPCFG = LoadConfig()


def verify_embeddings():
    # Ensure you are using the processed directory (preprocessed documents)
    vectordb = Chroma(persist_directory=APPCFG.persist_directory, embedding_function=APPCFG.embedding_model)

    # Perform a dummy search to retrieve some documents
    docs = vectordb.similarity_search(query="dummy query", k=3)

    # Log the document content and embedding
    for i, doc in enumerate(docs):
        print(f"Document {i + 1} content:\n", doc.page_content)
        print(f"Document {i + 1} metadata:\n", doc.metadata)
