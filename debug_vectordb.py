#!/usr/bin/env python3

"""
Debug script to check what's inside the Chroma vector database
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from src.utils.load_config import LoadConfig

APPCFG = LoadConfig()

print(f"Checking vector database at: {APPCFG.persist_directory}")
print(f"Directory exists: {os.path.exists(APPCFG.persist_directory)}")

try:
    # Try to load the vector database
    vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                      embedding_function=APPCFG.embedding_model)

    print(f"Successfully loaded vectordb")
    print(f"Collection count: {vectordb._collection.count()}")

    # Try to get all documents
    try:
        all_docs = vectordb.get()
        print(f"Total documents in database: {len(all_docs['ids']) if all_docs['ids'] else 0}")

        if all_docs['ids']:
            print("First few document IDs:")
            for i, doc_id in enumerate(all_docs['ids'][:3]):
                print(f"  {i + 1}: {doc_id}")

        # Try a simple search
        print("\nTrying a simple search...")
        search_results = vectordb.similarity_search("test", k=1)
        print(f"Search returned {len(search_results)} results")

    except Exception as e:
        print(f"Error getting documents: {e}")

except Exception as e:
    print(f"Error loading vectordb: {e}")
    print(f"Error type: {type(e).__name__}")