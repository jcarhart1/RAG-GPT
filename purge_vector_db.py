import shutil
import os

# Delete the entire vectordb directory
if os.path.exists('data/vectordb'):
    shutil.rmtree('data/vectordb')
    print("Vector database deleted successfully")