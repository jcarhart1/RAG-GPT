"""
This module is not part of the main RAG-GPT pipeline and it is only for showing how we can perform RAG using openai and vectordb in the terminal.

To execute the code, after preparing the python environment and the vector database, in the terminal execute:

python src\terminal_q_and_a.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
import yaml
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_chroma import Chroma
from typing import List, Tuple
from src.utils.load_config import LoadConfig
from dotenv import load_dotenv

load_dotenv(".env")  # This will load the .env file

# For loading openai credentials
client = OpenAI()
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

APPCFG = LoadConfig()


with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Load the embedding function
embedding = OpenAIEmbeddings()
# Load the vector database
vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                  embedding_function=embedding)

print("Number of vectors in vectordb:", vectordb._collection.count())
print("Here are the vectors in vectordb:", vectordb._collection)

# Prepare the RAG with openai in terminal
while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() =='q':
        break
    question = "# user new question:\n" + question

    docs = vectordb.similarity_search(question, k=APPCFG.k)
    # for x, score in docs:
    #     print(f"* [SIM={score:3f}] {x.page_content} [{x.metadata}]")
    retrieved_docs_page_content: List[Tuple] = [str(x.page_content)+"\n\n" for x in docs]
    retrieved_docs_str = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)

    prompt = retrieved_docs_str + "\n\n" + question

    response = client.chat.completions.create(
        model=APPCFG.llm_engine,
        messages=[
            {"role": "system", "content": APPCFG.llm_system_role},
            {"role": "user", "content": prompt}
        ]
    )
    print(response)
    # print(response['choices'][0]['message']['content'])
    print(response.choices[0].message.content)

