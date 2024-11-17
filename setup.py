from setuptools import setup, find_packages

setup(
    name="rag_app",
    version="0.1.0",  # Increment versions as needed
    packages=find_packages(),
    install_requires=[
        # List dependencies specific to your RAG app, e.g., "transformers", "torch"
    ],
    author="Joe Carhart",
    author_email="carhart77@gmail.com",
    description="A RAG application for document processing and summarization.",
    url="https://github.com/jcarhart1/RAG-GPT",  # Link to the GitHub repo
)
