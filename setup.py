from setuptools import setup, find_packages

setup(
    name="chatsapi",
    version="0.1.0",
    description="The World's Fastest AI Agent Framework. Based on SBERT & SpaCy Transforms.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Sathnindu Kottage",
    author_email="hello@bysatha.com",
    url="https://github.com/chatsapi/chatsapi",  # Replace with your GitHub repo
    packages=find_packages(),
    install_requires=[
        "sentence_transformers",
        "spacy",
        "hnswlib",
        "rank_bm25",
        "llama-index",
        "llama-index-llms-openai",
        "llama-index-llms-gemini",
        "llama-index-llms-llama-api",
        "llama-index-llms-ollama"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
