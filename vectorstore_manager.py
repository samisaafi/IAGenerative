from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config.config import RAGChatbotConfig

class VectorStoreManager:
    """Chroma Vector Store Manager"""

    def __init__(self, config: RAGChatbotConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

    def create_vector_store(self, texts: List[str]) -> Chroma:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

        documents = splitter.create_documents(texts)

        vectordb = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=self.config.PERSIST_DIRECTORY
        )
        vectordb.persist()
        return vectordb

    def load_vector_store(self) -> Chroma:
        return Chroma(
            persist_directory=self.config.PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )

    def delete_vector_store(self) -> None:
        import shutil
        shutil.rmtree(self.config.PERSIST_DIRECTORY)