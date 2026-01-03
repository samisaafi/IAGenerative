from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config.config import RAGChatbotConfig
from data.data_loader import DataLoader
from vectorstore.vectorstore_manager import VectorStoreManager
from llm.lmstudio_llm import LMStudioLLM


class RAGChatbot:
    """Main RAG Chatbot"""

    def __init__(self, config: RAGChatbotConfig):
        self.config = config
        self.vectorstore = None
        self.qa_chain = None

    def initialize_from_data(self, path: str, text_column: str | None = None):
        loader = DataLoader()

        if path.endswith(".csv"):
            texts = loader.load_csv(path, text_column)
        elif path.endswith(".json"):
            texts = loader.load_json(path, text_column)
        else:
            texts = loader.load_txt(path)

        vs = VectorStoreManager(self.config)
        self.vectorstore = vs.create_vector_store(texts)
        self._create_chain()

    def initialize_from_existing(self):
        vs = VectorStoreManager(self.config)
        self.vectorstore = vs.load_vector_store()
        self._create_chain()

    def _create_chain(self):
        llm = LMStudioLLM.create_llm(self.config)

        prompt = PromptTemplate(
            template="""
Use the context below to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RESULTS}
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def chat(self):
        print("\n💬 RAG Chatbot ready (type 'exit' to quit)\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit"):
                break

            result = self.qa_chain({"query": question})
            print("\nBot:", result["result"])
