from langchain_community.llms import OpenAI
from config.config import RAGChatbotConfig

class LMStudioLLM:
    """LM Studio LLM wrapper"""

    @staticmethod
    def create_llm(config: RAGChatbotConfig):
        return OpenAI(
            base_url=config.LM_STUDIO_API_BASE,
            api_key=config.LM_STUDIO_API_KEY,
            temperature=0.7,
            max_tokens=500
        )
