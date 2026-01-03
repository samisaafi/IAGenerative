from config.config import RAGChatbotConfig
from chatbot.rag_chatbot import RAGChatbot


def main():
    config = RAGChatbotConfig()
    bot = RAGChatbot(config)

    print("1. Create new RAG from dataset")
    print("2. Load existing RAG")

    choice = input("Choice: ").strip()

    if choice == "1":
        path = input("Dataset path (CSV/JSON/TXT): ").strip()
        col = input("Text column (optional): ").strip()
        bot.initialize_from_data(path, col if col else None)
    else:
        bot.initialize_from_existing()

    bot.chat()


if __name__ == "__main__":
    main()
