from rag_chatbot import RAGChatbot
import os

def main():
    print("=" * 70)
    print("🎮 Chatbot RAG - Analyse de données avec LM Studio")
    print("=" * 70)
    print("\n⚠️  Assurez-vous que :")
    print("   1. LM Studio est ouvert")
    print("   2. Un modèle est chargé")
    print("   3. Le serveur local est démarré (http://localhost:1234)")
    print("\n" + "=" * 70 + "\n")
    
    # Chemin par défaut vers votre CSV
    default_csv = "data\\vgsales.csv"
    
    csv_path = input(f"📁 Entrez le chemin vers votre CSV (Entrée pour '{default_csv}') : ").strip()
    
    if not csv_path:
        csv_path = default_csv
    
    # Enlever les guillemets
    csv_path = csv_path.strip('"').strip("'")
    
    # Vérifier que le fichier existe
    if not os.path.exists(csv_path):
        print(f"\n❌ Erreur : Le fichier '{csv_path}' n'existe pas")
        return
    
    # Initialiser le chatbot
    try:
        chatbot = RAGChatbot(csv_path=csv_path)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'initialisation : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Afficher les infos sur les données
    print(chatbot.get_data_info())
    
    print("\n" + "=" * 70)
    print("✅ Chatbot prêt ! Posez vos questions sur les données")
    print("   Commandes spéciales :")
    print("   - 'info' : Afficher les informations sur les données")
    print("   - 'quit', 'exit', 'quitter' : Sortir")
    print("=" * 70 + "\n")
    
    # Suggestions de questions
    print("💡 Exemples de questions que vous pouvez poser :")
    print("   - Quel est le jeu le plus vendu ?")
    print("   - Quels sont les meilleurs jeux par plateforme ?")
    print("   - Quelles sont les statistiques de vente par région ?")
    print("   - Quel éditeur a le plus de succès ?")
    print("\n" + "-" * 70 + "\n")
    
    # Boucle de conversation
    while True:
        question = input("👤 Vous : ").strip()
        
        if question.lower() in ['quit', 'exit', 'quitter', 'q']:
            print("\n👋 Au revoir !")
            break
        
        if question.lower() == 'info':
            print(chatbot.get_data_info())
            continue
        
        if not question:
            continue
        
        print("\n⏳ Analyse en cours...\n")
        result = chatbot.ask(question)
        
        print(f"🤖 Assistant : {result['answer']}")
        print(f"\n📊 {len(result['sources'])} sources de données consultées")
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()