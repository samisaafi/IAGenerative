from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from lmstudio_llm import LMStudioLLM
import pandas as pd
import os

class RAGChatbot:
    # Define system prompt as class constant
    SYSTEM_PROMPT = """
Tu es un assistant IA expert en analyse de données de ventes de jeux vidéo.
RÈGLES STRICTES (OBLIGATOIRES) :
- Tu as ACCÈS aux données ci-dessous.
- Tu DOIS répondre à la question en analysant TOUTES les données fournies.
- Tu N'ÉCRIS JAMAIS de code.
- Tu N'EXPLIQUES PAS ta méthode.
- Tu NE DIS JAMAIS "je ne peux pas", "je n'ai pas accès", "je ne sais pas".
- Si un calcul est nécessaire (max, min, somme, comparaison), FAIS-LE en regardant TOUTES les données.
- Si plusieurs informations sont nécessaires, COMBINE-LES.
- Si la réponse est approximative, précise-le clairement.
- Quand on te demande "les meilleurs", donne au moins le TOP 3-5.
Réponds toujours en français, de manière claire et concise.
"""

    def __init__(self, csv_path=None):
        print("🔧 Initialisation du chatbot RAG avec LM Studio...")
        
        # Configuration du modèle LM Studio
        lm_studio_url = "http://localhost:1234/v1"
        self.llm = LMStudioLLM(base_url=lm_studio_url, temperature=0.7)
        print(f"✓ Connexion à LM Studio : {lm_studio_url}")
        
        # Configuration des embeddings (local)
        embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"⏳ Chargement des embeddings : {embeddings_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        print("✓ Embeddings chargés")
        
        # Base de données vectorielle
        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        self.df = None
        
        # Charger le CSV si fourni
        if csv_path:
            self.load_csv(csv_path)
    
    def load_csv(self, csv_path):
        """Charger et analyser le fichier CSV avec encodage sécurisé"""
        print(f"\n📊 Chargement du fichier CSV : {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")

        # Essayez plusieurs encodages pour éviter les erreurs Unicode
        encodings = ["utf-8", "utf-16", "latin1", "cp1252"]
        for enc in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=enc)
                print(f"✓ CSV chargé avec succès (encoding={enc})")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(f"Impossible de lire le fichier {csv_path} avec les encodages standards.")

        print(f"✓ {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"✓ Colonnes : {', '.join(self.df.columns.tolist())}")

        # Créer des documents textuels à partir du CSV
        documents = self._create_documents_from_csv()
        print(f"✓ {len(documents)} documents créés à partir des données")

        # Diviser en chunks plus grands pour avoir plus de contexte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Augmenté de 500 à 1500
            chunk_overlap=200,  # Augmenté de 50 à 200
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ {len(chunks)} chunks créés")

        # Créer la base vectorielle avec client persistant
        print("⏳ Création de la base vectorielle...")
        import chromadb
        from chromadb.config import Settings
        
        # Créer le client avec les bons paramètres
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name="video_games_sales"
        )
        print("✓ Base vectorielle créée et persistée")

        # Créer la chaîne QA
        self._create_qa_chain()

    
    def _create_documents_from_csv(self):
        """Convertir les données CSV en documents textuels avec plus d'informations"""
        documents = []
        
        # Créer un résumé général détaillé
        summary = f"""
Dataset : Données sur les ventes de jeux vidéo
Nombre total d'entrées : {len(self.df)}
Colonnes disponibles : {', '.join(self.df.columns.tolist())}

Résumé statistique complet :
{self.df.describe(include='all').to_string()}
"""
        documents.append(Document(page_content=summary, metadata={"type": "summary"}))
        
        # TOP jeux par ventes globales
        if 'Global_Sales' in self.df.columns and 'Name' in self.df.columns:
            top_games = self.df.nlargest(20, 'Global_Sales')  # TOP 20
            top_text = "TOP 20 des jeux les plus vendus (Global_Sales) :\n"
            for idx, row in top_games.iterrows():
                top_text += f"- {row['Name']} ({row.get('Platform', 'N/A')}): {row['Global_Sales']} millions\n"
            documents.append(Document(page_content=top_text, metadata={"type": "top_games"}))
        
        # TOP jeux par plateforme
        if 'Platform' in self.df.columns and 'Global_Sales' in self.df.columns:
            platforms = self.df['Platform'].unique()[:15]  # Top 15 plateformes
            for platform in platforms:
                platform_data = self.df[self.df['Platform'] == platform].nlargest(10, 'Global_Sales')
                platform_text = f"TOP 10 jeux sur {platform} :\n"
                for idx, row in platform_data.iterrows():
                    platform_text += f"- {row['Name']}: {row['Global_Sales']} millions\n"
                documents.append(Document(page_content=platform_text, metadata={"type": "platform", "platform": platform}))
        
        # Statistiques par région
        region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
        available_regions = [col for col in region_cols if col in self.df.columns]
        if available_regions:
            region_text = "Statistiques de ventes par région (en millions) :\n"
            for col in available_regions:
                total = self.df[col].sum()
                mean = self.df[col].mean()
                region_text += f"- {col}: Total = {total:.2f}M, Moyenne = {mean:.2f}M\n"
            documents.append(Document(page_content=region_text, metadata={"type": "regions"}))
        
        # TOP éditeurs
        if 'Publisher' in self.df.columns:
            top_publishers = self.df.groupby('Publisher')['Global_Sales'].sum().nlargest(15)
            pub_text = "TOP 15 éditeurs par ventes totales :\n"
            for pub, sales in top_publishers.items():
                pub_text += f"- {pub}: {sales:.2f} millions\n"
            documents.append(Document(page_content=pub_text, metadata={"type": "publishers"}))
        
        # TOP genres
        if 'Genre' in self.df.columns:
            top_genres = self.df.groupby('Genre')['Global_Sales'].sum().nlargest(10)
            genre_text = "TOP 10 genres par ventes totales :\n"
            for genre, sales in top_genres.items():
                genre_text += f"- {genre}: {sales:.2f} millions\n"
            documents.append(Document(page_content=genre_text, metadata={"type": "genres"}))
        
        return documents
    
    def _create_qa_chain(self):
        """Créer la chaîne de question-réponse"""
        if not self.vectorstore:
            raise ValueError("Veuillez d'abord charger des données")
        
        # Template de prompt avec SYSTEM_PROMPT intégré
        prompt_template = f"""{self.SYSTEM_PROMPT}

Contexte (données extraites) :
{{context}}

Question : {{question}}

Réponse :"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Créer le retriever avec plus de résultats
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Augmenté de 5 à 10 pour plus de contexte
        )
        
        print("✓ Chaîne QA créée avec succès\n")
    
    def ask(self, question):
        """Poser une question sur les données"""
        if not self.vectorstore:
            return {
                "answer": "Aucune donnée n'a été chargée. Veuillez charger un fichier CSV d'abord.",
                "sources": []
            }
        
        try:
            # Récupérer les documents pertinents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            # Construire le contexte
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Créer le prompt complet
            full_prompt = self.prompt.format(context=context, question=question)
            
            # Obtenir la réponse du modèle
            answer = self.llm(full_prompt)
            
            return {
                "answer": answer,
                "sources": relevant_docs
            }
        
        except Exception as e:
            return {
                "answer": f"Erreur lors de la génération de la réponse : {e}",
                "sources": []
            }
    
    def get_data_info(self):
        """Obtenir des informations sur les données chargées"""
        if self.df is None:
            return "Aucune donnée chargée"
        
        info = f"""
📊 Informations sur les données :
- Nombre de lignes : {len(self.df)}
- Nombre de colonnes : {len(self.df.columns)}
- Colonnes : {', '.join(self.df.columns.tolist())}
"""
        return info


# Example usage
if __name__ == "__main__":
    # Initialize the chatbot with a CSV file
    chatbot = RAGChatbot(csv_path="your_data.csv")
    
    # Get data information
    print(chatbot.get_data_info())
    
    # Ask questions
    questions = [
        "Quel est le jeu qui a le plus de ventes?",
        "Quelle est la moyenne des ventes globales?",
        "Quels sont les 3 meilleurs jeux par région?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = chatbot.ask(question)
        print(f"💬 Réponse: {response['answer']}")
        print(f"📚 Sources utilisées: {len(response['sources'])} documents")