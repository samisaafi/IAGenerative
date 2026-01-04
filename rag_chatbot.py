from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from lmstudio_llm import LMStudioLLM
import pandas as pd
import os

class RAGChatbot:
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
        """Charger et analyser le fichier CSV"""
        print(f"\n📊 Chargement du fichier CSV : {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
        
        # Lire le CSV
        self.df = pd.read_csv(csv_path)
        print(f"✓ CSV chargé : {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"✓ Colonnes : {', '.join(self.df.columns.tolist())}")
        
        # Créer des documents textuels à partir du CSV
        documents = self._create_documents_from_csv()
        print(f"✓ {len(documents)} documents créés à partir des données")
        
        # Diviser en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ {len(chunks)} chunks créés")
        
        # Créer la base vectorielle
        print("⏳ Création de la base vectorielle...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("✓ Base vectorielle créée et persistée")
        
        # Créer la chaîne QA
        self._create_qa_chain()
    
    def _create_documents_from_csv(self):
        """Convertir les données CSV en documents textuels"""
        documents = []
        
        # Créer un résumé général
        summary = f"""
Dataset : Données sur les ventes de jeux vidéo
Nombre total d'entrées : {len(self.df)}
Colonnes disponibles : {', '.join(self.df.columns.tolist())}

Résumé statistique :
{self.df.describe(include='all').to_string()}
"""
        documents.append(Document(page_content=summary))
        
        # Créer un document pour chaque ligne (limité aux 1000 premières pour la performance)
        max_rows = min(1000, len(self.df))
        for idx, row in self.df.head(max_rows).iterrows():
            # Convertir chaque ligne en texte descriptif
            row_text = " | ".join([f"{col}: {row[col]}" for col in self.df.columns if pd.notna(row[col])])
            documents.append(Document(page_content=row_text))
        
        # Créer des documents d'agrégation si des colonnes numériques existent
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            agg_text = "Statistiques agrégées :\n"
            for col in numeric_cols:
                agg_text += f"{col} - Total: {self.df[col].sum():.2f}, Moyenne: {self.df[col].mean():.2f}, Max: {self.df[col].max():.2f}\n"
            documents.append(Document(page_content=agg_text))
        
        return documents
    
    def _create_qa_chain(self):
        """Créer la chaîne de question-réponse"""
        if not self.vectorstore:
            raise ValueError("Veuillez d'abord charger des données")
        
        # Template de prompt adapté pour les données
        prompt_template = """Tu es un assistant spécialisé dans l'analyse de données. Tu réponds aux questions en te basant UNIQUEMENT sur les données fournies dans le contexte.

Contexte (données extraites) :
{context}

Question : {question}

Instructions :
- Réponds en français
- Base-toi UNIQUEMENT sur les données fournies dans le contexte
- Si la réponse nécessite des calculs, explique ton raisonnement
- Si l'information n'est pas dans les données, dis "Je ne trouve pas cette information dans les données"
- Sois précis et donne des chiffres quand c'est pertinent

Réponse :"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Créer le retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Récupérer plus de résultats pour les données
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