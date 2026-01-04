import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pickle
import os

class CSVProcessor:
    """Classe pour traiter et analyser les données CSV"""
    
    def __init__(self, csv_path: str):
        """
        Initialise le processeur CSV
        
        Args:
            csv_path: Chemin vers le fichier CSV
        """
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.chroma_client = None
        self.collection = None
        
    def load_data(self) -> pd.DataFrame:
        """Charge les données CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Données chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
            print(f"📊 Colonnes: {list(self.df.columns)}")
            
            # Afficher les premières lignes
            print("\n📋 Aperçu des données:")
            print(self.df.head())
            
            # Statistiques de base
            print("\n📈 Statistiques de base:")
            print(f"  - Période: {self.df['Year'].min()} à {self.df['Year'].max()}")
            print(f"  - Jeux uniques: {self.df['Name'].nunique()}")
            print(f"  - Plateformes: {self.df['Platform'].nunique()}")
            print(f"  - Genres: {self.df['Genre'].nunique()}")
            
            return self.df
        except Exception as e:
            print(f"❌ Erreur lors du chargement du CSV: {e}")
            return None
    
    def analyze_data(self):
        """Analyse les données et génère des insights"""
        if self.df is None:
            print("❌ Veuillez d'abord charger les données avec load_data()")
            return
        
        print("\n🔍 Analyse approfondie des données:")
        
        # Ventes totales par région
        print("\n💰 Ventes totales (en millions):")
        if 'NA_Sales' in self.df.columns:
            print(f"  - Amérique du Nord: {self.df['NA_Sales'].sum():.2f}")
        if 'EU_Sales' in self.df.columns:
            print(f"  - Europe: {self.df['EU_Sales'].sum():.2f}")
        if 'JP_Sales' in self.df.columns:
            print(f"  - Japon: {self.df['JP_Sales'].sum():.2f}")
        if 'Other_Sales' in self.df.columns:
            print(f"  - Autres régions: {self.df['Other_Sales'].sum():.2f}")
        if 'Global_Sales' in self.df.columns:
            print(f"  - Mondiales: {self.df['Global_Sales'].sum():.2f}")
        
        # Top 10 des jeux
        print("\n🏆 Top 10 des jeux les plus vendus:")
        top_games = self.df.nlargest(10, 'Global_Sales')[['Name', 'Platform', 'Year', 'Genre', 'Global_Sales']]
        print(top_games.to_string(index=False))
        
        # Par plateforme
        print("\n🎮 Ventes par plateforme (top 10):")
        platform_sales = self.df.groupby('Platform')['Global_Sales'].sum().nlargest(10)
        print(platform_sales.to_string())
        
        # Par genre
        print("\n🎭 Ventes par genre:")
        genre_sales = self.df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
        print(genre_sales.to_string())
        
        # Par année
        print("\n📅 Ventes par année:")
        yearly_sales = self.df.groupby('Year')['Global_Sales'].sum().tail(10)
        print(yearly_sales.to_string())
    
    def prepare_for_rag(self, persist_directory: str = "./chroma_db"):
        """Prépare les données pour le RAG avec ChromaDB"""
        print("\n🔄 Préparation des données pour RAG...")
        
        # Charger le modèle d'embedding
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Modèle d'embedding chargé")
        except:
            print("⚠️  Utilisation d'embeddings simples (fallback)")
            self.model = None
        
        # Initialiser ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Créer ou obtenir la collection
        collection_name = "vg_sales_data"
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"✅ Collection '{collection_name}' chargée")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Données de ventes de jeux vidéo"}
            )
            print(f"✅ Collection '{collection_name}' créée")
        
        # Vérifier si la collection est vide
        if self.collection.count() == 0:
            self._create_embeddings()
        
        print(f"✅ Base vectorielle prête: {self.collection.count()} documents")
    
    def _create_embeddings(self):
        """Crée les embeddings pour les données"""
        print("  📝 Création des embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in self.df.iterrows():
            # Créer un texte riche pour chaque jeu
            doc_text = f"""
Jeu: {row['Name']}
Plateforme: {row['Platform']}
Année: {row['Year']}
Genre: {row['Genre']}
Éditeur: {row['Publisher']}
Ventes Amérique du Nord: {row['NA_Sales']} millions
Ventes Europe: {row['EU_Sales']} millions
Ventes Japon: {row['JP_Sales']} millions
Ventes autres régions: {row['Other_Sales']} millions
Ventes mondiales: {row['Global_Sales']} millions
"""
            documents.append(doc_text)
            metadatas.append({
                'name': str(row['Name']),
                'platform': str(row['Platform']),
                'year': int(row['Year']) if not pd.isna(row['Year']) else 0,
                'genre': str(row['Genre']),
                'publisher': str(row['Publisher']),
                'global_sales': float(row['Global_Sales'])
            })
            ids.append(str(idx))
        
        # Ajouter les documents à ChromaDB
        if self.model:
            # Utiliser SentenceTransformer pour les embeddings
            embeddings = self.model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            # Sans embedding personnalisé
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"  ✅ {len(documents)} documents ajoutés à la base vectorielle")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Recherche des jeux similaires à la requête"""
        if not self.collection:
            print("❌ Base vectorielle non initialisée")
            return []
        
        try:
            if self.model:
                # Avec embedding personnalisé
                query_embedding = self.model.encode([query]).tolist()[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                # Recherche textuelle simple
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            
            # Formater les résultats
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            return formatted_results
        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return []