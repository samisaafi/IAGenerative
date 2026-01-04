from openai import OpenAI
from typing import List, Dict, Any, Optional

class LMStudioLLM:
    """Wrapper pour utiliser LM Studio comme LLM"""
    
    def __init__(self, base_url="http://localhost:1234/v1", temperature=0.7):
        """
        Initialise la connexion à LM Studio
        
        Args:
            base_url: URL de l'API LM Studio (par défaut: http://localhost:1234/v1)
            temperature: Créativité des réponses (0.0 à 1.0)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        self.temperature = temperature
        self.base_url = base_url
    
    def __call__(self, prompt: str) -> str:
        """Appeler le modèle LM Studio avec un prompt simple"""
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur de connexion à LM Studio : {str(e)}\nAssurez-vous que LM Studio est lancé avec le serveur actif sur {self.base_url}"
    
    def generate_with_context(self, context: str, question: str) -> str:
        """Génère une réponse avec un contexte fourni"""
        prompt = f"""En vous basant sur le contexte suivant, répondez à la question.

Contexte:
{context}

Question: {question}

Réponse (soyez précis et utilisez uniquement les informations du contexte):"""
        
        return self(prompt)
    
    def test_connection(self) -> bool:
        """Teste la connexion à LM Studio"""
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": "Test de connexion. Répondez simplement par 'OK'."}],
                temperature=0.1,
                max_tokens=10
            )
            return True
        except:
            return False