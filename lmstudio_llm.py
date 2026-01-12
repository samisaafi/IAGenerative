import requests
import json

class LMStudioLLM:
    """Wrapper pour utiliser LM Studio comme backend LLM via l'API OpenAI (compatible Python 3.13)"""
    
    def __init__(self, base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=2000):
        """
        Initialise le client LM Studio
        
        Args:
            base_url: URL du serveur LM Studio (par défaut: http://localhost:1234/v1)
            temperature: Température pour la génération (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens à générer
        """
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_endpoint = f"{self.base_url}/chat/completions"
    
    def __call__(self, prompt):
        """
        Génère une réponse à partir d'un prompt
        
        Args:
            prompt: Le prompt à envoyer au modèle
            
        Returns:
            str: La réponse générée par le modèle
        """
        try:
            # Préparer la requête
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Envoyer la requête
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            # Vérifier la réponse
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Erreur HTTP {response.status_code}: {response.text}"
            
        except requests.exceptions.ConnectionError:
            return f"❌ Erreur de connexion à LM Studio sur {self.base_url}\n\n💡 Vérifiez que:\n   1. LM Studio est lancé\n   2. Un modèle est chargé\n   3. Le serveur est démarré sur le port 1234"
        
        except requests.exceptions.Timeout:
            return "❌ Timeout: Le modèle met trop de temps à répondre. Essayez avec un prompt plus court."
        
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
    def generate(self, prompt, **kwargs):
        """
        Méthode alternative pour la génération (compatible avec certaines interfaces LangChain)
        """
        return self.__call__(prompt)