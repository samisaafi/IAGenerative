import openai

class LMStudioLLM:
    """Wrapper pour utiliser LM Studio comme LLM (OpenAI legacy API)"""

    def __init__(self, base_url="http://localhost:1234/v1", temperature=0.7):
        """
        Initialise la connexion à LM Studio

        Args:
            base_url: URL de l'API LM Studio
            temperature: Créativité des réponses (0.0 à 1.0)
        """
        openai.api_base = base_url
        openai.api_key = "not-needed"
        self.temperature = temperature
        self.base_url = base_url

    def __call__(self, prompt: str) -> str:
        """Appeler LM Studio avec un prompt simple"""
        try:
            response = openai.ChatCompletion.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return (
                f"Erreur de connexion à LM Studio : {str(e)}\n"
                f"Vérifiez que LM Studio est lancé avec le serveur actif sur {self.base_url}"
            )

    def generate_with_context(self, context: str, question: str) -> str:
        """Génère une réponse en utilisant un contexte fourni"""
        prompt = f"""Répondez à la question en utilisant uniquement le contexte ci-dessous.

Contexte:
{context}

Question: {question}

Réponse:"""
        return self(prompt)

    def test_connection(self) -> bool:
        """Teste la connexion à LM Studio"""
        try:
            response = openai.ChatCompletion.create(
                model="local-model",
                messages=[{"role": "user", "content": "Réponds uniquement par OK."}],
                temperature=0.0,
                max_tokens=5
            )
            return True
        except Exception:
            return False
