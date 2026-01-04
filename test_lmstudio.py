from openai import OpenAI

print("🔍 Test de connexion à LM Studio...\n")

try:
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )
    
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": "Dis bonjour en français"}],
        temperature=0.7
    )
    
    print("✅ Connexion réussie !")
    print(f"🤖 Réponse du modèle : {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Erreur : {e}")
    print("\n⚠️  Vérifiez que :")
    print("   1. LM Studio est ouvert")
    print("   2. Un modèle est chargé")
    print("   3. Le serveur est démarré")