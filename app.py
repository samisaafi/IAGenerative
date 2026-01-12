from flask import Flask, render_template_string, request, jsonify
from rag_chatbot import RAGChatbot

app = Flask(__name__)

# HTML embarqué directement dans le code
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot RAG - Ventes de Jeux Vidéo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.9; }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.5;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .sources {
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            font-style: italic;
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        #questionInput {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        #questionInput:focus { border-color: #667eea; }
        #sendBtn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        #sendBtn:hover { transform: scale(1.05); }
        #sendBtn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: scale(1);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading.active { display: block; }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            padding: 20px;
            background: #f9f9f9;
            border-bottom: 1px solid #e0e0e0;
        }
        .examples h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: white;
            border: 1px solid #667eea;
            color: #667eea;
            border-radius: 15px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        .welcome {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        .welcome h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Chatbot RAG</h1>
            <p>Analyse des ventes de jeux vidéo</p>
        </div>
        <div class="examples">
            <h3>💡 Questions suggérées :</h3>
            <button class="example-btn" onclick="askExample('Quel est le jeu le plus vendu ?')">Jeu le plus vendu</button>
            <button class="example-btn" onclick="askExample('Quels sont les meilleurs jeux par plateforme ?')">Meilleurs par plateforme</button>
            <button class="example-btn" onclick="askExample('Quelles sont les statistiques de vente par région ?')">Stats par région</button>
            <button class="example-btn" onclick="askExample('Quel éditeur a le plus de succès ?')">Meilleur éditeur</button>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="welcome">
                <h2>Bienvenue ! 👋</h2>
                <p>Posez-moi des questions sur les ventes de jeux vidéo.</p>
                <p>Vous pouvez cliquer sur les suggestions ci-dessus ou taper votre propre question.</p>
            </div>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyse en cours...</p>
        </div>
        <div class="input-container">
            <input type="text" id="questionInput" placeholder="Posez votre question ici..." onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendQuestion()">Envoyer</button>
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const questionInput = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');

        function addMessage(text, isUser, sourcesCount) {
            const welcome = chatContainer.querySelector('.welcome');
            if (welcome) welcome.remove();
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            let content = '<div class="message-content">' + text;
            if (!isUser && sourcesCount > 0) {
                content += '<div class="sources">📚 ' + sourcesCount + ' sources consultées</div>';
            }
            content += '</div>';
            messageDiv.innerHTML = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;
            addMessage(question, true, 0);
            questionInput.value = '';
            sendBtn.disabled = true;
            loading.classList.add('active');
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                if (!response.ok) throw new Error('Erreur de connexion');
                const data = await response.json();
                addMessage(data.answer, false, data.sources_count);
            } catch (error) {
                addMessage('❌ Erreur : ' + error.message, false, 0);
            } finally {
                sendBtn.disabled = false;
                loading.classList.remove('active');
            }
        }

        function askExample(question) {
            questionInput.value = question;
            sendQuestion();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') sendQuestion();
        }
    </script>
</body>
</html>
'''

# Initialiser le chatbot au démarrage
print("🚀 Initialisation du chatbot...")
csv_path = "data/vgsales.csv"
chatbot = RAGChatbot(csv_path=csv_path)
print("✅ Chatbot prêt !")

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'Question vide'}), 400
    response = chatbot.ask(question)
    return jsonify({
        'answer': response['answer'],
        'sources_count': len(response['sources'])
    })

if __name__ == '__main__':
    import os
    os.environ['FLASK_SKIP_DOTENV'] = '1'
    print("\n" + "="*70)
    print("🌐 Interface web lancée sur : http://localhost:5000")
    print("📱 Ouvrez votre navigateur et accédez à cette URL")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, load_dotenv=False)