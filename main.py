from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List
from fuzzywuzzy import fuzz


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
detected_emotions = []


# Função para ler textos de um arquivo
def read_texts_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        texts = [text.strip() for text in texts]  # Remover espaços em branco
        return texts
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return []

# Leitura dos textos
alegria = read_texts_from_file(alegria_path)
tristeza = read_texts_from_file(tristeza_path)
raiva = read_texts_from_file(raiva_path)
medo = read_texts_from_file(medo_path)
surpresa = read_texts_from_file(surpresa_path)
saudacoes_usuario = read_texts_from_file(saudacoes_usuario_path)
respostas_saudacoes_felizes = read_texts_from_file(respostas_saudacoes_felizes_path)
saudacoes_tristes = read_texts_from_file(saudacoes_tristes_path)
saudacoes_raiva = read_texts_from_file(saudacoes_raiva_path)
saudacoes_medo = read_texts_from_file(saudacoes_medo_path)
saudacoes_surpresa = read_texts_from_file(saudacoes_surpresa_path)
despedidas_usuario = read_texts_from_file(despedidas_usuario_path)
despedidas_sistema = read_texts_from_file(despedidas_sistema_path)
perguntas_dados_tecnicos = read_texts_from_file(perguntas_dados_tecnicos_path)

# Verificação de leitura de textos
if not (alegria and tristeza and raiva and medo and surpresa and saudacoes_usuario and respostas_saudacoes_felizes
        and saudacoes_tristes and saudacoes_raiva and saudacoes_medo and saudacoes_surpresa and despedidas_usuario and despedidas_sistema and perguntas_dados_tecnicos):
    print("Erro na leitura dos arquivos de texto.")
    exit()

# Combinação dos textos e criação dos rótulos
texts = alegria + tristeza + raiva + medo + surpresa
labels = ([0] * len(alegria)) + ([1] * len(tristeza)) + ([2] * len(raiva)) + ([3] * len(medo)) + ([4] * len(surpresa))

# Tokenização e padronização
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 10
data = pad_sequences(sequences, maxlen=maxlen)

# Convertendo para arrays NumPy
data = np.array(data)
labels = to_categorical(np.array(labels), num_classes=5)

# Carregando o modelo salvo
model_path = r"C:\Users\pedro\OneDrive\Área de Trabalho\pedro\modelo_emocoes.h5"
modelo_carregado = load_model(model_path)
print("Modelo carregado com sucesso.")

# Função para pré-processar o texto
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

# Função para prever emoção
def predict_emotion(text):
    preprocessed_text = preprocess_text(text)
    prediction = modelo_carregado.predict(preprocessed_text)
    emotion_classes = ["Alegria", "Tristeza", "Raiva", "Medo", "Surpresa"]
    emotion = emotion_classes[np.argmax(prediction)]
    return emotion

# Função para verificar similaridade de texto
def is_similar(text, texts, threshold=0.5):
    vectorizer = TfidfVectorizer().fit_transform([text] + texts)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]
    max_similarity = np.max(similarity_scores)
    return max_similarity > threshold

# Função para detectar se o texto é uma saudação
def is_greeting(text, threshold=80):
    for greeting in saudacoes_usuario:
        if fuzz.ratio(text.lower(), greeting.lower()) > threshold:
            return True
    return False

def is_farewell(text, threshold=80):
    for farewell in despedidas_usuario:
        if fuzz.ratio(text.lower(), farewell.lower()) > threshold:
            return True
    return False

# Função para encontrar a pergunta mais similar
def find_most_similar_question(user_input, questions):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + questions)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]
    max_similarity_idx = np.argmax(similarity_scores)
    similar_question = questions[max_similarity_idx]
    similarity_score = similarity_scores[max_similarity_idx]
    return similar_question, similarity_score

# Função para extrair ID da pergunta
def extract_id_from_question(question):
    match = re.search(r'\bID (\d+)\b', question)
    if match:
        return int(match.group(1))
    return None

def extract_onu_id_and_intent(question):
    patterns = {
        'estado': r"(estado|status)",
        'uptime': r"(uptime|tempo de atividade)",
        'causa_queda': r"(causa da (última )?queda|motivo da (última )?queda)",
        'detalhes_tecnicos': r"(detalhes técnicos|especificações|dados técnicos)",
        'rx_power': r"(potência de recepção|rx power)",
        'tx_power': r"(potência de transmissão|tx power)",
        'sn': r"(número de série|sn)",
        'descricao': r"(descrição)",
        'tipo': r"(tipo)",
        'distancia': r"(distância)",
        'onlines': r"(onlines|estado 'online')"
    }
    
    # Extrair o ID da ONU
    onu_id_match = re.search(r'ONU (\d+)', question, re.IGNORECASE)
    onu_id = int(onu_id_match.group(1)) if onu_id_match else None
    
    # Identificar intenção
    for intent, pattern in patterns.items():
        if re.search(pattern, question, re.IGNORECASE):
            return intent, onu_id
    
    return None, onu_id
def generate_technical_response(intent, onu_id):
    dados_tecnicos = consultar_dados_tecnicos()
    onu_data = next((onu for onu in dados_tecnicos["ONUs"] if onu["id"] == onu_id), None) if onu_id is not None else None

    if intent == 'estado':
        if onu_data:
            return f"O estado da ONU com ID {onu_id} é {onu_data['state']}."
        else:
            return "Aqui estão os estados de todas as ONUs:\n" + "\n".join([f"ID {onu['id']}: {onu['state']}" for onu in dados_tecnicos["ONUs"]])
    
    elif intent == 'uptime':
        if onu_data:
            return f"O uptime da ONU com ID {onu_id} é {onu_data['uptime']}."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'causa_queda':
        if onu_data:
            return f"A causa da última queda da ONU com ID {onu_id} foi {onu_data['downcause']}."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'detalhes_tecnicos':
        if onu_data:
            return (
                f"Aqui estão alguns detalhes sobre a ONU com ID {onu_id}: "
                f"Estado: {onu_data['state']}, "
                f"Uptime: {onu_data['uptime']}, Downtime: {onu_data['downtime']}, "
                f"Causa da Queda: {onu_data['downcause']}, SN: {onu_data['sn']}, "
                f"Tipo: {onu_data['type']}, Distância: {onu_data['distance']}m, "
                f"Rx Power: {onu_data['rx_power']}dBm, Tx Power: {onu_data['tx_power']}dBm, "
                f"Descrição: {onu_data['description']}."
            )
        else:
            return "ID da ONU não encontrado."
    elif intent == 'rx_power':
        if onu_data:
            return f"A potência de recepção (Rx Power) da ONU com ID {onu_id} é {onu_data['rx_power']}dBm."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'tx_power':
        if onu_data:
            return f"A potência de transmissão (Tx Power) da ONU com ID {onu_id} é {onu_data['tx_power']}dBm."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'sn':
        if onu_data:
            return f"O número de série (SN) da ONU com ID {onu_id} é {onu_data['sn']}."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'descricao':
        if onu_data:
            return f"A descrição da ONU com ID {onu_id} é {onu_data['description']}."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'tipo':
        if onu_data:
            return f"O tipo da ONU com ID {onu_id} é {onu_data['type']}."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'distancia':
        if onu_data:
            return f"A distância da ONU com ID {onu_id} é {onu_data['distance']}m."
        else:
            return "ID da ONU não encontrado."

    elif intent == 'onlines':
        onlines = [onu for onu in dados_tecnicos["ONUs"] if onu["state"] == "online"]
        if onlines:
            return "As seguintes ONUs estão online:\n" + "\n".join([f"ID {onu['id']}: {onu['description']}" for onu in onlines])
        else:
            return "Nenhuma ONU está online no momento."

    return "Intenção não reconhecida ou ID da ONU não encontrado."
@app.get('/api/suporte-tecnico')
def consultar_dados_tecnicos():
    return {
        "ONUs": [
            {
                "id": 0,
                "state": "online",
                "uptime": "2024-09-03 17:31:17",
                "downtime": "2024-09-03 17:10:49",
                "downcause": "dying-gasp",
                "sn": "4648545409031423",
                "type": "E4L-H5410WA",
                "distance": 1844,
                "rx_power": -24.09,
                "tx_power": 2.49,
                "description": "fabio.luiz"
            },
            {
                "id": 1,
                "state": "online",
                "uptime": "2024-09-02 15:28:13",
                "downtime": "2024-09-02 15:26:45",
                "downcause": "dying-gasp",
                "sn": "495442538B3EC6BF",
                "type": "110Gb",
                "distance": 2008,
                "rx_power": -21.55,
                "tx_power": 2.01,
                "description": "elienson.santana"
            },
            {
                "id": 2,
                "state": "online",
                "uptime": "2024-07-10 10:00:50",
                "downtime": "2024-07-10 09:59:46",
                "downcause": "dying-gasp",
                "sn": "44443138B3D4241C",
                "type": "F601",
                "distance": 1877,
                "rx_power": -22.00,
                "tx_power": 2.45,
                "description": "lucelita.araujo"
            },
            {
                "id": 3,
                "state": "online",
                "uptime": "2024-08-30 15:50:46",
                "downtime": "2024-08-30 15:49:12",
                "downcause": "dying-gasp",
                "sn": "495442538BADD444",
                "type": "110Gb",
                "distance": 1841,
                "rx_power": -21.42,
                "tx_power": 0.98,
                "description": "cleide.maria"
            }
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(user_input: str = Form(...)):
    emotion = predict_emotion(user_input)
    detected_emotions.append(emotion)

    intent, onu_id = extract_onu_id_and_intent(user_input)
    response = None

    if intent:
        response = generate_technical_response(intent, onu_id)

    if not response:
        if emotion == "Alegria":
            response = "Oi, que alegria te ver! Em que posso te ajudar hoje?"
        elif emotion == "Tristeza":
            response = "Sinto muito que você esteja triste. Como posso te ajudar?"
        elif emotion == "Raiva":
            response = "Parece que você está chateado. O que posso fazer para ajudar?"
        elif emotion == "Medo":
            response = "Percebo que você está com medo. Como posso ajudar a te tranquilizar?"
        elif emotion == "Surpresa":
            response = "Você está surpreso! O que aconteceu?"

    return JSONResponse(content={"emotion": emotion, "response": response})

@app.post("/humanized-response", response_class=JSONResponse)
async def humanized_response(user_input: str = Form(...)):
    emotion = predict_emotion(user_input)
    
    # Encontrar a pergunta mais similar
    similar_question, similarity_score = find_most_similar_question(user_input, perguntas_dados_tecnicos)

    # Extrair o ID da ONU da pergunta similar
    onu_id = extract_id_from_question(similar_question)
    
    response = None
    if onu_id is not None:
        dados_tecnicos = consultar_dados_tecnicos()
        onu_data = next((onu for onu in dados_tecnicos["ONUs"] if onu["id"] == onu_id), None)
        if onu_data:
            response = (
                f"Olá! Eu sou uma IA e posso te ajudar com informações técnicas. "
                f"Aqui estão alguns detalhes sobre a ONU com ID {onu_id}: "
                f"Estado: {onu_data['state']}, "
                f"Uptime: {onu_data['uptime']}, Downtime: {onu_data['downtime']}, "
                f"Causa da Queda: {onu_data['downcause']}, SN: {onu_data['sn']}, "
                f"Tipo: {onu_data['type']}, Distância: {onu_data['distance']}m, "
                f"Rx Power: {onu_data['rx_power']}dBm, Tx Power: {onu_data['tx_power']}dBm, "
                f"Descrição: {onu_data['description']}."
            )
    else:
        response = "Eu sou uma IA de machine learning e não fui treinado o suficiente para responder essa resposta."

    return JSONResponse(content={"emotion": emotion, "response": response})

@app.get("/api/emotions-stats", response_class=JSONResponse)
def get_emotions_stats():
    emotion_counts = {
        "Alegria": detected_emotions.count("Alegria"),
        "Tristeza": detected_emotions.count("Tristeza"),
        "Raiva": detected_emotions.count("Raiva"),
        "Medo": detected_emotions.count("Medo"),
        "Surpresa": detected_emotions.count("Surpresa")
    }
    return JSONResponse(content=emotion_counts)

dados_tecnicos = {
    "ONUs": [
        {
            "id": 0,
            "state": "online",
            "uptime": "2024-09-03 17:31:17",
            "downtime": "2024-09-03 17:10:49",
            "downcause": "dying-gasp",
            "sn": "4648545409031423",
            "type": "E4L-H5410WA",
            "distance": 1844,
            "rx_power": -24.09,
            "tx_power": 2.49,
            "description": "fabio.luiz"
        },
        {
            "id": 1,
            "state": "online",
            "uptime": "2024-09-02 15:28:13",
            "downtime": "2024-09-02 15:26:45",
            "downcause": "dying-gasp",
            "sn": "495442538B3EC6BF",
            "type": "110Gb",
            "distance": 2008,
            "rx_power": -21.55,
            "tx_power": 2.01,
            "description": "elienson.santana"
        },
        {
            "id": 2,
            "state": "online",
            "uptime": "2024-07-10 10:00:50",
            "downtime": "2024-07-10 09:59:46",
            "downcause": "dying-gasp",
            "sn": "44443138B3D4241C",
            "type": "F601",
            "distance": 1877,
            "rx_power": -22.00,
            "tx_power": 2.45,
            "description": "lucelita.araujo"
        },
        {
            "id": 3,
            "state": "online",
            "uptime": "2024-08-30 15:50:46",
            "downtime": "2024-08-30 15:49:12",
            "downcause": "dying-gasp",
            "sn": "495442538BADD444",
            "type": "110Gb",
            "distance": 1841,
            "rx_power": -21.42,
            "tx_power": 0.98,
            "description": "cleide.maria"
        }
    ]
}

@app.get("/adicionar-cliente", response_class=HTMLResponse)
async def get_adicionar_cliente(request: Request):
    return templates.TemplateResponse("adicionar_cliente.html", {"request": request})

@app.post("/adicionar-cliente", response_class=JSONResponse)
async def post_adicionar_cliente(
    state: str = Form(...),
    uptime: str = Form(...),
    downtime: str = Form(...),
    downcause: str = Form(...),
    sn: str = Form(...),
    type: str = Form(...),
    distance: int = Form(...),
    rx_power: float = Form(...),
    tx_power: float = Form(...),
    description: str = Form(...),
    field_name: List[str] = Form([]),
    field_value: List[str] = Form([])
):
    novo_cliente = {
        "id": len(dados_tecnicos["ONUs"]),
        "state": state,
        "uptime": uptime,
        "downtime": downtime,
        "downcause": downcause,
        "sn": sn,
        "type": type,
        "distance": distance,
        "rx_power": rx_power,
        "tx_power": tx_power,
        "description": description
    }
    
    for name, value in zip(field_name, field_value):
        novo_cliente[name] = value
    
    dados_tecnicos["ONUs"].append(novo_cliente)
    
    return JSONResponse(content={"message": "Cliente adicionado com sucesso!", "cliente": novo_cliente})

@app.get("/clientes", response_class=HTMLResponse)
async def get_clientes(request: Request):
    return templates.TemplateResponse("clientes.html", {"request": request, "dados_tecnicos": dados_tecnicos})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")