import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Dados de exemplo (substitua pelos seus dados reais)
texts = [
    "Eu estou muito feliz!",
    "Sinto uma tristeza profunda.",
    "Estou tão irritado com isso.",
    "Isso me deixa com muito medo.",
    "Que surpresa maravilhosa!",
    # Adicione mais exemplos aqui...
]

labels = [0, 1, 2, 3, 4]  # Labels correspondentes: 0 - Alegria, 1 - Tristeza, 2 - Raiva, 3 - Medo, 4 - Surpresa

# Tokenização e padronização
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 10  # Comprimento máximo das sequências
data = pad_sequences(sequences, maxlen=maxlen)

# Convertendo labels para categorias
labels = to_categorical(np.array(labels), num_classes=5)

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Construindo o modelo
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=128, input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(5, activation='softmax'))  # 5 classes de emoções

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Salvando o modelo
model.save('modelo_emocoes.h5')
print("Modelo treinado e salvo com sucesso.")
