import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('movie_statistic_dataset.csv')

# Suponiendo que 'genres' contiene una lista de géneros separados por comas
df['genres'] = df['genres'].apply(lambda x: x.split(',')[0])  # Tomamos el primer género

# Codificar la variable objetivo (género)
le = LabelEncoder()
df['genres'] = le.fit_transform(df['genres'])

print(df.columns)

df = df.dropna(subset=['runtime_minutes'])  # Drop rows with missing runtime

df['production_year'] = pd.to_datetime(df['production_date']).dt.year

X = df[['production_year', 'genres', 'runtime_minutes']] # Updated features
y = df['genres']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def crear_modelo(num_capas, num_neuronas, activacion, regularizador):
    model = keras.Sequential()
    for _ in range(num_capas):
        model.add(layers.Dense(num_neuronas, activation=activacion, kernel_regularizer=regularizador))
    model.add(layers.Dense(len(le.classes_), activation='softmax'))  # Capa de salida
    return model

# Ejemplo de 3 modelos diferentes:
modelos = [
    crear_modelo(2, 64, 'relu', None),            # Modelo 1: 2 capas, 64 neuronas, ReLU
    crear_modelo(3, 32, 'tanh', keras.regularizers.l2(0.01)),  # Modelo 2: 3 capas, 32 neuronas, Tanh, L2
    crear_modelo(1, 128, 'sigmoid', keras.regularizers.l1(0.01)) # Modelo 3: 1 capa, 128 neuronas, Sigmoid, L1
]

for model in modelos:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) 

for i, model in enumerate(modelos):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Modelo {i+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")