# 1. IMPORT DELLE LIBRERIE

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 2. CARICAMENTO DEL DATASET

# (In Colab spesso si monta Google Drive)
# drive.mount('/content/drive')

# Percorso del file CSV
file_path = "path/al/file.csv"

# Controllo esistenza file
if not os.path.exists(file_path):
    raise FileNotFoundError("File CSV non trovato")

# Lettura del dataset
df = pd.read_csv(
    file_path,
    low_memory=False,
    na_values=['-', 'NA', 'ND', 'n/a', '']
)

# Rimozione colonne completamente vuote
df.dropna(axis=1, how='all', inplace=True)

# Rimozione righe senza target
df.dropna(subset=['target'], inplace=True)

# 3. PREPROCESSING / FEATURE ENGINEERING

# Conversione colonne temporali (se presenti)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year   # feature numerica utile

# Encoding di variabili categoriche (se presenti)
le = LabelEncoder()
df['categoria_encoded'] = le.fit_transform(df['categoria'].astype(str))

# 4. SELEZIONE FEATURE E TARGET

features = [
    'feature1',
    'feature2',
    'feature3',
    'year'
]

X = df[features]      # matrice delle feature
y = df['target']      # vettore target

print("Distribuzione classi:")
print(y.value_counts())

# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% test
    random_state=42      # riproducibilità
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 6. SCALING (STANDARDIZZAZIONE)

scaler = StandardScaler()

# fit SOLO sul training
X_train_scaled = scaler.fit_transform(X_train)

# transform anche sul test
X_test_scaled = scaler.transform(X_test)

print("Shapes after scaling:",
      X_train_scaled.shape,
      X_test_scaled.shape)

# 7. TRAINING DEL MODELLO

model = DecisionTreeClassifier(
    max_depth=5,         # iperparametro
    random_state=42,
    class_weight='balanced'  # utile se classi sbilanciate
)

model.fit(X_train_scaled, y_train)

# 8. PREDIZIONI

y_pred = model.predict(X_test_scaled)

# 9. VALUTAZIONE DEL MODELLO

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y.unique())
)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 10. ANALISI IPERPARAMETRI (ESERCIZIO CLASSICO)

min_depth = 2
max_depth = 10
test_sizes = np.arange(0.2, 0.9, 0.1)

accuracy_list_list = []

for test_size in test_sizes:

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )

    # scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    accuracy_list = []

    for depth in range(min_depth, max_depth):
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        accuracy_list.append(accuracy_score(y_te, y_pred))

    accuracy_list_list.append(accuracy_list)

# 11. GRAFICO FINALE

x_axis = list(range(min_depth, max_depth))

plt.figure(figsize=(10,6))

for i, acc in enumerate(accuracy_list_list):
    plt.plot(x_axis, acc, label=f"test_size={test_sizes[i]:.2f}")

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Tree Depth")
plt.legend()
plt.grid(True)
plt.show()
