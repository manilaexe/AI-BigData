# =============================================================================
# Importazione delle librerie necessarie
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
### 1. Caricamento del dataset e analisi esplorativa preliminare
In questa sezione il dataset viene caricato in un DataFrame Pandas.
Successivamente vengono eseguiti controlli iniziali per:
- verificare la struttura dei dati e i tipi delle variabili,
- individuare eventuali valori mancanti o anomali,
- osservare la distribuzione della variabile target (classi del farmaco).
"""

# Definizione del percorso del file
file_path = "./drug200.csv"

# Verifica dell’esistenza del file nel percorso specificato
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File non trovato nel percorso: {file_path}")

# Lettura del dataset con gestione esplicita di valori mancanti (es. NA, ND, stringhe vuote, ecc.).
# Il dataset è di dimensioni contenute, quindi può essere caricato interamente in memoria
df = pd.read_csv(file_path, low_memory=False, na_values=['NA', 'ND', '', '-'])

# Visualizzazione delle informazioni principali del DataFrame (colonne, tipi, non-null)
print(df.info())

# Analisi della distribuzione percentuale della variabile target tramite grafico a barre
df['Drug'].value_counts(normalize=True).plot(kind='bar')
plt.show()

# Visualizzazione delle prime righe per un controllo rapido dei dati caricati
df.head()

"""
### 2. Pre-processing e codifica delle variabili categoriali
Poiché il dataset è di dimensioni ridotte ed è sbilanciato, e visto che non è stato richiesto un modello specifico 
procedo con un albero decisionale. Il dataset contiene variabili categoriali (es. sesso, pressione sanguigna, colesterolo),
e l’implementazione degli alberi decisionali di scikit-learn non supporta direttamente feature categoriali,
si procede con una codifica numerica tramite Label Encoding.

Nota:
- Non viene applicata normalizzazione delle feature, poiché gli alberi decisionali non ne traggono beneficio
  (sono in larga parte invarianti rispetto a trasformazioni di scala).
- Non ci sono valori missing quindi non serve fare una pulizia di righe o colonne.
- Non ho conoscienza del significato e delle relazioni rta loro delle etichette, quindi non raggruppo le etichette
"""

from sklearn.preprocessing import LabelEncoder

# Stampa delle etichette originali presenti nelle colonne categoriali
print("Etichette originali per Sex, BP, Cholesterol: ")
print(np.unique(df['Sex']))
print(np.unique(df['BP']))
print(np.unique(df['Cholesterol']))

# Applicazione del Label Encoding alle colonne categoriali
lencoder = LabelEncoder()
df['Sex'] = lencoder.fit_transform(df['Sex'])
df['BP'] = lencoder.fit_transform(df['BP'])
df['Cholesterol'] = lencoder.fit_transform(df['Cholesterol'])

# Verifica delle etichette numeriche ottenute dopo la trasformazione
print("\nEtichette codificate (numeriche): ")
print(np.unique(df['Sex']))
print(np.unique(df['BP']))
print(np.unique(df['Cholesterol']))

# Controllo della struttura del dataset dopo la trasformazione
df.info()
df.head()

"""
### 3. Addestramento del modello di classificazione (Decision Tree)
In questa sezione viene addestrato un classificatore Decision Tree per prevedere la classe del farmaco (Drug)
a partire dalle variabili numeriche disponibili.

Scelte progettuali:
- Suddivisione del dataset in training set e test set con rapporto 70% / 30%.
- Impostazione di max_depth = 4 per limitare la complessità del modello e ridurre il rischio di overfitting. In base ai risultati vedrò se aumentare la profondità.
- OPZIONALE: Uso di random_state per garantire riproducibilità dei risultati.
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Selezione automatica delle feature numeriche
features = df.select_dtypes(include=[np.number]).columns.tolist()
nfeatures = len(features)

# Identificazione delle classi target presenti nel dataset
classes = np.unique(df['Drug'])
n_classes = len(classes)

# Definizione della variabile target
target = 'Drug'

# Creazione di matrice delle feature (X) e vettore target (y)
X = df[features]
y = df[target]

# Suddivisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=69
)

# Inizializzazione e addestramento del modello Decision Tree
model = DecisionTreeClassifier(random_state=69, max_depth=4)
model.fit(X_train, y_train)

# Visualizzazione grafica dell’albero decisionale appreso
plt.title("Visualizzazione dell’albero decisionale (Decision Tree)")
plot_tree(model, feature_names=features, class_names=classes, filled=True)
plt.show()

"""
### 4. Valutazione delle prestazioni del modello
La valutazione viene effettuata sul test set utilizzando metriche standard per la classificazione:
- Accuracy: percentuale di predizioni corrette.
- Matrice di confusione normalizzata: utile per analizzare l’accuratezza per classe.
"""

# Generazione delle predizioni sul test set
y_pred = model.predict(X_test)

# Calcolo e stampa delle metriche di performance
print(f"Accuracy Decision Tree: {accuracy_score(y_test, y_pred)}")

# Calcolo della matrice di confusione normalizzata per riga (true labels)
cm = confusion_matrix(y_test, y_pred, normalize='true')

# Visualizzazione della matrice di confusione tramite heatmap
sns.heatmap(cm,
            vmin=0.0, vmax=1.0,
            cmap='Blues', linewidths=0.5, fmt='0.2f',
            annot=True,
            xticklabels=classes, yticklabels=classes)
plt.title("Matrice di confusione normalizzata")
plt.xlabel("Classi predette")
plt.ylabel("Classi reali")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

"""
COMMENTI SUI RISULTATI
L'accuratezza ottenuta è 0.967, sicuramente molto buona. La matrice di correlazione mostra che il modello
tende a confondere alcune DrugX con DrugY e un buon numero di DrugB con DrugA. Probabilmente ci sono similitudini tra X-Y
e tra A-B-C. Si potrebbe provare a modificare alcuni parametri o non limitare la profondità dell'albero per vedere
come cambia il comportamento del modello.
"""

"""
### 5. Esperimenti aggiuntivi e analisi di robustezza OPZIONALI
In questa sezione vengono effettuati ulteriori esperimenti per valutare la sensibilità del modello rispetto a:
- una riduzione della profondità massima dell’albero (max_depth = 3),
- rimozione limite profondità albero.

L’obiettivo è verificare se le prestazioni rimangono stabili e identificare possibili segnali di overfitting
o variazioni significative dovute alla configurazione del modello.
"""


# Addestramento di un albero con profondità più contenuta per valutare l’impatto sulla performance
model = DecisionTreeClassifier(random_state=69, max_depth=3)
model.fit(X_train, y_train)

# Visualizzazione dell’albero ottenuto
plt.title("Visualizzazione dell’albero decisionale (max_depth=3)")
plot_tree(model, feature_names=features, class_names=classes, filled=True)
plt.show()

# Predizioni e metriche sul test set
y_pred = model.predict(X_test)

print(f"Accuracy Decision Tree (max_depth=3): {accuracy_score(y_test, y_pred)}")

# Matrice di confusione normalizzata
cm = confusion_matrix(y_test, y_pred, normalize='true')

sns.heatmap(cm,
            vmin=0.0, vmax=1.0,
            cmap='Blues', linewidths=0.5, fmt='0.2f',
            annot=True,
            xticklabels=classes, yticklabels=classes)
plt.title("Matrice di confusione normalizzata (max_depth=3)")
plt.xlabel("Classi predette")
plt.ylabel("Classi reali")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Addestramento con la stessa profondità del modello principale (max_depth=4) ma con train:test = 60:40
model = DecisionTreeClassifier(random_state=69)
model.fit(X_train, y_train)

plt.title("Visualizzazione dell’albero decisionale (no max_depth")
plot_tree(model, feature_names=features, class_names=classes, filled=True)
plt.show()

y_pred = model.predict(X_test)

print(f"Accuracy Decision Tree (no max_depth): {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred, normalize='true')

sns.heatmap(cm,
            vmin=0.0, vmax=1.0,
            cmap='Blues', linewidths=0.5, fmt='0.2f',
            annot=True,
            xticklabels=classes, yticklabels=classes)
plt.title("Matrice di confusione normalizzata (no max_depth)")
plt.xlabel("Classi predette")
plt.ylabel("Classi reali")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

"""
COMMENTI SUI RISULTATI
Riducendo la profondità massima il modello va peggio. Togliendo il limite di pronfondità ottengo lo stesso albero,
sintomo che max_depth=4 era sufficiente a gestire il dataset. 
"""
