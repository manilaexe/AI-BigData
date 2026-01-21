# Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/...' # Set your current working directory where the csv file is located

# Check if file exists
file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")

# Load dataset (first 1,000,000 rows for testing)
df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA','ND', 'n/a', ''], nrows=1000000)

df = df.dropna(axis=1, how='all')
df.dropna(subset='status', inplace=True)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df['date'] = pd.to_datetime(df['date'], errors='coerce') #converte date in formato datetime
df['year'] = df['date'].dt.year #estrae l'anno come feature numerica utile per trend temporali

#scelta delle features 
features = [
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'year', 'longitude', 'latitude'
] #feature numeriche per predire status
#tolte pm2.5_avg e pm1-_avg per avitare target leakage (sono troppo correlati con l'output)
# Removed pm2.5_avg, pm10_avg --> features were too correlated with the target (remember Notebook 1) --> "target leakage"

print("\nCount per bin:") 
print(df['status'].value_counts(dropna=False))

# Copy of the DataFrame to group the labels
df_group = df.copy()

#riorganizza le classi del target -> tutte le classi pericolose diventano "Poor"
#lóbiettivo è ridurre il numero di classi e bilanciare il dataset
df_group['status'] = df_group['status'].replace({
    "Hazardous": "Poor",
    "Very Unhealthy": "Poor",
    "Unhealthy": "Poor"
})

print("\nCount per bin:")
print(df_group['status'].value_counts(dropna=False))

#definizione degli assi x e y 
X = df_group[features] #contiene le colonne/features numeriche selezionate per predire il target
y = df_group['status'] #contiene la colonna target "status" da predire

# Split dataset
#divide il dataset con il 70% per il training e il 30% per il test
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
    X, y,
    test_size=0.3,   # 30% for test set and 70% for training set
    random_state=42  # for reproducibility
)

#standardizzazine
#normalizza le features per avere media 0 e deviazione standard 1
scaler = StandardScaler()
X_train_scaled_group = scaler.fit_transform(X_train_group)  # fit computes parameters (mean μ and std σ) only on training data
X_test_scaled_group  = scaler.transform(X_test_group)       # apply the μ and σ computed from training set

print("\nTraining set shape:", X_train_group.shape)
print("Testing set shape: ", X_test_group.shape)
print("Shapes after scaling:", X_train_scaled_group.shape, X_test_scaled_group.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#inizializza e addestra il modello 
model_group = DecisionTreeClassifier(random_state=42) #crea un albero decisionale 
                                                      #ogni nodo è una regola di split su una feature
                                                      #le foglie sono la classe predetta (Poor, Moderate...)
model_group.fit(X_train_scaled_group, y_train_group) #addestra il modello sui dati normalizzati di training

# Predictions
y_pred_group = model_group.predict(X_test_scaled_group) #predice le classi del test set

# Evaluation
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test_group, y_pred_group):.2f}")

print((636231+324939)/1000000) #calcola la percentuale di dati in due categorie specifiche sul totale di un milione di righe (numeri presi a mano dall'output precedente

print("Features used:", X.columns.tolist()) #passa a plot_tree i nomi delle features da mostrare nei nodi dell'albero

#GRAFICO
plt.figure(figsize=(20, 10)) 
plot_tree(
    model_group,
    max_depth=2, #mostra i primi 2 livelli dell'albero
    feature_names=X.columns, #indica al plot_tree quali sono i nomi delle feature da mostrare nei nodi nell'albero
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True, #colora i nodi in base alla classe predominante
    rounded=True, #stonda i bordi
    fontsize=5  #cambia la dimensione del font
)
plt.title("Decision Tree (Depth ≤ 2)")
plt.show()

#MATRICE DI CONFUSIONE
#confronta predizioni con valori reali
cm = confusion_matrix(y_test_group, y_pred_group)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique())) #mostra le etichette ordinate
disp.plot(cmap='Blues') #scala di colori
plt.title("Confusion Matrix")
plt.show()
