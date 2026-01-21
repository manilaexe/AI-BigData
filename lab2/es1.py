#Build and analyse the tree considering the original dataset, considering all the 6 classes.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/Colab Notebooks'  

file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV non trovato in: {file_path}")

df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''],nrows=1_000_000)

df = df.dropna(axis=1, how='all') # Rimuove colonne completamente vuote

df.dropna(subset=['status'], inplace=True)# Rimuove righe senza target

df['date'] = pd.to_datetime(df['date'], errors='coerce') # Conversione della data

df['year'] = df['date'].dt.year # Estrazione dell'anno come feature numerica

# Feature numeriche
features = [
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'year', 'longitude', 'latitude'
]


print("\nDistribuzione classi originali (6 classi):")
print(df['status'].value_counts(dropna=False))

#---INIZIO ESERCIZIO---#

# Copy of the original DataFrame
df_ex1 = df.copy()

# Select only columns from features
X_ex1 = df_ex1[features]
y_ex1 = df_ex1['status']

# Split dataset starting from X_ex1 and y_ex1. Remember to set random_state=42
X_train_ex1, X_test_ex1, y_train_ex1, y_test_ex1=train_test_split(
    X_ex1, y_ex1,
    test_size=0.3,
    random_state=42
)
# Scaling(standardizzazione)
scaler=StandardScaler()
X_train_scaled_ex1=scaler.fit_transform(X_train_ex1)
X_test_scaled_ex1=scaler.transform(X_test_ex1)

print("\nTrain set shape: ", X_train_ex1.shape)
print("Test set shape: ", X_test_ex1.shape)
print("Shapes after scaling", X_train_ex1.shape, X_test_ex1.shape)

# Initialize and train the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model_ex1=DecisionTreeClassifier(random_state=42)
model_ex1.fit(X_train_ex1, y_train_ex1)

# Predictions
y_pred_ex1=model_ex1.predict(X_test_ex1)

# Evaluation
print("\nModel Performance: ")
print(f"Accuracy: {accuracy_score(y_test_ex1, y_pred_ex1):.2f}")

#matrice di confusione
mex1=confusion_matrix(y_test_ex1, y_pred_ex1)
dispex1=ConfusionMatrixDisplay(confusion_matrix=mex1, display_labels=sorted(y_ex1.unique()))
dispex1.plot(cmap='Purples')
plt.show()
