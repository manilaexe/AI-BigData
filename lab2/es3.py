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

#---INIZIO ESERCIZIO---

import numpy as np
min_depth = 2
max_depth = 10
test_sizes = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
accuracy_list_list = []
size_list = []

# To make faster the execution we reduce the dataset to the first 900 rows
X = X.head(900) #riduce il dataset
y = y.head(900)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for test_size in test_sizes:
    size_list.append(test_size) 
    # Split the dataset
    X_train3, X_test3, y_train3, y_test3=train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # Standardization
    scaler=StandardScaler()
    X_train3_scaled=scaler.fit_transform(X_train3)
    X_test3_scaled=scaler.transform(X_test3)

    # List of accuracies for increasing tree depths
    accuracy_list=[]

    # For each depth in [min_depth, max_depth] traina Tree and append accuracy in accuracy_list
    for depth in range(min_depth, max_depth):
        model3=DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
        model3.fit(X_train3_scaled, y_train3)

        #predictions
        y_pred3=model3.predict(X_test3_scaled)

        #evaluation
        accuracy=accuracy_score(y_test3, y_pred3)
        accuracy_list.append(accuracy)

   # Append accuracy_list in accuracy_list_list
    accuracy_list_list.append(accuracy_list)

