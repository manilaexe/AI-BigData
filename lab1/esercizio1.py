import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import os

drive.mount('/content/drive')

cwd = 'drive/MyDrive/Colab Notebooks'

file_path=cwd+'/air_quality.csv'
if not os.path.exists(file_path):
  raise FileNotFoundError(f"il file non esiste")

df=pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''], nrows=1000000)

#Dal DataFrame risultante, rimuovi le colonne in cui i valori mancanti superano il 5% delle righe totali. Stampa la forma del DataFrame risultante.
missing_percentage=df.isna().mean() #calcolo la percentuale di valori mancanti per colonna
df_clean=df.loc[:, missing_percentage<=0.05] #seleziono solo le colonne con meno o uguali al 5% valori mancanti
print(df_clean.shape)

#Dal DataFrame risultante, contiamo e rimuoviamo tutte le righe in cui abbiamo almeno 3 colonne con un valore mancante. Stampiamo la forma del DataFrame risultante.
missing_per_row=df_clean.isnull().sum(axis=1) #conta i valori mancanti per riga
print("righe con almeno 3 valori mancanti: ", (missing_per_row>=3).sum())
df_clean=df_clean.loc[missing_per_row<3] #rimuove le righe con almeno 3 colonne nulle
print(df_clean.shape)

#Scrivere le statistiche del DataFrame risultante.
print("Info dataset")

print("Statistiche descrtittive")
print(df_clean.describe(include='all'))

print("percentuale di valori mancanti per colonna")
print(df_clean.isnull().mean()*100)

#Mostra la heatmap dei valori mancanti utilizzando Seaborn
plt.figure(figsize=(12,6))
sns.heatmap(df_clean.isnull(), cbar=False, cmap='viridis')#matrice dove ho i valori nulli
plt.title("Valori mancanti")
plt.show()

#Rappresenta graficamente l'aqi rispetto alla data per le prime 365 righe per il nome del sito 'Hukou'
df_hukou=df_clean[df_clean['sitename']=="Hukou"] #filtro i dati per sito
df_hukou_365=df_hukou.head(365)
# Convert the 'date' column to datetime objects
#df_hukou_365['date'] = pd.to_datetime(df_hukou_365['date'])

plt.figure(figsize=(20,5))
plt.plot(df_hukou_365['date'], df_hukou_365['aqi'], marker='o', linestyle='-', linewidth=2)
plt.title("AQI vs data")
plt.xlabel("Data")
plt.ylabel("AQI")
plt.xticks(rotation=45, fontsize = 2) #ruota le etichette
plt.grid(True) #attiva la griglia di sfondo
plt.show()
