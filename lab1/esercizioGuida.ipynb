#diventa come un pc linux, monta il drive
from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/Colab Notebooks' # Set your current working directory where the csv file is located

#importare le librerie
import os 
import pandas as pd #per manipolazione
import matplotlib.pyplot as plt #per i grafici
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# definisco dov'è il file e se esiste
file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")

# uso pandas per leggere il csv 
df = pd.read_csv(file_path,low_memory=False, na_values=['-', 'NA','ND', 'n/a', ''], nrows=1000000)  # prendo solo nrows righe e sostituisco con na_values tutte le righe che non hanno valore
    #df=dataset completo (dataframe come se fossi su excel/csv)


print(f"Original dataset size: {df.shape}")

print("\nFirst 5 rows of the dataset:")
print(df.head()) #stampo i primi 5 elementi per definizione head(n) 

print("\nDataset info:") 
print(df.info()) #informazioni dataset
                 #id, etichetta colonna, numero colonne non-nulle, tipo di dato
                 #se ho TANTI valori nulli posso cestinarli 
                 #se ho righe con etichette mancanti conviene cancellare le righe

print("\nDescriptive statistics:")
print(df.describe(include='all')) #informazioni descrittive

print("\nDistribution of numerical columns:")
df.hist(figsize=(12,6)) # Here the dataframe create a plot using matplot lib. The next rows set how to show the plots
plt.tight_layout() # Better layout
plt.show() # Shows the plot

df = df.dropna(axis=1, how='all') #cancello le colonne in cui tutti i valori della colonna sono NaN (0=riga, 1=colonna) 

print("\nPercentage of missing values per column (after dropping empty columns):")
print(df.isnull().mean() * 100) #calcolo la percentuale di righe nulle
print(f"Dataset size (should be the original size): {df.shape}\n\n") 

df.dropna(subset='status', inplace=True) #toglie le righe dove lo stato ha dei valori mancanti
print(df.isnull().mean() * 100)
print(f"Final dataset size after removing rows where status has missing values: {df.shape}")

#one-hot-encoding -> se ho una variabile categoria con n valori diversi non numerici posso traspformare questi valori in features diverse
#che mi creano una rappresentazione vettoriale di quei valori, vettori di tutti 0 tranne uno a 1 che mi indica il valore correto 

from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer #classe OneHotEncoder
df_alt = df.copy() #copio il Dataset

encoder = OneHotEncoder(sparse_output=False, drop=None)  # drop=None tiene tutte le colonne
status_reshaped = df_alt['status'].values.reshape(-1, 1)
status_onehot = encoder.fit_transform(status_reshaped) #posso passare l'array e ottengo la codifica one hot

#inserisco i nuovi valori del dataframe specificando il nome delle colonne
status_onehot_df = pd.DataFrame(
    status_onehot,
    columns=[f"status_{cat}" for cat in encoder.categories_[0]],
    index=df_alt.index
)

df_alt = pd.concat([df_alt, status_onehot_df], axis=1)#concateno i dataframe

print("\nExample of one-hot encoding on 'status':")
print(df_alt.head())

#converte i valori cateogrici in numerici 
df["status"] = df["status"].replace({ #definisco un vocabolario
    "Hazardous": 5,
    "Very Unhealthy": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Moderate": 1,
    "Good": 0
})
print(df)

from sklearn.preprocessing import LabelEncoder  #Automatic with LabelEncoder

#uso il vocabolario interno a sklearn
df["pollutant"] = df["pollutant"].fillna("Unknown")

le_pollutant = LabelEncoder()
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))

le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))

le_sitename = LabelEncoder()
df["sitename"] = le_sitename.fit_transform(df["sitename"].astype(str))


print(df[["pollutant", "status", "county", "sitename"]].dtypes) #stampa che tipo di oggetto e'

print("\nDistribution of the target variable 'status':") 
print(df["status"].value_counts())


df_alt = df.copy()


bins = [0, 50, 100, 150, 200, 300, 500] #creo intervalli
labels_num = [0, 1, 2, 3, 4, 5]  #etichette a intervalli

df_alt['aqi_discretized'] = pd.cut( #creo colonna numerica
    df_alt['aqi'], #con .cut inserisco i valori discretizzati con le soglie di bins 
    bins=bins,
    labels=labels_num,
    right=True,
    include_lowest=True
).astype('Int64')

df_alt = df_alt[df_alt['aqi_discretized'].notna()] #rimuovo i nulli

print("\nFirst rows with discretized AQI (official threshold):")
print(df_alt[['aqi', 'aqi_discretized']].head(5)) #stampo

print("\nCount per bin:")
print(df_alt['aqi_discretized'].value_counts(dropna=False))

#GRAFICO SCATTER PLOT
plt.figure(figsize=(12, 6)) #dimensione figura
colors = df_alt['match'].map({True: 'orange', False: 'red'}) #scelgo i colori
plt.scatter(df_alt.index, df_alt['status'], c=colors, alpha=0.6)
plt.xlabel("Examples (dataset rows)") #etichetta asse x
plt.ylabel("Status (0=Good ➜ 5=Hazardous)") #etichetta asse y
plt.title("Comparison between 'status' and 'aqi_discretized'") #titolo grafico 
plt.show() #lo stampa

#MATRICE DI CONFUSIONE
import seaborn as sns #fatta per heatmap

cm = pd.crosstab(df_alt['status'], df_alt['aqi_discretized'], rownames=['status'], colnames=['aqi_bin']) #creo matrice
cm_norm = cm.div(cm.sum(axis=1), axis=0) #la normalizzo

plt.figure(figsize=(7, 5)) #altra figura
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1) 
plt.xlabel("Discretized AQI")
plt.ylabel("Status")
plt.title("Normalized confusion matrix")
plt.tight_layout() #grafico stretto 
plt.show()


#conviene farlo
#come si distribuiscono i valori mancanti
print("Count of missing values per column:") 
print(df.isnull().sum()) #conto valori nulli 

#CREO HEATMAP capisco dove ho carenze di raccolta dati 
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis') #matrice dove ho null
plt.title('Missing Values Map')
plt.show()


# Convert 'pm2.5_avg' to numeric and drop rows with missing 'status' or 'pm2.5_avg'
df['pm2.5_avg'] = pd.to_numeric(df['pm2.5_avg'], errors='coerce')
df = df.dropna(subset=['status', 'pm2.5_avg'])

# Direct plot of pm2.5_avg vs status
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['pm2.5_avg'], y=df['status'], alpha=0.5)
plt.title("pm2.5_avg vs Air Quality Status")
plt.xlabel("pm2.5_avg")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()

#ESERCIZIO1
#Converti i valori di data da stringhe in oggetti datetime, che Python e Pandas possono interpretare come date reali
df['date']=pd.to_datetime(df['date'], errors='coerce') #trasforma una riga ripo 2024-06-12 15:00 in unoggetto datetime, errore serve a mettere una data a NaN se malformata
df['hour']=df['date'].dt.hour #estrae l'ora e crea una feature "hour"

#definisci le features per il plot
df['no2']=pd.to_numeric(df['no2'], errors='coerce') #converte la colonna no2 in numeri
df['o3']=pd.to_numeric(df['o3'], errors='coerce') #converte la colonna no2 in numeri 
features_to_plot=['no2', 'o3', 'hour']

#genera uno scatter plot per ogni feature
for feature in features_to_plot: #itera su tutte le features
  plt.figure(figsize=(12,6)) 
  colors=df_alt['match'].map({True: 'green', False: 'red'}) 
  sns.scatterplot(data=df, x=feature, y='status', alpha=0.5) #.scatterplot(dataframe=dataframe di orgine, assex, assey, trasparenza per vedere i punti)
  plt.xlabel(f"{feature} vs Air quality") #la f permette di inserire il valore di una variabile direttamente dentro il testo
  plt.ylabel(feature)
  plt.title(f"{feature} vs Air Quality Status")
  plt.tight_layout()
  plt.show()



