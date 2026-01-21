#regressione più semplice crea una retta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/...' #carica dov'è il dataset

#controlla se il file esiste
file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")

#carica le prime 1000000 righe
df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=1000000)

#DATA CLEANING
empty_cols = df.columns[df.isna().all()].tolist() #rimuove tutte le colonne con valori completamente mancanti
print("Columns removed because completely empty:", empty_cols)

from sklearn.preprocessing import LabelEncoder

df = df.dropna(axis=1, how='all') #cancella le colonne completamente mancanti

#trasformo tutte le features in numerico
le_pollutant = LabelEncoder()
le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))
le_sitename = LabelEncoder()
df["sitename"] = le_sitename.fit_transform(df["sitename"].astype(str))

# Select only numeric columns (excluding 'aqi', which is the target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() #seleziono solo le features numeriche
numeric_cols.remove('aqi')  #perché è la colonna target


df_clean = df.dropna(subset=numeric_cols + ['aqi']).copy() #

X = df_clean[numeric_cols]
y = df_clean['aqi']

print("Numeric columns retained:", numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #split del dataset 

#creo il modello
model = LinearRegression() #classe che permette di addestrare il regressore lineare
model.fit(X_train, y_train) #con fit riesco ad andare addestrare il modello fit(dataset di training, dataset di training) 

#VALUTAZIONE DEL MODELLO
y_pred = model.predict(X_test) #per usare il modello addestrato e predizione etichette

#come si muove aqi nel dataset (maxl, min, range)
print(f"Minimum value of aqi: {y_test.min():.2f}")
print(f"Maximum value of aqi: {y_test.max():.2f}")
print(f"Range of values: {y_test.max()-y_test.min():.2f}")

#metriche
mse = mean_squared_error(y_test, y_pred) #calcolo la squared error 
rmse = root_mean_squared_error(y_test, y_pred) #calcolo la root di square error
rmse_mine = np.sqrt(mse) #controllo se la funzione è corretta (non serve)

# Output
print("\n\nModel Performance (Multivariate Linear Regression):")
print(f"Mean Squared Error: {mse:.2f}") #importante perché mi porta ad usare la stessa unità di misura dell'etichetta 
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Root Mean Squared Error (computed starting from MSE): {rmse_mine:.2f} <- clearly equals to above value")
print(f"RMSE on aqi range (%): {(rmse/(y_test.max()-y_test.min()))*100:.2f}") #percentuale errore relativo

#PLOT
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect prediction') 
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual AQI vs Predicted AQI")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Create lists containing linear regression models and results for each model
models = [LinearRegression() for i in range(4)]
y_preds = [None for i in range(4)]
mses = [None for i in range(4)]
rmses = [None for i in range(4)]

# Define the set of features for model 1 Remove 'sitename'
feature1=numeric_cols.copy()
feature1.remove('sitename')

# Define the set of features for model 2 Remove information about geographical position ('siteid', 'sitename', 'county', 'latitude', 'longitude')
feature2=feature1.copy()
feature2.remove('siteid')
feature2.remove('county')
feature2.remove('latitude')
feature2.remove('longitude')

# Define the set of features for model 3 Remove geographical position and average values of pollutant ('pm2.5_avg', 'pm10_avg', 'so2_avg')
feature3=feature2.copy()
feature3.remove('pm2.5_avg')
feature3.remove('pm10_avg')
feature3.remove('so2_avg')

# Define the set of features for model 4 Remove geographical position, avg pollutant values and wind information (removed 'windspeed', 'winddirec')
feature4=feature3.copy()
feature4.remove('windspeed')
feature4.remove('winddirec')

# For each set of features
featureSet=[feature1, feature2, feature3, feature4]

for i in range(len(featureSet)):
  # Remove features from X
  X_train_set=X_train[featureSet[i]]
  X_test_set=X_test[featureSet[i]]

  #check sizes
  print(f"\nSize considering feature set for model {i+1}")
  print(f"X_train_set: {X_train_set.shape}")
  print(f"X_tain_set: {X_test_set.shape}")

  # Create and train a new Linear Regression model
  models[i]=LinearRegression()
  models[i].fit(X_train_set, y_train)

  # Check performance
  y_preds[i]=models[i].predict(X_test_set)
  mses[i]=mean_squared_error(y_test, y_preds[i])
  rmses[i]=root_mean_squared_error(y_test, y_preds[i])

  print(f"Prestazioni modello {i+1}")
  print(f"Mean squared errore: {mses[i]:.2f}")
  print(f"Root mean square error: {rmses[i]:.2f}")

# Plot comparison of predicted vs actual values for all the models
# Suggestion: create a different plot for each model or create a single plot where
# the results of the worst models first because they have larger scatter plots which,
# if printed last, would completely cover the scatter plots of the best models.
for i in range(len(featureSet)):
  plt.scatter(y_test, y_preds[3-i], alpha=0.6, label=f"Predicted values for model {i+1}")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Regressione lineare: Valori previsti e Valori reali")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


