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









#REGRESSIONE POLINOMIALE NON LA CHIEDE ALL'ESAME
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# Training
poly_model.fit(X_train, y_train)

# Predictions and metric calculations
y_pred_poly = poly_model.predict(X_test)
mse_poly    = mean_squared_error(y_test, y_pred_poly)
rmse_poly   = root_mean_squared_error(y_test, y_pred_poly)

print("Polynomial Regression (degree 2) on the features:", numeric_cols)
print(f"Mean Squared Error: {mse_poly:.2f}")
print(f"Root Mean Squared Error: {rmse_poly:.2f}")

# Scatter plot: actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Match')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Polynomial Regression (degree 2): Predicted vs Actual')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
