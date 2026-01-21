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
  model[i]=LinearRegression()
  model[i].fit(X_train_set, y_train)

  # Check performance
  y_pred=model[i].predict(X_test_set)
  mse[i]=mean_squared_error(y_test, y_pred)
  rmse[i]=root_mean_squared_error(y_test, y_pred)

  print(f"Prestazioni modello {i+1}")
  print(f"Mean squared errore: {mse[i]:.2f}")
  print(f"Root mean square error: {rmse[i]:.2f}")

# Plot comparison of predicted vs actual values for all the models
# Suggestion: create a different plot for each model or create a single plot where
# the results of the worst models first because they have larger scatter plots which,
# if printed last, would completely cover the scatter plots of the best models.
for i in range(len(featureSet)):
  plt.scatter(y_test, y_pred[3-1], alpha=0.6, label=f"Predicted values for model {i+1}")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Regressione lineare: Valori previsti e Valori reali")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
