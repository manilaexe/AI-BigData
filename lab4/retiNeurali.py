# Import of libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score


import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical

import os
from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/Colab Notebooks' # Set your current working directory where the csv file is located

# Check if file exists
file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")

# Load dataset (first 300,000 rows for testing)
df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=300000)

# Let's check that the column type has been read correctly
print(df.dtypes)

# Target to be predicted
target = "status"

# Drop rows with missing target
df = df.dropna(subset=[target])

print("\nUnique values in the 'status' column after initial loading:")
class_names = df['status'].unique()
num_classes=len(df['status'].unique())
print(class_names)
print("\nNumber of classes:", num_classes)

# display the first 5 rows
df.head()

# Removing columns that are either non-numeric or not useful for prediction
columns_to_drop = ["sitename", "county","aqi","unit","siteid","pollutant","date"]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Handling potential infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns with null values
df = df.dropna(axis=1, how='all')

# Let's select only the numeric columns from the dataframe and remove the target column from them
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

# Drop rows with NaN values only in the numeric features
df = df.dropna(subset=num_cols, how='any')

# Define the feature schema before splitting the data
feature_cols = num_cols.copy()

# Print checks
print(df.info())

print("\nClass count in the cleaned DataFrame:")
print(df[target].value_counts().sort_index())


print("\nDimensions after cleaning:", df.shape)

# Definition of fractions for training and validation
test_frac = 0.2
val_frac = 0.1

# First split: train/test
train_df, test_df = train_test_split(
    df, # Here we do not create X and y like in previous notebooks, just to see a differrent workflow
    test_size=test_frac,
    random_state=42
)

# Second split: within the training set, we also separate the validation set
train_df, val_df = train_test_split(
    train_df,
    test_size=val_frac,
    random_state=42
)

# Check of the classes present in the different sets
all_classes = set(np.unique(df[target]))
train_classes = set(np.unique(train_df[target]))
test_classes  = set(np.unique(test_df[target]))

# Check for missing classes
missing_train = all_classes - train_classes
missing_test  = all_classes - test_classes

# Print checks
print("Missing classes in train:", missing_train)
print("Missing classes in test:", missing_test)

print("Classes present in the train_df:", sorted(train_df[target].unique()))
print("Classes present in test_df:", sorted(test_df[target].unique()))

# Convert the DataFrames into NumPy arrays
X_train = train_df[feature_cols].values
X_val   = val_df[feature_cols].values
X_test  = test_df[feature_cols].values

# Extract the target as a NumPy array of integers
y_train = train_df[target]
y_val   = val_df[target]
y_test  = test_df[target]

# Numeric encoding
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_idx = encoder.transform(y_train)
y_val_idx   = encoder.transform(y_val)
y_test_idx  = encoder.transform(y_test)

# One-hot encoding
y_train_oh = to_categorical(y_train_idx, num_classes=num_classes)
y_val_oh   = to_categorical(y_val_idx,   num_classes=num_classes)
y_test_oh = to_categorical(y_test_idx, num_classes=num_classes)

# Standardization: Bring all features to the same scale (mean = 0, std = 1) to stabilize training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Print checks
print("Training set dimensions:", X_train.shape)
print("Test set dimensions:", X_test.shape)

# Define the model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
# Model filling
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model training
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=50,
    batch_size=128,
    verbose=1
)

# Model evaluation on the test set
loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# Predictions on the test set
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

# Confusion matrix
cm = confusion_matrix(y_test_idx, y_pred, normalize="true")
plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

# Visualization of the loss trend
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss trend during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.grid(True)
plt.show()

# Visualization of the validation accuracy trend
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Validation Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy trend during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)
plt.show()

