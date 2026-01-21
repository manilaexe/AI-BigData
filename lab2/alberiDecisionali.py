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

#pulizia dei dati
df = df.dropna(axis=1, how='all') #rimuove colonne completamente vuote
df.dropna(subset='status', inplace=True) #rimuove righe in cui la colonna status è vuota

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Convert 'date' column to datetime and extract year
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

# Feature selection
features = [
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'year', 'longitude', 'latitude'
]
# Removed pm2.5_avg, pm10_avg --> features were too correlated with the target (remember Notebook 1) --> "target leakage"

print("\nCount per bin:")
print(df['status'].value_counts(dropna=False))

# Copy of the DataFrame to group the labels
df_group = df.copy()

# Conversion of categorical columns into numeric
df_group['status'] = df_group['status'].replace({
    "Hazardous": "Poor",
    "Very Unhealthy": "Poor",
    "Unhealthy": "Poor"
})

print("\nCount per bin:")
print(df_group['status'].value_counts(dropna=False))

X = df_group[features]
y = df_group['status']

# Split dataset
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
    X, y,
    test_size=0.3,   # 30% for test set and 70% for training set
    random_state=42  # for reproducibility
)

# Scaling
scaler = StandardScaler()
X_train_scaled_group = scaler.fit_transform(X_train_group)  # fit computes parameters (mean μ and std σ) only on training data
X_test_scaled_group  = scaler.transform(X_test_group)       # apply the μ and σ computed from training set

print("\nTraining set shape:", X_train_group.shape)
print("Testing set shape: ", X_test_group.shape)
print("Shapes after scaling:", X_train_scaled_group.shape, X_test_scaled_group.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
model_group = DecisionTreeClassifier(random_state=42)
model_group.fit(X_train_scaled_group, y_train_group)

# Predictions
y_pred_group = model_group.predict(X_test_scaled_group)

# Evaluation
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test_group, y_pred_group):.2f}")

print((636231+324939)/1000000)

print("Features used:", X.columns.tolist())

plt.figure(figsize=(20, 10))
plot_tree(
    model_group,
    max_depth=2,
    feature_names=X.columns,
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=5  # increase the text size in the nodes
)
plt.title("Decision Tree (Depth ≤ 2)")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test_group, y_pred_group)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
