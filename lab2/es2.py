

#---INIZIO ESERCIZIO---

# Copy of the original DataFrame
df_ex2 = df.copy()

# import LabelEncoder
from sklearn.preprocessing import LabelEncoder 

# Encode of county
le_country=LabelEncoder()
df_ex2["county"]=le_country.fit_transform(df_ex2["county"].astype(str))

# Encode of sitename
le_sitename=LabelEncoder()
df_ex2["sitename"]=le_sitename.fit_transform(df_ex2["sitename"].astype(str))

print(df_ex2[["county", "sitename"]].dtypes) #stampa il tipo di oggetto

# Select only columns from features
X2 = df_ex2[features]
y2 = df_ex2['status']

# Split dataset starting from X_ex2 and y_ex2. Remember to set random_state=42
X_train2, X_test2, y_train2, y_test2=train_test_split(
    X2, y2,
    test_size=0.3,
    random_state=42
)

# Scaling
scaler=StandardScaler()
X_train2_scaled=scaler.fit_transform(X_train2)
X_test2_scaled=scaler.transform(X_test2)

print("\nTraining set shape", X_train2.shape)
print("Testing set shape", X_test2.shape)
print("Shapes after scaling: ", X_train2_scaled.shape, X_test2_scaled.shape)

# Initialize and train the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model2=DecisionTreeClassifier(random_state=42)
model2.fit(X_train2_scaled, y_train2)

# Predictions
y_pred2=model2.predict(X_test2_scaled)

# Evaluation
print("\nModel Performance: ")
print(f"Accuracy: {accuracy_score(y_test2, y_pred2):.2f}")

# Confusion Matrix
cm2=confusion_matrix(y_test2, y_pred2)
disp2=ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=sorted(y2.unique()))
disp2.plot(cmap='Purples')
plt.title("Matrice confusa come me")
plt.show()
