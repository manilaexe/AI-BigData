

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
