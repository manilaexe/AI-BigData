"""
Questo blocco serve a spiegare cosa farò in questo file:
Il primo blocco sarà lo scheletro, ovvero roba che devi fare sempre
Il secondo blocco avrà lo scheletro e in più i metodi/funzioni aggiuntive che servono per la regressione lineare
Il terzo blocco avrà la parte aggiuntiva della rete neurale fatta apposta per la regressione lineare
Il quarto blocco avrà lo scheletro e in più i metodi/funzioni aggiuntive che servono per l'albero decisionale
Il quinto blocco avrà la parte aggiuntiva della rete neurale fatta apposta per l'albero decisionale
"""
    
"""
Spiegazione per capire quale algoritmo usare:
1)Guardare richiesta esercizio e vedere su cosa vuole il prof target; se questo è object allora abbiamo due 
possibilità:
    -Prima: Il csv è piccolo? Allora bisogna usare l'albero
    -Seconda: Il csv è grande(migliaia o milioni di righe)? Allora puoi o fare l'albero e poi implementare 
    sotto la rete neurale(in questo esercizio lo faccio) o fare quello(non hai altre possibilità)

2)Il target è di tipo int o float(numerico): usare la regressione lineare e se il csv è grande applicare la
rete neurale(nel dubbio fare sempre)

3)Il clustering(un qualcosa uscito dall'inferno): Usare quando(penso) il prof chiede di voler dividere le 
variabili in più gruppi e si usa QUANDO NON HAI UN TARGET
"""

#Questi sono i 5 import che vanno sempre messi in qualsiasi caso, per impararli ho usato una regola speciale: la 
#regola della MNOP S, imparo la prima lettera così che visual studio me la completa da solo
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#La prima cosa da fare SEMPRE è creare un file_path e l'if per vedere se esiste, poi fare il read_csv e controllare 
#che tipologia di dati sono presenti nel dataset

file_path = "data/drug200.csv"
#controllo se esiste il file_path con il csv, in caso non esista darà errore
if not os.path.exists(file_path):
    raise FileNotFoundError("csv non trovato")
#read_csv serve per leggere il csv(autoesplicativo)
df = pd.read_csv(file_path, low_memory=False, na_values=["NA", "ND", "NaN", "", "-"])

df.info()#consiglio: quando inizi l'esame bisogna vedere i dati all'interno del dataset e quindi con df.info() vedi 
#il tipo dei dati e con df.info vedrai che tipo di dati ci sono

print(df.isnull().sum().tolist())#con questa riga vedi quanti nulli ci sono e in caso di righe o colonne vuote fare 
#gli eventuali comandi
#1) elimina le colonne che ha TUTTE le righe vuote
#2) elimina le righe vuote della colonna Drug(il target), lo fai solo di questo perché se lo facessi di tutte 
#elimineresti delle righe di Drug che è la colonna principale
#In questo caso vedendo che non ci sono righe vuote puoi anche non farle queste due righe, ma è buona norma farlo
df = df.dropna(axis=1, how="all")
df = df.dropna(how="any", subset="Drug")

target = df["Drug"]#creare ssempre target della roba che vuole il prof
#Serve per trasformare gli object in numerici, bisogna farlo sempre per tutti e se farlo sul target dipende 
#dall'algoritmo utilizzato
label = LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])
df["BP"] = label.fit_transform(df["BP"])
#....
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()#num_cols ti da le colonne di tipo numerico

#Creo X e y e con il train_test_split metto il 70% in X_train e y_train e il 30% in X_test e y_test
X = df[num_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
#Questo dovrebbe essere lo scheletro che è uguale per tutti ma sinceramente non so neanche se abbia senso fare 
#uno scheletro? Anche perché alla fine sono piccole le cose che cambiano nei vari
#algoritmi, solo albero e regressione

#codice per regressione lineare
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Questi sono gli import necessari per la regressione lineare, qui per impararli ho usato la tecnica speciale 
#del PP MM L(per gli import dopo se non te li ricordi puoi selezionare es.preprocessing 
#fare tasto destro e andare su Go definition, li ci sono tutte le possibilità di import e ti ricordi dopo)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = "data/drug200.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("csv non trovato")
df = pd.read_csv(file_path, low_memory=False, na_values=["NA", "ND", "NaN", "", "-"])

df.info()

print(df.isnull().sum().tolist())
df = df.dropna(axis=1, how="all")
df = df.dropna(how="any", subset="Drug")

target = df["Drug"]

label = LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])
df["BP"] = label.fit_transform(df["BP"])
#....
df[target] = label.fit_transform(df[target])#in questo caso trasformo anche il target in valore perché nella 
#regressione lineare dovrebbe essere di base un numero

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

cm = df[num_cols].corr()
plt.figure(figsize=[10,10])
sns.heatmap(cm, annot=True)#questa è la matrice di correlazione e mostra quanta correlazione c'è fra una variabile 
#all'altra, la correlazione che tieni dipende, ma se vuoi un default puoi eliminare
                           #tutti quelli sotto il -0.15/0.15
plt.show()
#qui poi eliminerai le colonne con poca correlazione
#num_cols.remove("nomeVariabile")
X = df[num_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
#è importante capire cosa fa lo StadardScaler e lo spiegherò in davvero poche righe(per una migliore 
#spiegazione chiedere a chat gpt): range circa -1 a 1; valore es.100000 = 1; valore es. 5= 0.5
scaler = StandardScaler()
#qui facciamo prima fit e poi solo transform perché facendo fit tu crei un'etichetta e essendo che X_test 
#ha gli stessi valori di X_train e quindi le stesse etichette metterò solo tranform, si
#usa quindi solo con valori uguali, sennò mettere sempre fit_transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
Metodo per imparare a memoria le variabili da mettere in queste righe sotto:
fit = addestramento quindi metterò i train
predict = vediamo quando è accurato il modello e quindi lo testiamo con X_test
rmse = è l'accuratezza quindi giustamente confronti i test con le previsioni(pred)
GRAFICI: se vedi lo scatter è uguale a quello che metti in rmse e plot è letteralmente solo y_test 
ripetuto 4 volte
"""
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"L'accuratezza della regressione lineare è: {rmse*100:.2f}")

plt.figure(figsize=[10,10])
plt.scatter(y_test, y_pred, alpha=0.6, color="purple")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color="red")
plt.show()

#se guardi bene il poly è una cazzata tremenda, l'unica cosa da imparare è la prima riga ed è fatta 
#da make_pipeline(PolynomialFeatures(...), StandardScaler()) poi il resto è un copia incolla dal
#model di sopra, il poly è un metodo per verificare se è più accurato, la differenza con la regressione 
#lineare è che la regressione ha una linea retta, invece il poly ha una linea curva
poly = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), StandardScaler())
poly.fit(X_train_scaled, y_train)
y_pred_poly = poly.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = root_mean_squared_error(y_test, y_pred_poly)
print(f"L'accuratezza della regressione lineare è: {rmse_poly*100:.2f}")

#Rete neurale regressione lineare, ancora non ho ben capito il codice sul perché si fa così, però abbiamo 
#una libreria che da il prof chiamato tenserflow e aiuta molto perché la maggior parte
#dei metodi che usiamo sono presenti nella libreria
import tensorflow as tf
import keras
from keras import layers, optimizers

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_val = scaled.transform(X_val)
X_test = scaled.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='linear')])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mse"])

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=64,
                    batch_size=128,
                    verbose=1,)

loss, mse = model.evaluate(X_val, y_val, verbose=0)
print(mse)

y_pred_neur = model.predict(X_test, verbose=0).ravel()

plt.figure(figsize=(6,6))
plt.plot(history.history["mse"], color="pink")
plt.plot(history.history["val_mse"], color="purple")
plt.show()

plt.figure(figsize=(6,6))
plt.plot(history.history["loss"], color="pink")
plt.plot(history.history["val_loss"], color="purple")
plt.show()

#albero decisionale
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

file_path = "data/drug200.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("csv non trovato")

df = pd.read_csv(file_path, low_memory=False, na_values=["NA", "ND", "NaN", "", "-"])

df.info()

df = df.dropna(axis=1, how="all")
df = df.dropna(how="any", subset="Drug")
print(df.isnull().sum().tolist())

# Questo ti crea un grafico e barre e ti dice se il dataset è sbilanciato(le barre sono diverse fra 
#di loro(una molto più lunga e l'altra più corta)) o bilanciato(le barre sono tutte vicine fra di loro)
#A QUANTO PARE QUESTO GRAFICO CHE TI ESCE SERVE SOLO PER FARE DA LECCACULO, serve per dire al prof che 
#sai fare le cose: visto che il grafico è sbilanciato e con poche righe sai che devi fare l'albero
df['Drug'].value_counts(normalize=True).plot(kind='bar')
plt.show()

#qui trasformi tutti gli object in int APPARTE PER IL TARGET CHE DEVI LASCIARLO OBJECT(con l'albero 
#non devi trasformarlo), il motivo dovrebbe essere perché per l'addestramento della macchina 
#vuole tutti int
label = LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])
df["BP"] = label.fit_transform(df["BP"])
df["Cholesterol"] = label.fit_transform(df["Cholesterol"])

target = "Drug"
#questi serviranno poi per i grafici sopra
classes = df[target].unique()
num_classes = len(classes)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#qui serve lo standard scaler tipo nella regressione lineare, nelle reti neurali e nel cluster, nell'albero 
#decisionale è "facoltativo" ma è meglio se non lo metti così dimostri al prof che sei grande
#magari scrivendo un commento dicendo che nell'albero decisionale non serve mettere lo standard scaler
#In questo blocco di codice crei l'albero, lo addesti e predici i valori così poi da vedere quanto è accurato il modello
model = DecisionTreeClassifier(random_state=69, max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#QUESTO CAMBIA PER ALGORITMO:
#Nell'albero si usa accuracy_score, invece nella regressione lineare MSE e RMSE, serve solo per vedere l'accuratezza
acc = accuracy_score(y_test, y_pred)
print(f"accuratezza:{acc*100:.2f}")

#stampa dell'albero
plt.figure(figsize=[10,10])
plot_tree(model, feature_names=num_cols, class_names=classes, filled=True)
plt.show()

cm = confusion_matrix(y_test, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(cmap="Blues")
plt.show()

#Con questo abbiamo TEORICAMENTE finito, ma al prof non va bene che finisca così, quindi bisogna fare più 
#prove, quindi basta fare copia incolla del codice di prima e modificare es.MaxDepth = 3 al
#posto di 4, nel train_test_split mettere 0.4 quindi 60% su X e un 40% su y, per commentare meglio vedere 
#il file del prof che lo dice nel suo metodo soave;
#qui nel grafico notiamo che confonde i dati fra drugC e drugX e tra DrugB e DrugY, quindi proviamo a fare 
#diversi test per vedere cosa succede ai grafici e all'accuratezza

#rete neurale per l'albero decisionale
import tensorflow as tf
import keras
from keras import layers, optimizers
#allora qui c'è da dire una coas importante, essendo che il target è un oggetto, la rete neurale vuole tutti 
#valori e quindi dovremo trasformare target in numero e ricreare y perché prima aveva
#valore oggetto
label = LabelEncoder()
df[target] = label.fit_transform(df[target])
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, random_state=69)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
    ])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss= "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=128,
    verbose=1
)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"accuratezza: {accuracy*100:.2f}%")
class_names = label.classes_
y_pred = model.predict(X_test_scaled).argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="BuPu")
plt.show()

