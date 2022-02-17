import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


df = pd.read_csv('/content/iris.csv')
df.head()

df.describe()


df.info()


df['Class']=df['Class'].map({
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
    
})


df.head(5)



X = df[['Sepal_Length', ' Sepal_Width', ' Petal_Length', ' Petal_Width']]
X


y=df[['Class']]
y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier(criterion='entropy')
id3_model = id3.fit(X_train, y_train)


y_id3 = id3_model.predict(X_test)
y_id3



accuracy = accuracy_score(y_test, y_id3)
recall = recall_score(y_test, y_id3, average='micro')
precision = precision_score(y_test, y_id3, average='micro')
f1 = f1_score(y_test, y_id3, average='micro')
print("Accuracy:", accuracy)
print("Error Rate:", 1.0-accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)



from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_id3)
cm


print(classification_report(y_test,y_id3))




cart = DecisionTreeClassifier(criterion='gini')
cart_model = cart.fit(X_train, y_train)
y_cart = cart_model.predict(X_test)
y_cart



accuracy = accuracy_score(y_test, y_cart)
recall = recall_score(y_test, y_cart, average='micro')
precision = precision_score(y_test, y_cart, average='micro')
f1 = f1_score(y_test, y_cart, average='micro')
print("Accuracy:", accuracy)
print("Error Rate:", 1.0-accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)



from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_cart)
cm



print(classification_report(y_test,y_id3))