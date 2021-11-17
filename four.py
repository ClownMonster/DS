from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
import pandas as pd

df = pd.read_csv('C:/Users/gayat/Downloads/iris.csv')
df.head()

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0,test_size=0.25)
classifier = KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')

classifier.fit(X_train,y_train)

#predict the test resuts 
y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred) 
print('Confusion matrix is as follows\n',cm) 
print('Accuracy Metrics') 
print(classification_report(y_test,y_pred))
print(" correct predicition",accuracy_score(y_test,y_pred)) 
print(" wrong predicition",(1-accuracy_score(y_test,y_pred)))
