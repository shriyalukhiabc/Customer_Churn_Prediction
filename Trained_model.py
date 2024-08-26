import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data=pd.read_csv('C:\\Users\\shriy\\PycharmProjects\\customer_churn_prediction\\synthetic_customer_data.csv')
print(data)

data=data.drop(['CustomerID','TechSupport','PaperlessBilling','ContractType','InternetService','Gender','PaymentMethod'],axis=1)

print(data.head())

le=preprocessing.LabelEncoder()
data['Churn']=le.fit_transform(data['Churn'])
print(data.head())

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

with open("classifier.pkl","wb") as file:
    pickle.dump(classifier,file)

accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy}\n")

precision=precision_score(y_test,y_pred,average='weighted')
print(f"precision:{precision}\n")

recall=recall_score(y_test,y_pred,average='weighted')
print(f"recall:{recall}\n")

cm=confusion_matrix(y_test,y_pred)
print(f"Confusion matrix:\n{cm}\n")

