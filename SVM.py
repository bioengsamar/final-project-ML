from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd 

def load_data(filename):
    data = pd.read_csv(filename)
    x=data.iloc[:,0:21]
    y=data.iloc[:,21]
    x = x - x.mean() #calculate mean to center the data.
    x = (x - x.min()) / (x.max() - x.min()) #Standard normalization
    x = (x * 2) -1 #scaling between 1, -1
    return x.values, y.values

def model(X_train, X_test, y_train, y_test):
    model = svm.SVC(decision_function_shape='ovo', kernel='poly', degree=6) # SVM for multi-class classification using built-in one-vs-one method
    model.fit(X_train,y_train)
    accuracy= model.score(X_test ,y_test)
    return accuracy


    

if __name__ == "__main__":
    path="fetal_health.csv" #labels= 1-> Normal, 2-> Suspect,  3-> Pathological
    x, y=load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)
    accuracy= model(X_train, X_test, y_train, y_test)
    print('Model accuracy is: ', accuracy) #0.8779342723004695 without and with center data & without normalization, scaling
                                           #0.9295774647887324 with center data and normalization & without scaling
                                           #0.9389671361502347 with center data, normalization & scaling
    
