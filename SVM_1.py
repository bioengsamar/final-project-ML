from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd 

def load_data(filename):
    data = pd.read_csv(filename)
    x=data.iloc[:,0:21]
    y=data.iloc[:,21]
    return x.values, y.values

def model(X_train, X_test, y_train, y_test):
    global model
    model = svm.SVC(decision_function_shape='ovo') # SVM for multi-class classification using built-in one-vs-one method
    model.fit(X_train,y_train)
    accuracy= model.score(X_test ,y_test)
    return accuracy

def predict(x, y_actual):
    yhat = model.predict(x.reshape(1,-1))
    if yhat == y_actual:
        return True, yhat
    else:
        return False, ('y_actual:', y_actual)
    

if __name__ == "__main__":
    path="fetal_health.csv" #lables= 1-> Normal, 2-> Suspect,  3-> Pathological
    x, y=load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)
    accuracy= model(X_train, X_test, y_train, y_test)
    print('Model accuracy is: ', accuracy) #0.8403755868544601
    
    y_pred=predict(X_test[60], y_test[60])
    
    print(y_pred) #(False, ('y_actual:', 2.0))
