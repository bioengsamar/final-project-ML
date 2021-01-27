from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
def load_data(filename):
    data = pd.read_csv(filename)
    x=data.iloc[:,0:21]
    y=data.iloc[:,21]
    scaler = preprocessing.StandardScaler().fit(x) #standard scaling
    X_scaled = scaler.transform(x)
    return x.values,X_scaled, y.values

def model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100 ,random_state=50) # number of trees =100
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy= model.score(X_test ,y_test)
    return accuracy,y_pred 



    

if __name__ == "__main__":
    path="fetal_health.csv" #labels= 1-> Normal, 2-> Suspect,  3-> Pathological
    x,x_s, y=load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=5000) 
    accuracy,labels_predected= model(X_train, X_test, y_train, y_test)
    print(f'Random forest model accuracy withh 100 tree (original data) accuracy is:  {round(accuracy*100,2)}') #94.2
    X_train, X_test, y_train, y_test = train_test_split(x_s, y, test_size=0.3,random_state=5000)
    accuracy,labels_predected_for_Scaled_Data= model(X_train, X_test, y_train, y_test)
    print(f'Random forest model accuracy withh 100 tree (standrized data) accuracy is: {round(accuracy*100,2)}') #94.51
    
