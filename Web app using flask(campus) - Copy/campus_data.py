import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"C:\Z-FILES\DS-ML\Web app using flask(campus)\Placement_Data_Full_Class.csv")
data1 = pd.read_csv(r"C:\Z-FILES\DS-ML\Web app using flask(campus)\Placement_Data_Full_Class.csv")

#KNN -- 2,12 imp features
#poly_reg -- 2,4,6,7,10,12 imp features

#KNN Model
X = data.drop(columns=['sl_no','gender','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','status','salary'])
Y = data['status']



labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)

ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
ohe.fit(X[['ssc_p','mba_p']])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)


#pipeline_KNN 

col_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['ssc_p','mba_p']),remainder='passthrough')
classifier = KNeighborsClassifier(weights='distance')
pipe = make_pipeline(col_trans,classifier)
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print('Accuracy score of KNN model : ',"{:.2f}".format(accuracy_score(y_test,y_pred)))
pickle.dump(pipe,open("campus_data(KNN).pkl","wb"))
print(pipe.predict(pd.DataFrame([[67,58.8]], columns=['ssc_p','mba_p'])))



#Regression model

#removing null values
data1 = data1.dropna()
X1 = data1.drop(columns=['sl_no','gender','ssc_b','hsc_b','degree_t','workex','specialisation','status','salary'])
Y1 = data1['salary']

ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
ohe.fit(X1[['ssc_p','hsc_p','hsc_s','degree_p','etest_p','mba_p']])

x1_train,x1_test,y1_train,y1_test = train_test_split(X1,Y1,test_size=0.20,random_state=42)


#pipeline_regression


lin_reg_2 = LinearRegression()


col_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['ssc_p','hsc_p','hsc_s','degree_p','etest_p','mba_p']),remainder='passthrough')
pipe = make_pipeline(col_trans,lin_reg_2)
pipe.fit(x1_train,y1_train)
y1_pred = pipe.predict(x1_test)

print("R2 Score : ", "{:.2f}".format(r2_score(y1_test,y1_pred)))
pickle.dump(pipe, open("campus_data(linear_reg).pkl","wb"))
print(pipe.predict(pd.DataFrame([[67,91,'Commerce',58,55,58.8]], columns=['ssc_p','hsc_p','hsc_s','degree_p','etest_p','mba_p'])))
