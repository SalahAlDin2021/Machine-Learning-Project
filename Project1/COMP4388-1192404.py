import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from matplotlib import pylab
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score,mean_squared_error,confusion_matrix

##Please uncomment the task and run to show the result of task
#read the Bejaia Region Dataset(from row 1 read 122 line)
dataSet1=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=1,nrows=122)
#read the Bejaia Region Dataset(from row 126 read 122 line)
dataSet2=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=126,nrows=122)
#Get the attributes of data set (day, month, year, Temp, HR, Ws,....  )
l=dataSet1.columns
'''replace the fire to 1 and not fire to 0 in the last column(Classes) 
to get the mean and Standard Deviation of fire and not fire(mean, 
max,1st&2nd&3rdQuartile are not important in this feature)'''
# dataSet1[l[13]].replace(['not fire','fire'],[0,1],inplace=True)
# dataSet2[l[13]].replace(['not fire','fire'],[0,1],inplace=True)

# #task1
# '''Print the Summary of coulmns from 3 to 12 of 2 data sets ,because the first 3 coulmns
# is day,month,year and the summary of this features doesn’t useful'''
# print("Bejaia Region Dataset")
print(dataSet1[l].describe().to_string());
print("222")
# print("Sidi-Bel Abbes Region Dataset")
# print(dataSet1[l[3:14]].describe().to_string());

# #task2
# splitedDataset  = [rows for _, rows in dataSet1.groupby(l[13])]
# plt.title('Density Plot Of Temperatures in Bejaia Region Dataset')
# plt.ylabel('Temperature')
# plt.xlabel('Density')
# plt.plot(splitedDataset [0][l[3]],label = "fire")
# plt.plot(splitedDataset [1][l[3]],label = "not fire")
# plt.legend()
# plt.show()
# splitedDataset  = [rows for _, rows in dataSet2.groupby(l[13])]
# plt.title('Density Plot Of Temperatures in Sidi-Bel Abbes Region Dataset')
# plt.ylabel('Temperature')
# plt.xlabel('Density')
# plt.plot(splitedDataset[0][l[3]],label = "fire")
# plt.plot(splitedDataset [1][l[3]],label = "not fire")
# plt.legend()
# plt.show()

# #task3
# Var_Corr = dataSet1[l[3:13]].corr()
# sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
# fig = pylab.gcf()
# fig.canvas.manager.set_window_title('correlation of Bejaia Region Dataset')
# plt.show()
# # i use this line to check where the error (becouse that doesnt get the feature DC and FWI)
# # and i find that missing ',' between this 2 values (14.6 9) and i edit it
# dataSet2[l[9]]=dataSet2[l[9]].astype(float)
#
# Var_Corr = dataSet2[l[3:13]].corr()
# sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
# fig = pylab.gcf()
# fig.canvas.manager.set_window_title('correlation of Sidi-Bel Abbes Region Dataset')
# plt.show()

# #task4
# dependentFeature =dataSet1[l[13]]
# dependentFeature.columns=['Classes']
# Var_Corr = dataSet1[l[3:12]].corrwith(dependentFeature)
# sns.heatmap(Var_Corr[:,np.newaxis], xticklabels=dependentFeature.columns, yticklabels=dataSet1[l[3:12]].columns, annot=True)
# fig = pylab.gcf()
# fig.canvas.manager.set_window_title("correlation between independent and dependent('classes') features")
# plt.show()



# #task5.2
# x=np.array(dataSet1[l[10]])
# y=np.array(dataSet1[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# #reshape data from 1D Array to 2D Array
# x_train= x_train.reshape(-1, 1);y_train= y_train.reshape(-1, 1);
# x_test = x_test.reshape(-1, 1);y_test=y_test.reshape(-1,1)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Bejaia Region Dataset")
# print("accurecy: ",acc)
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))

# print("------------------------------------------------")
# x=np.array(dataSet2[l[10]])
# y=np.array(dataSet2[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# #reshape data from 1D Array to 2D Array
# x_train= x_train.reshape(-1, 1);y_train= y_train.reshape(-1, 1);
# x_test = x_test.reshape(-1, 1);y_test=y_test.reshape(-1,1)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Sidi-Bel Abbes Region Dataset")
# print("accurecy: "+str(acc))
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))




# #task5.3
# x=np.array(dataSet1[l[10]],dataSet1[l[8]])
# y=np.array(dataSet1[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# #reshape data from 1D Array to 2D Array
# x_train= x_train.reshape(-1, 1);y_train= y_train.reshape(-1, 1);
# x_test = x_test.reshape(-1, 1);y_test=y_test.reshape(-1,1)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Bejaia Region Dataset")
# print("accurecy: ",acc)
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))

# print("------------------------------------------------")
# x=np.array(dataSet2[l[10]])
# y=np.array(dataSet2[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# #reshape data from 1D Array to 2D Array
# x_train= x_train.reshape(-1, 1);y_train= y_train.reshape(-1, 1);
# x_test = x_test.reshape(-1, 1);y_test=y_test.reshape(-1,1)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Sidi-Bel Abbes Region Dataset")
# print("accurecy: "+str(acc))
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))





# # task5.4
# x=np.array(dataSet1[l[3:11]])
# y=np.array(dataSet1[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Bejaia Region Dataset")
# print("accurecy: ",acc)
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))
#
# print("----------------------------------------")
# x=np.array(dataSet2[l[3:11]])
# y=np.array(dataSet2[l[12]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# linear=linear_model.LinearRegression()
# linear.fit(x_train,y_train)
# acc=linear.score(x_test,y_test)
# y_predict=linear.predict(x_test)
# print("Sidi-Bel Abbes Region Dataset")
# print("accurecy: ",acc)
# print("coeff",linear.coef_)
# print("Mean Square Error: ",mean_squared_error(y_test,y_predict))
# print("r2_score: ",r2_score(y_test,y_predict))


# # task6
# x=np.array(dataSet1[l[3:12]])
# y=np.array(dataSet1[l[13]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# linear=linear_model.LogisticRegression()
# linear.fit(x_train,y_train)
# y_predict=linear.predict(x_test)
# print("Bejaia Region Dataset")
# print("confusion matrix: ",confusion_matrix(y_test,y_predict))
# print("accurecy: ",linear.score(x_test,y_test))
# print("precision: ",precision_score(y_test,y_predict))
# print("recall: ",recall_score(y_test,y_predict))
# print("----------------------------------------")
# x=np.array(dataSet2[l[3:12]])
# y=np.array(dataSet2[l[13]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# logistic=linear_model.LogisticRegression()
# logistic.fit(x_train,y_train)
# y_predict=linear.predict(x_test)
# print("Sidi-Bel Abbes Region Dataset")
# print("confusion matrix: ",confusion_matrix(y_test,y_predict))
# print("accurecy: ",linear.score(x_test,y_test))
# print("precision: ",precision_score(y_test,y_predict))
# print("recall: ",recall_score(y_test,y_predict))


# # task7
# x=np.array(dataSet1[l[3:12]])
# y=np.array(dataSet1[l[13]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train)
# y_predict=knn.predict(x_test)
# print("Bejaia Region Dataset")
# print("confusion matrix: ",confusion_matrix(y_test,y_predict))
# print("accurecy: ",knn.score(x_test,y_test))
# print("precision: ",precision_score(y_test,y_predict))
# print("recall: ",recall_score(y_test,y_predict))
# print("----------------------------------------")
# x=np.array(dataSet2[l[3:12]])
# y=np.array(dataSet2[l[13]])
# x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train)
# y_predict=knn.predict(x_test)
# print("Sidi-Bel Abbes Region Dataset")
# print("confusion matrix: ",confusion_matrix(y_test,y_predict))
# print("accurecy: ",accuracy_score(y_test,y_predict))
# print("precision: ",precision_score(y_test,y_predict))
# print("recall: ",recall_score(y_test,y_predict))
