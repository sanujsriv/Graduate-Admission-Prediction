import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error,mean_squared_error,roc_curve,roc_auc_score,auc
from sklearn.metrics import accuracy_score,r2_score
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer
from sklearn.externals import joblib 
import sklearn
print()

############################################################
# Supressing Sklearn Future Warnings 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
###########################################################

###################  load data ################################

dataset= pd.read_csv('DATA\graduateAdmissions\Admission_Predict.csv',na_values=['.'])

#print(dataset.head())
#print(dataset.info())
#pd.set_option('display.max_columns', None)
#print(dataset.describe())

#################### rename columns ############################

dataset=dataset.rename(columns = {'Chance of Admit ':'Chance of Admit'})
dataset=dataset.rename(columns = {'Serial No.':'SN'})
dataset=dataset.rename(columns = {'SOP Rating':'SOP'})
dataset=dataset.rename(columns = {'LOR ':'LOR'})
dataset=dataset.rename(columns = {'GRE Score':'GRE'})
dataset=dataset.rename(columns = {'TOEFL Score':'TOEFL'})
dataset=dataset.rename(columns = {'University Rating':'u_rating'})
# print(dataset.info())
# print(dataset.head())
# print(dataset.describe())
column_dataset=[d for d in dataset.columns]
#print(column_dataset)

########## Splitting Training & Testing Data: Holdout #############

split = StratifiedShuffleSplit(n_splits = 1,test_size =0.4, random_state =42)
for train_index,test_index in split.split(dataset,dataset["u_rating"],dataset["SOP"]):
    xtrain=dataset.loc[train_index]
    xtest=dataset.loc[test_index]
    
#print((xtrain['SOP'].value_counts())/(len(xtrain)), "\n")
#print((dataset['SOP'].value_counts())/(len(dataset)))

################## creat a copy of the train data ################

data_train = xtrain.copy()

################ correlation with chance of admit ############################

corr_matrix = data_train.corr()

corr_target = corr_matrix["Chance of Admit"].sort_values(ascending = False)
#print(corr_target)
pd.plotting.scatter_matrix(data_train[column_dataset], figsize= (18,12))
plt.savefig('correlaion_with_thetarget.png')
data_train.plot(kind ="scatter", x = "CGPA", y ="Chance of Admit",alpha =0.1)
plt.savefig('correlaion_between_CGPA_AND_thetarget.png')

####################### create a copy eith out the target ################

data_noid = data_train.drop(columns = ['Chance of Admit'])

################### feature scaling ##########################

scaler = MinMaxScaler()
fit_scaler = scaler.fit(data_noid)
#print(fit_scaler)
scaler_transform  = scaler.transform(data_noid)
dataset = pd.DataFrame({'SN': data_train["SN"], 'GRE': scaler_transform[:, 0], "TOFEL": scaler_transform[:, 1], "u_rating" : scaler_transform[:, 2], "SOP": scaler_transform[:, 3], "LOR":scaler_transform[:, 4], "CGPA": scaler_transform[:, 5], "Research": scaler_transform[:, 6], "Change of Admit": scaler_transform[:, 7]})

################### Visualization: Plotting histograms ###########################

dataset.hist(bins =10,figsize =(15,10))
plt.savefig('historgram_after_scaling.png')
#plt.show()


################## creating xtrain,ytrain,xtrain,xtest #################### 

data_train = xtrain.copy()
data_test = xtest.copy()

xtrain=data_train.drop(['Chance of Admit'],axis=1)
ytrain=data_train['Chance of Admit']

xtest=data_test.drop(['Chance of Admit'],axis=1)
ytest=data_test['Chance of Admit']

## ######### Categorizing Continous(Chance Of Prediction) ###########

ylabel_train=[1 if yt>=0.75 else 0 for yt in ytrain]
ylabel_test=[1 if yl>=0.75 else 0 for yl in ytest]

## ###################### LINEAR REGRESSION ###########################
#the model is saved in the folder models umder the name "LinearRegression.pkl"

time_LinReg=[]
for i in range(30):
  tstart=time.process_time()
  reg=LinearRegression()
  Lreg_fit=reg.fit(xtrain,ylabel_train)
  reg_prediction=reg.predict(xtest)
  ## print(r2_score(ytest,reg_prediction))
  tstop=time.process_time()
  time_LinReg.append((tstop-tstart))
reg_pred_acc=[1 if rg>=0.925 else 0 for rg in reg_prediction]
#
#joblib.dump(reg,"models/LinearRegression.pkl")
## print(accuracy_score(reg_pred_acc,ylabel_test))
#
#figure=plt.figure()
#c=list(range(1,81))
#plt.plot(c,ytest, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,reg_prediction, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_LINEAR REGRESSION .png')
#
#conf_mt=confusion_matrix(ylabel_test,reg_prediction)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_LinearRegression.png')
##

## #################### DECISION TREE REGRESSOR ##########################
#the model is saved in the folder models under the name "DecisionTreeRegressor.pkl"
#
time_DTReg=[]
for i in range(30):
  tstart=time.process_time()
  dtree_reg=DecisionTreeRegressor()
  dtree_reg.fit(xtrain,ylabel_train)
  dtree_reg_prediction=dtree_reg.predict(xtest)
  tstop=time.process_time()
  time_DTReg.append((tstop-tstart))
#
#joblib.dump(dtree_reg,"models/DecisionTreeRegressor.pkl")
#print('Time Elasped: DT Regression -'+str(tstop-tstart))
#dtree_reg_acc=[1 if dt>=0.75 else 0 for dt in dtree_reg_prediction]
#print(dtree_reg_acc)
#print(dtree_reg_prediction)
#print(accuracy_score(dtree_reg_prediction,ylabel_test))
#print(ytest.tolist())
#
#figure=plt.figure()
#c=list(range(1,81))
#plt.plot(c,ytest, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,dtree_reg_prediction, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_DECISION TREE REGRESSOR.png')
#
#conf_mt=confusion_matrix(ylabel_test,dtree_reg_prediction)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_DecisionTreeRegressor.png')
###


## ##################### DECISION TREE ###################################
#the model is saved in the folder models under the name "DecisionTreeClassifier.pkl"
#
time_DTClf=[]
for i in range(30):
  tstart=time.process_time()
  dec_tree=DecisionTreeClassifier()
  dec_tree_fit=dec_tree.fit(xtrain,ylabel_train)
  dec_tree_prediction=dec_tree.predict(xtest)
  tstop=time.process_time()
  time_DTClf.append((tstop-tstart))
#  
#joblib.dump(dec_tree,"models/DecisionTreeClassifier.pkl")
#dec_tree_acc=[1 if dt>=0.75 else 0 for dt in dec_tree_prediction]
#
#
#print('Time Elasped: Linear Regression -'+str(tstop-tstart))
#print(dec_tree_prediction)
#print(accuracy_score(ylabel_test,dec_tree_prediction))
#print()  
#print(confusion_matrix(ylabel_test,dec_tree_prediction))
#print()
#print(classification_report(ylabel_test,dec_tree_prediction))
#
#print('Mean Absolute Error:',mean_absolute_error(ylabel_test, dec_tree_prediction))
#print('Mean Squared Error:', mean_squared_error(ylabel_test, dec_tree_prediction))
#print('Root Mean Squared Error:', np.sqrt(mean_squared_error(ylabel_test, dec_tree_prediction)))
#
#figure=plt.figure()
#c=list(range(1,81))
#plt.plot(c,ytest, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,dec_tree_prediction, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_DecisionTreeClassifier.png')
#
#conf_mt=confusion_matrix(ylabel_test,dec_tree_prediction)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_DecisionTreeClassifier.png')

## ########################## KNN #######################################
#the model is saved in the folder models under the name "knn.pkl"
#
time_KNN=[]
for i in range(30):
  tstart=time.process_time()
  knn= KNeighborsClassifier(n_neighbors=8,metric='euclidean') 
  ## Because '8' has max accuracy ## See ---Best_K
  knn_fit=knn.fit(xtrain,ylabel_train)
  prediction_knn=knn.predict(xtest)
  tstop=time.process_time()
  time_KNN.append((tstop-tstart))
#joblib.dump(knn,"models/knn.pkl")
#knn_acc=[1 if dt>=0.75 else 0 for dt in prediction_knn]
#
#figure=plt.figure()
#c=list(range(1,81))
#plt.plot(c,ytest, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,prediction_knn, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_KNN.png')
#
#
#
#print('Time Elasped: KNN -'+str(tstop-tstart))
#print(accuracy_score(ylabel_test, prediction_knn))
#print(prediction_knn)
#print(conf_mt)
#print(classification_report(ylabel_test,prediction_knn))
#sb.set(font_scale=1.8)
#conf_mt=confusion_matrix(ylabel_test,prediction_knn)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_KNN.png')

# ####---- Best_K
#acc_scores = []
#for acc in range(1,50):
#    knn_n = KNeighborsClassifier(n_neighbors = acc)
#    knn_n.fit(xtrain, ylabel_train)
#    acc_scores.append(knn_n.score(xtest, ylabel_test))
#
#fig,axes_knn=plt.subplots(1,1,figsize=(9,6))
#plt.plot(range(1,50),acc_scores)
#plt.savefig('acc_score.png')
#
#
## # ##https://stackoverflow.com/questions/6282058/writing-numerical-values-on-the-plot-with-matplotlib
#
#for i,j in zip(range(1,50),acc_scores):
#     axes_knn.annotate(str(i)+','+str(j),xy=(i,j))
#plt.xlabel("k")
#plt.ylabel("accuracy")
#plt.show()
#plt.savefig('knn_best_K.png')


## ################### Logistic Regression ###########################
##the model is saved in the folder models under the name "Logistic Regression.pkl"

time_logreg=[]
for i in range(30):
  tstart=time.process_time()
  log_regres=LogisticRegression()
  log_reg_fit=log_regres.fit(xtrain,ylabel_train)
  log_predict=log_regres.predict(xtest)
  tstop=time.process_time()
  time_logreg.append((tstop-tstart))
  
#joblib.dump(log_regres,"models/Logistic_Regression.pkl")
#
#figure=plt.figure()
#c=list(range(1,81))
#
#plt.plot(c,ylabel_test, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,log_predict, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_Logistic_Regression.png')
#
#conf_mt=confusion_matrix(ylabel_test,log_predict)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_Logistic_Regression.png')


# print('Time Elasped: Logistic Regression -'+str(tstop-tstart))
## print(log_predict)
## print(accuracy_score(ylabel_test,log_predict))
## print(confusion_matrix(ylabel_test,log_predict))
## print(classification_report(ylabel_test,log_predict))
#
## ######################## PCA ######################################
#
## # pca=PCA(n_components=2, whiten=2)
## # dataset_pca=pca.fit_transform(x)
## # dataset_pca=pd.DataFrame(data=dataset_pca,columns=['PC1','PC2'])
## # dataset_pca_final=pd.concat([dataset_pca,y], axis=1)
## # print(dataset_pca_final)
#
## #######################################################################
#
#
#
######################## Random Forests ##################################
#yl\+0100.
 

time_Random=[]
for i in range(30):
  tstart=time.process_time()
  rand_for_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=17, max_features=2, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=60,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
  rand_for_fit=rand_for_clf.fit(xtrain, ylabel_train)
  rand_for_predict = rand_for_clf.predict(xtest)
  tstop=time.process_time()
  time_Random.append((tstop-tstart))
  
#final_mse = sklearn.metrics.mean_squared_error(ylabel_test,rand_for_predict)
#final_rmse = np.sqrt(final_mse)
#print("rmse",final_rmse )

#
joblib.dump(rand_for_clf,"models/RandomForest.pkl")
#
#figure=plt.figure()
#c=list(range(1,81))
#
#plt.plot(c,ylabel_test, color = 'red', linewidth = 1, label='Test')
#plt.plot(c,rand_for_predict, color = 'blue', linewidth = 1, label='Predicted')
#plt.xlabel('Testing record _#_')
#plt.ylabel('Chances')
#plt.legend() # Plot legend for notifying testing and training data
##
#plt.grid(alpha = 0.5) # Grid Visibility
#figure.suptitle('Actual vs Predicted') # Title of Graph
#plt.savefig('Actual_vs_predicted_Random_Forests.png')
#
#conf_mt=confusion_matrix(ylabel_test,rand_for_predict)
#f, ax = plt.subplots(figsize=(5, 5))
#conf_map=sb.heatmap(conf_mt,annot=True,linewidths=0.25,cmap='coolwarm', linecolor='white',fmt="d",ax=ax,cbar=False,annot_kws={'ha':"left", 'va':"top"},xticklabels=True)
#conf_fig=conf_map.get_figure()
#conf_fig.savefig('conf_mt_Random_Forests.png')

## forest_score = accuracy_score(rand_for_predict,ylabel_test)
#
#
## print(forest_score)
#
############################### SVM #####################################
#
##################### SVR ####################
#
xtrain_svr=xtrain[['GRE','TOEFL','SOP','LOR','CGPA']]
xtest_svr=xtest[['GRE','TOEFL','SOP','LOR','CGPA']]
time_SVR=[]
for i in range(30):
  tstart=time.process_time()
  svr = SVR()
  svr.fit(xtrain_svr,ylabel_train)
  pred_svr = svr.predict(xtest_svr)
  tstop=time.process_time()
  time_SVR.append((tstop-tstart))
# print('Time Elasped: SVR -'+str(tstop-tstart))
pred_svr_acc=[1 if sr>=0.75 else 0 for sr in pred_svr]
#joblib.dump(svr,"models/svr.pkl")
# print(accuracy_score(ylabel_test,pred_svr_acc))
#
## ################### SVC #####################
#
time_SVC=[]
for i in range(30):
  tstart=time.process_time()
  svc=SVC()
  svc_fit=svc.fit(xtrain,ylabel_train)
  pred_svc=svc.predict(xtest)
  tstop=time.process_time()
  time_SVC.append((tstop-tstart))
  
#joblib.dump(svc,"models/SVC.pkl")


################## Running Time ##############################
time_all=[sum(time_LinReg)/len(time_LinReg),
          sum(time_DTClf)/len(time_DTClf),
          sum(time_DTReg)/30,
          sum(time_KNN)/30,
          sum(time_logreg)/30,
          sum(time_Random)/30,
          sum(time_SVR)/30,
          sum(time_SVC)/30]
print(time_all)
#####################################################################

###################### ROC CURVE For various _Models_ ################################ 
#
models_used=[rand_for_fit,log_reg_fit,knn_fit,dec_tree_fit]
model_names=['Random Forests','Logistic Regression','KNN','Decision Tree']
i=0

for m in models_used:
 probs = m.predict_proba(xtest)
 preds = probs[:,1]
 fpr, tpr, threshold = roc_curve(ylabel_test, preds)
 roc_aucScore = roc_auc_score(ylabel_test,preds)
 roc_auc = auc(fpr,tpr)  
 ## Yields same output ##
 f,ax=plt.subplots(1,1)
 plt.plot(fpr, tpr, 'b--', label = 'AUC = %0.2f' % roc_aucScore)
 plt.legend(loc = 'lower right')
 plt.title('ROC : '+model_names[i])
 plt.plot([0, 1], [0, 1],'r--')
 plt.xlim([0, 1])
 plt.ylim([0, 1])
 plt.ylabel('TPR')
 plt.xlabel('FPR')
 plt.savefig('ROC_score_'+model_names[i]+'.png')
 i=i+1




################### Comparing the Algorithms ############################
#
algo_list=['LReg','LoReg','Dtree','DtreeReg','KNN','RandFor','SVR','SVC']

accuracy_comp=np.array([accuracy_score(reg_pred_acc,ylabel_test),accuracy_score(log_predict,ylabel_test),accuracy_score(dec_tree_prediction,ylabel_test),accuracy_score(dtree_reg_prediction,ylabel_test),accuracy_score(prediction_knn,ylabel_test),accuracy_score(rand_for_predict,ylabel_test),accuracy_score(pred_svr_acc,ylabel_test),accuracy_score(pred_svc,ylabel_test)])

f,axes=plt.subplots(1,1,figsize=(6,6))
bar_cmp=sb.barplot(algo_list,accuracy_comp,color='Orange')

# Adding annotation to barplot : Refer StackOverFlow ||:https://stackoverflow.com/questions/45946970/displaying-of-values-on-barchart
axes=bar_cmp
for p in ax.patches:
  axes.annotate(p.get_height(), (p.get_x() + p.get_width() / 3.,p.get_height()),ha='center', va='center', fontsize=10, color='gray', xytext=(0, 20),textcoords='offset points')
_ = bar_cmp.set_ylim(0,1.0) 
plt.xlabel('Accuracy Scores')
plt.ylabel('Algorithms')
plt.title('Best Algorithm')
plt.savefig('Comparing_Algorithm.png')

