#load dataset
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeClassifier
iris_dataset=load_iris()
X=iris_dataset['data']
y=iris_dataset['target']
in_sample_accuracy=[]
out_of_sample_accuracy=[]
for i in range(1,11):
    #print(i)
    tree=DecisionTreeClassifier(max_depth=4)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y)
    tree.fit(X_train,y_train)
    in_sample_accuracy.append(tree.score(X_train,y_train))
    out_of_sample_accuracy.append(tree.score(X_test,y_test))


ind=list(range(1,11))
ind.append('mean')
ind.append('std')
in_sample_accuracy.append(np.mean(in_sample_accuracy))
in_sample_accuracy.append(np.std(in_sample_accuracy[:-1]))
out_of_sample_accuracy.append(np.mean(out_of_sample_accuracy))
out_of_sample_accuracy.append(np.std(out_of_sample_accuracy[:-1]))
accuracy=pd.DataFrame([in_sample_accuracy,out_of_sample_accuracy,],
                        columns=ind,
                        index=['in sample ','out of sample '])
pd.set_option('precision',3)
accuracy





X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y)

CVS=[]
scores=cross_val_score(DecisionTreeClassifier(max_depth=4),X_train,y_train,cv=10)
print(scores)
CVS.append(scores)
pd.set_option('precision',3)
result=pd.DataFrame(CVS,columns=list(range(1,11)),)
result['mean']=result.mean(1)
result['std']=result.std(1)
## run the DecisionTree
dt=DecisionTreeClassifier(max_depth=4)
dt.fit(X_train,y_train)
result['Out-of-sample-accuracy']=dt.score(X_test,y_test)
result


print("My name is Yuchen Duan")
print("My NetID is: yuchend3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
