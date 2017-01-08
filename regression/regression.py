from sklearn import linear_model #linearmodel for regression
clf = linear_model.Lasso(alpha=0.1) #lasso is a type of regression algorithm
clf.fit([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],[2,4,6,8,10,12,14,16]) #fit the line using this training input
print(clf.predict([10,10])) #predict the output for unseen values