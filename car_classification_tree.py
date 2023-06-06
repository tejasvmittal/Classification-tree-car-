import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# reading data using pandas into a dataframe
# dataframe is like a spreadsheet
df = pd.read_csv("car_evaluation/car.data", header=None)
df.columns = ["buying", "maint", "doors",
              "persons", "lug_boot", "safety", "rating"]

# missing data can be checked by printing unique values for each column

x = df.drop("rating", axis=1).copy()
y = df["rating"].copy()

# formatting y values
y[y == "unacc"] = 0
y[(y == "acc") | (y == "vgood") | (y == "good")] = 1
y = y.astype("int")

# using one hot encoding on all the variables that are categorical.
x_encoded = pd.get_dummies(
    x, columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])


# this data can now be used to create classification trees.

x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(x_train, y_train)
# plt.figure(figsize=(15,7.5))
# plot_tree(clf, filled=True, rounded=True, class_names=["Unacc","Acc"], feature_names=x_encoded.columns)


y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm, display_labels=["Unacc", "Acc"]).plot()
# plt.show()
# As seen by the confusion matrix, our predictions are very accurate already with false negative rate being 1.47% and
# false positive rate of 0%

path = clf.cost_complexity_pruning_path(x_train, y_train) # get all compatible ccp alpha values for our tree. 
alphas = path.ccp_alphas
cross_validation_scores = []

for alpha in alphas:
    temp_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    scores = cross_val_score(temp_clf, x_train, y_train, cv=5)
    cross_validation_scores.append([alpha, np.mean(scores), np.std(scores)])

results = pd.DataFrame(cross_validation_scores, columns=["Alpha", "Mean score", "Std score"])
# results.plot(x="Alpha", y="Mean score", yerr="Std score", marker='o', linestyle='--')
# plt.show()

ideal_alpha = results[(results["Alpha"]>0.0007) & (results["Alpha"]<0.001)]
ideal_alpha = float(ideal_alpha["Alpha"]) #obtained the best alpha value as a float.

#now we have everything to complete our pruned classification tree.
clf_final = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_alpha)
clf_final = clf_final.fit(x_train, y_train)
plt.figure(figsize=(15, 7.5))
plot_tree(clf, filled=True, rounded=True, class_names=["Unacc","Acc"], feature_names=x_encoded.columns)
plt.show()

# y_pred = clf_final.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm, display_labels=["Unacc", "Acc"]).plot()
# plt.show()

#The new confusion matrix results in a lowered false negative rate of 0.7299% but increased the false positive rate to 0.3413% 
