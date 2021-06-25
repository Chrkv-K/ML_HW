import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report

data = pd.read_csv('C:/Users/ekate/Desktop/HW/HW1/news_fake_n_true.csv', sep=',', decimal=',')
data_clean = data.drop(['Unnamed: 0','title','subject','date'], axis=1)
data = data_clean.dropna()
count = Counter(" ".join(data[data['label'] == 0]["text"]).lower().split()).most_common(50)
df_0 = pd.DataFrame.from_dict(count)
df_0 = df_0.rename(columns={0: "words in true", 1 : "count"})
count = Counter(" ".join(data[data['label'] == 1]["text"]).lower().split()).most_common(50)
df_1 = pd.DataFrame.from_dict(count)
df_1 = df_1.rename(columns={0: "words in fake", 1 : "count"})
vectorizer = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(data["text"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['label'], test_size=0.2, random_state=50)


                                ###     naive_bayes     ###

# list_alpha = np.arange(1/100000, 20, 0.11) 
# score_train = np.zeros(len(list_alpha))
# score_test = np.zeros(len(list_alpha))
# recall_test = np.zeros(len(list_alpha))
# precision_test= np.zeros(len(list_alpha))
# count = 0

# for alpha in list_alpha:
#     bayes = naive_bayes.MultinomialNB(alpha=alpha)
#     bayes.fit(X_train, y_train)   
#     score_train[count] = bayes.score(X_train, y_train)
#     score_test[count]= bayes.score(X_test, y_test)
#     recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
#     precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
#     count = count + 1

# matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
# models = pd.DataFrame(data = matrix, columns = ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
# best_index = models['Test Precision'].idxmax()
# print(models.iloc[best_index])

                                ###     SVM     ###

# list_C = np.arange(500, 600, 100) #100000

# score_train = np.zeros(len(list_C))
# score_test = np.zeros(len(list_C))
# recall_test = np.zeros(len(list_C))
# precision_test= np.zeros(len(list_C))
# count = 0
# for C in list_C:
#     svc = svm.SVC(C=C)
#     svc.fit(X_train, y_train)
#     score_train[count] = svc.score(X_train, y_train)
#     score_test[count]= svc.score(X_test, y_test)
#     recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
#     precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
#     count = count + 1

# matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
# models = pd.DataFrame(data = matrix, columns = ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
# best_index = models['Test Precision'].idxmax()

# print(models.iloc[best_index, :])


                            ###     Boosting     ###

gb_clf_es = GradientBoostingClassifier(n_iter_no_change=5, verbose=10)
gb_clf_es.fit(X_train, y_train)
es_y_pred = gb_clf_es.predict(X_test)
print(classification_report(y_test, es_y_pred))

                            ###     Decision Tree     ###

# tree_clf = DecisionTreeClassifier()
# tree_clf.fit(X_train, y_train)
# tree_y_pred = tree_clf.predict(X_test)
# print(classification_report(y_test, tree_y_pred))

                            ###     Random Forest     ###

# forest = RandomForestClassifier(n_estimators=500, verbose=1)
# forest.fit(X_train, y_train)
# forest_y_pred = forest.predict(X_test)
# print(classification_report(y_test, forest_y_pred))