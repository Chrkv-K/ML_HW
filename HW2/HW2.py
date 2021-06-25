import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

data = pd.read_csv('C:/Users/ekate/Desktop/HW/HW2/IMDB_Movie_Data.csv', sep=',', decimal=',')
data_clean = data.drop(['Rank', 'Title', "Genre", "Description", "Actors", "Year", "Runtime (Minutes)","Rating", "Votes"], axis=1)
data_clean = data_clean.rename(columns={"Revenue (Millions)": "Revenue"})
data_new = data_clean.dropna()

#попробуем предугадать кассовые сборы по имени режиссера

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data_new["Director"])
df_without_Director = data_new.drop(['Director', 'Metascore'], axis=1)
y = df_without_Director.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

regressor = Ridge(alpha=5) 
regressor.fit(X_train, y_train)
preds = regressor.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))

clf = linear_model.Lasso(alpha=0.07)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))


#попробуем предугадать рейтинг по кассовым сборам

df_Revenue = data_new.drop(['Director', 'Metascore'], axis=1)
df_Metascore = data_new.drop(['Director', 'Revenue'], axis=1)
X = df_Revenue.values
y = df_Metascore.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

regressor = Ridge(alpha=0.03) 
regressor.fit(X_train, y_train)
preds = regressor.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))

clf = linear_model.Lasso(alpha=0.4)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))

clf = linear_model.MultiTaskElasticNet(alpha=0.07)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))