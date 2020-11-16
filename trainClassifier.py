import scipy
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import keras
from sklearn.utils import shuffle

# df = pd.read_csv('Data.csv')
# df['cond1'] = np.where((df['Fever'] * df['Tiredness'] * df['Dry-Cough'] == 1), 1, 0)
# col_list= list(df)
# col_list = col_list[0:8]
# print(col_list)
# df['cond2'] = np.where(df[col_list].sum(axis=1) > 5, 1, 0)
# df['cond3'] = np.where(df['Severity_Severe'] == 1, 1, 0)
#
#
# df['covid'] = np.where((df['cond1'] + df['cond2'] + df['cond3'] >0), 1, 0)
# df.to_csv('changed.csv',index=False)

df = pd.read_csv('changed.csv')
columns_to_exclude = [ 'covid', 'cond1', 'cond2', 'cond3']
df_train = df.drop(columns_to_exclude, axis=1)
df_target = df.loc[:, df.columns == 'covid']
X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
X_train,y_train = shuffle(X_train, y_train)
X_test,y_test = shuffle(X_test, y_test)
model = Sequential()
model.add(Dense(4, input_shape=(34,), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
clasifier = model.fit(np.array(X_train), np.array(y_train),validation_split=0.4, epochs=1, batch_size=16, verbose=1)
model.save('classifier_model.h5', clasifier)
# from sklearn.utils import shuffle
#
# from sklearn import tree
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# df = pd.read_csv('Data.csv')
# columns_to_exclude = [ 'covid']
# df_train = df.drop(columns_to_exclude, axis=1)
# df_target = df.loc[:, df.columns == 'covid']
# X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
# X_train,y_train = shuffle(X_train, y_train)
# X_test,y_test = shuffle(X_test, y_test)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
#
# with open('tree.pkl', 'wb') as f:
#     pickle.dump(clf, f)
