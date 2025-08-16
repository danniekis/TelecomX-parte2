#Importar base de datos y normalizar columnas anidadas#
import pandas as pd
import json

with open('TelecomX_Data.json') as f:
    data = json.load(f)

datos = pd.json_normalize(data, sep='_')
datos

#obteniendo información de las columnas para tratar datos#
datos.info()

#convirtiendo columnas 'Charges_Total' a float64#
datos['account_Charges_Total'] = pd.to_numeric(datos['account_Charges_Total'], errors='coerce')

#quitando datos vacíos en columna 'Churn'#
import numpy as np
datos['Churn'] = datos['Churn'].replace('', np.nan)

#Creando columna cuentas diarias#
datos['Cuentas_Diarias'] = datos['account_Charges_Monthly'] / 30

#Eliminando costumerID
datos = datos.drop(columns=['customerID'])

datos.sample(5)

#Separando base de datos
datos_cleaned = datos.dropna(subset=['Churn'])
X = datos_cleaned.drop('Churn', axis = 1)
y = datos_cleaned['Churn']

#Transformando las variables categoricas con onehotencoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

columnas = X.columns

one_hot = make_column_transformer((OneHotEncoder(drop='if_binary'),
            ['customer_gender','customer_Partner', 'customer_Dependents','phone_PhoneService',
              'phone_MultipleLines', 'internet_InternetService','internet_OnlineSecurity', 'internet_OnlineBackup',
              'internet_DeviceProtection', 'internet_TechSupport','internet_StreamingTV', 'internet_StreamingMovies',
             'account_Contract','account_PaperlessBilling', 'account_PaymentMethod']),
              remainder= 'passthrough', sparse_threshold=0, force_int_remainder_cols=False)

X = one_hot.fit_transform(X)

pd.DataFrame(X, columns=one_hot.get_feature_names_out(columnas))

one_hot.get_feature_names_out(columnas)

#transformando variable respuesta
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5, stratify=y)

#DummyClasiffier
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_test, y_test)

from sklearn.model_selection import KFold, cross_validate

kf_dummy = KFold(n_splits=5, shuffle=True, random_state=5)
cv_dummy = cross_validate(dummy_clf, X, y, cv=kf_dummy, scoring='accuracy')
cv_dummy['test_score']

#RandomForest
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(max_depth=3, random_state=5)
forest_clf.fit(X_train, y_train)
forest_clf.score(X_test, y_test)

kf_forest = KFold(n_splits=5, shuffle=True, random_state=5)
cv_forest = cross_validate(forest_clf, X, y, cv=kf_forest, scoring='accuracy')
cv_forest['test_score']

#DecisionTree
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=5)
tree_clf.fit(X_train, y_train)
tree_clf.score(X_test, y_test)

kf_tree = KFold(n_splits=5, shuffle=True, random_state=5)
cv_tree = cross_validate(tree_clf, X, y, cv=kf_tree, scoring='accuracy')
cv_tree['test_score']

#normalizando datos
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

normalizacion = MinMaxScaler()
X_train_normalizada = normalizacion.fit_transform(X_train_imputed)
pd.DataFrame(X_train_normalizada)

from sklearn.neighbors import KNeighborsClassifier

#KNN
from sklearn.neighbors import KNeighborsClassifier

# Impute missing values in the test set
X_test_imputed = imputer.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_normalizada, y_train)
knn_clf.score(normalizacion.transform(X_test_imputed), y_test)

lista = [('dummy', dummy_clf,X_test),('Arbol',tree_clf,X_test),('KNN',knn_clf,normalizacion.transform(X_test_imputed))]
for i in lista:
   print(f'La exactitud del modelo {i[0]}: {i[1].score(i[2],y_test)}')



param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

kf = KFold(n_splits=5, shuffle=True, random_state=5)

from sklearn.model_selection import GridSearchCV

tree_clf = DecisionTreeClassifier(random_state=5)

grid_search = GridSearchCV(tree_clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_tree_clf = grid_search.best_estimator_
test_accuracy = best_tree_clf.score(X_test, y_test)
print(f"Accuracy of the best Decision Tree Classifier on the test set: {test_accuracy}")

from sklearn.metrics import classification_report

y_pred = best_tree_clf.predict(X_test)
print(classification_report(y_test, y_pred))