import numpy as np
from pandas.io.stata import stata_epoch
from sklearn import datasets, linear_model, metrics, model_selection
import kagglehub
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib


# Скачиваем последнюю версию датасета
path = kagglehub.dataset_download("blastchar/telco-customer-churn")


# Загружаем файл
df = pd.read_csv(f"{path}/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pd.set_option('display.max_columns', None)
print(df.head())


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

#Разделяем данные
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
categorical_features = ['gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod']

#Создаем ветку для чисел
numerical_transformer = Pipeline(steps=[('input', SimpleImputer(strategy="mean")), ('scaler', StandardScaler())])
#Создае ветку для категорий
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy= 'most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                               ('cat', categorical_transformer, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = Pipeline(steps=[('processor', preprocessor), ('regressor', LogisticRegression(max_iter=10000))])



# Обучаем модель
model.fit(X_train, y_train)

# Предсказываем
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Проверяем accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#Сохранение модели
joblib.dump(model, "churn_model.pkl")
