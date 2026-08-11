import numpy as np
import streamlit as st
import pandas as pd

from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


st.write(''' # Clasificación con el dataset Iris ''')
st.image("Iris1.jpg", caption="Iris morada.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  #Pclass = st.number_input('Clase:', min_value=1, max_value=3, value = 1, step = 1)
  #Sex = st.number_input('Género:', min_value=0, max_value=1, value = 0, step = 1)
  #Age = st.number_input('Edad:', min_value=0, max_value=100, value = 0, step = 1)
  #SibSp = st.number_input('Hermanos(as)/Esposo(a):',min_value=0, max_value=10, value = 0, step = 1)
  #Parch = st.number_input('Padres/Hijos:', min_value=0, max_value=10, value = 0, step = 1)
  #Fare = st.number_input('Tarifa:')
  #Embarked = st.number_input('Lugar de Embarque:', min_value=0, max_value=2, value = 0, step = 1)
  sepal_length_cm = st.number_input('Longitud del Zepalo:', min_value=4.3, max_value=7.9, value = 5.1, step = 0.1)
  sepal_width_cm = st.number_input('Ancho del Zepalo:', min_value=2.0, max_value=4.4, value = 3.5, step = 0.1)
  petal_length_cm = st.number_input('Longitud del petalo:', min_value=1.0, max_value=6.9, value = 1.5, step = 0.1)
  petal_width_cm = st.number_input('Ancho del petalo:',min_value=0.1, max_value=2.5, value = 0.5, step = 0.1)
  #user_input_data = {'Pclass': Pclass,
   #                  'Sex': Sex,
    #                 'Age': Age,
     #                'SibSp': SibSp,
      #               'Parch': Parch,
       #              'Fare': Fare,
        #             'Embarked': Embarked}
  user_input_data = {'sepal_length_cm': sepal_length_cm,
                    'sepal_width_cm': sepal_width_cm,
                    'petal_length_cm': petal_length_cm,
                    'petal_width_cm': petal_width_cm,}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()


iris = load_iris()
X = iris.data
Y = iris.target

classifier = DecisionTreeClassifier(max_depth=4, criterion='entropy', min_samples_leaf=5, max_features=4, random_state=42)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción de tipo de iris')
if prediction == 0:
  st.write('Iris tipo setosa')
elif prediction == 1:
  st.write('Iris tipo versicolor')
else:
  st.write('Iris tipo virginica')
