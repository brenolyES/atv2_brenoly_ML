#importar as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

@st.cache
def load_data():
    url = "https://gist.githubusercontent.com/brenolyES/6f34e3f24dfe4992ca71c8bdeee4fa3f/raw/a31b3d2acd6db4ae8bcc954f3d7a275f3b35f46a/gistfile1.txt"
    df = pd.read_csv(url, sep=';')
    return df

def main():
    #analisando as primeiras entradas do df
    df = load_data()
    st.dataframe(df.head())

    st.write(df["position"].value_counts())

    st.write(df.groupby('position').mean())

    st.write(df.info())

    st.write(df.describe())

    fig, ax = plt.subplots()
    df.hist(bins=50, figsize=(20,15), ax=ax)
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # Selecionar os recursos de entrada (X) e a variável alvo (y)
    X = df[['age', 'hits', 'potential']]
    y = df['overall']

    # Definir o número de folds para a validação cruzada k-fold
    n_folds = 5

    # Inicializar o objeto de validação cruzada k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Inicializar o modelo de regressão linear
    model = LinearRegression()

    # Realizar a validação cruzada k-fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        st.write(f'Fold {fold+1}:')