import pandas as pd
import streamlit as st
import requests
import shap
from streamlit_shap import st_shap
import joblib
import plotly.graph_objects as go
import pytest

from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Bank Dashboard",
    page_icon="💶",
)

def formatter(value) :
    if (value == 0.875) :
        return 'Grade A'
    elif (value == 0.350) :
        return 'Grade B'
    elif (value == 0.300) :
        return 'Grade C'
    elif (value == 0.250) :
        return 'Grade D'
    return ''

def graph(value):
    options = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Score",
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "center": ['50%', '75%'],
                "radius": '90%',
                "min": 0,
                "max": 100,
                "splitNumber": 8,
                "axisLine": {
                    "lineStyle": {
                        "width": 10,
                    
                        "color": [
                            [0.25, '#229954'],
                            [0.30, '#32A826'],
                            [0.35, '#C02B2B'],
                            [1, '#781B1B']
                        ]
                    },
                },
                "pointer": {
                    "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                    "length": '12%',
                    "width": 20,
                    "offsetCenter": [0, '-60%'],
                    "itemStyle": {
                        "color": 'inherit'
                    }
                },
                "axisTick": {
                    "length": 0,
                    "lineStyle": {
                        "color": 'inherit',
                        "width": 2
                        }
                },
                "splitLine": {
                    "length": 0,
                    "lineStyle": {
                        "color": 'inherit',
                        "width": 5
                    }
                },
                "axisLabel": {
                    "color": '#464646',
                    "fontSize": 20,
                    "distance": -60,
                    "rotate": 'tangential',
                    "formatter": formatter(value)
                },
                "title": {
                    "offsetCenter": [0, '-10%'],
                    "fontSize": 20
                },
                "detail": {
                    "fontSize": 30,
                    "offsetCenter": [0, '-35%'],
                    "valueAnimation": True,
                    "formatter": value,
                    "color": 'inherit'
                },
                "data": [{"value": value, "name": "Score"}],
                }
        ],
    }
    st_echarts(options=options, width="100%", key=0)

@st.cache_resource
def load_model():
    loaded_model = joblib.load("model.pkl")
    return loaded_model

@st.cache_data
def load_data(customer_ID):
    # Charger le dataframe
    df = pd.read_csv("https://media.githubusercontent.com/media/Kromette/OC-P7-Model/main/df_small.csv", index_col=0)  
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    #Récupérer les informations du client
    X_customer = df.loc[df['SK_ID_CURR'] == int(customer_ID)]
    #X_customer = X_customer[feats]
    return df, X, X_customer

@st.cache_data
def load_sample(customer_ID):
    # Charger le dataframe
    df = pd.read_csv("https://media.githubusercontent.com/media/Kromette/OC-P7-Model/main/df_small.csv", index_col=0)  
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # Modifier l'index du dataframe pour pouvoir récupérer l'index associé au client
    df = df.reset_index()
    # Récupérer les informations du client
    X_customer = df.loc[df['SK_ID_CURR'] == int(customer_ID)]
    index = X_customer.index[0]
    #X_customer = X_customer[feats]
    return df, X, X_customer, index

st.title("Prédiction de la capacité d'emprunt")
score = False
customer_ID = st.text_input('ID du client')
loaded_model = load_model()
predict_btn = st.button('Obtenir le score')
if customer_ID == '':
    st.write('Veuillez rentrer un identifiant client')
else:
    # Prédiction du score
    if predict_btn:
        st.session_state['ID'] = customer_ID
        URL = 'https://modelfastapi.herokuapp.com/customer/{}'.format(st.session_state['ID'])
        pred = requests.get(url = URL)
        score = int(float(pred.text)*100)
        st.session_state['score'] = score

# Afficher le graphe si un score est enregistré
colA, colB = st.columns(2)

if 'score' in st.session_state:
    with colA:
        graph(st.session_state['score'])
    with colB:
        st.title(' ')
        st.title(' ')
        if st.session_state['score'] < 30:
            st.title(':green[Crédit accordé]')
        else :
            st.title(':red[Crédit refusé]')

    #with st.expander('Afficher la feature importance'):
    # Obtention de l'importance
    df, X, X_customer, index = load_sample(customer_ID)
    # compute SHAP values
    explainer = shap.Explainer(loaded_model, X)
    st.title('Importance globale')
    shap_values = explainer(X, check_additivity=False)
    st_shap(shap.plots.beeswarm(shap_values), height=400)
    st.title('Importance locale')
    st_shap(shap.plots.waterfall(shap_values[index]), height=400)

    #with st.expander("Afficher l'analyse univariée"):
    st.title('Analyse univariée')
    # Choisir les features à visualiser
    df_importance = pd.DataFrame(list(zip(loaded_model.feature_importances_, loaded_model.feature_name_)), columns=['Importance', 'Feature'])
    df_importance = df_importance.sort_values('Importance', ascending=True)
    data = df_importance[-10:]

    # Afficher la distibution de la feature choisie en fonction des classes
    features = data['Feature'].to_list()
    feat = st.selectbox("Choisissez la feature à visualiser", (features))
    # Sélectionner une partie des données
    fig2 = go.Figure()
    fig2.add_traces([go.Box(x=df['TARGET'], y=df[feat], name='Clients'),
                    go.Scatter(x=[-0.5, 1.5], y=[X_customer[feat].to_list()[0], X_customer[feat].to_list()[0]], mode = 'lines', marker_color='red', name='{}'.format(customer_ID))])
    st.plotly_chart(fig2, use_container_width=True)

    #with st.expander("Afficher l'analyse bivariée"):
    st.title('Analyse bivariée')
    # Analyse bivariée
    df['color'] = df['TARGET'].astype(str)
    feat1 = st.selectbox("Choisissez la feature 1", (features))
    feat2 = st.selectbox("Choisissez la feature 2", (features))
    # Créer un template
    #my_template = go.layout.Template()
    #my_template.data.scatter = [go.Scatter(marker=dict(color='red')),
    #                            go.Scatter(marker=dict(color='green'))]

    fig3 = go.Figure()
    #fig3.update_layout(template=my_template)
    fig3.add_traces([go.Scatter(x=df[feat1] , y=df[feat2], mode='markers', marker=dict(color=df['TARGET'], size=5, colorscale = [[0, 'green'], [1, 'red']]), name='Clients'),
                    go.Scatter(x=X_customer[feat1], y=X_customer[feat2], mode='markers', marker=dict(color='black', size=10), name='{}'.format(customer_ID))])
    fig3.update_layout(
        title='Analyse bivariée {} et {}'.format(feat1, feat2),
        xaxis_title=feat1,
        yaxis_title=feat2
    )
            
    st.plotly_chart(fig3, use_container_width=True)
    st.write('En vert les clients sans défaut de paiement et en rouge ceux qui présentent un défaut de paiement')


class TestDashboard():
    def test_formatter(self):
        # Arrange
        number = 0.300
        # Act
        x = formatter(number)
        # Assert
        assert x == 'Grade C'

    def test_load_sample(self):
        # Arrange
        customer_id = 200475
        # Act
        df, X, X_customer, index = load_sample(customer_id)
        # Assert
        assert index == 1

    def test_load_model(self):
        # Arrange
        # Act
        model = load_model()
        # Assert
        assert model.objective_ == 'binary'

