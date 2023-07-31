import streamlit as st
import pandas as pd
from pandas import Timestamp
import numpy as np
from utils import *
# import joblib
import joblib

# Import model and scaler pour la classification
model_classif1 = joblib.load(open("model/model_statut/rf_classif_cl1.joblib", "rb"))
scaler_classif1 = joblib.load(open("scaler/scaler_statut/scaler_classif_cl1.joblib", 'rb'))
model_classif2 = joblib.load(open("model/model_statut/rf_classif_cl2.joblib", "rb"))
scaler_classif2 = joblib.load(open("scaler/scaler_statut/scaler_classif_cl2.joblib", 'rb'))

# Import model and scaler pour le nombre d'incident
model_drabo = joblib.load(open("model/rf_model_drabo.joblib", "rb"))
model_dran = joblib.load(open("model/rf_model_dran.joblib", "rb"))
model_dras = joblib.load(open("model/rf_model_dras.joblib", "rb"))
model_dryop = joblib.load(open("model/rf_model_dryop.joblib", "rb"))

scaler_drabo = joblib.load(open("scaler/scale_model_drabo.joblib", 'rb'))
scaler_dran = joblib.load(open("scaler/scale_model_dran.joblib", "rb"))
scaler_dras = joblib.load(open("scaler/scale_model_dras.joblib", "rb"))
scaler_dryop = joblib.load(open("scaler/scale_model_dryop.joblib", "rb"))



st.set_page_config(page_title="Prediction Incident",
                   initial_sidebar_state="collapsed",
                   page_icon="chart_with_upwards_trend"
                )
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)
tabs = ['Statut Incident','Prediction Nombre Incident', 'Performance des modèles']
page = st.sidebar.selectbox("Selectionnez  une page", tabs)
add_bg_from_local("abstract-room-sun-card-blank.jpg") 



if page == "Statut Incident":

    st.title('Prediction Statut Incident des Departs 🧙🏻')
    st.markdown("""Cette application vous permet de faire les predictions des incidents sur les departs pour les heures suivantes à l'aide des données meteos.""")
    
    df =  pd.DataFrame()   

    st.subheader('1. Chargement de Données 🏋️')
    st.write("Importer les predicteurs meteos en fichier csv.")
  
    st.markdown("""**NB** : Les données meteos doivent avoir impérativement les variables suivantes : DATE, TEMP, DEW_POINT, HUMIDITY, WIND_SPEED, WIND_GUST, PRESSURE.""")
    input = st.file_uploader("Telechargez le fichier svp", type=["csv","xlsx","xls"])

    try :
        if input:
            with st.spinner('Loading Data...'):
                df = load_data(input)
                st.write("ploting data : ")
                st.dataframe(df)
    except:
        if input is None:
            st.write("Vous devez telecharger un fichier CVS comportant les predicteurs")

       
    st.subheader("2. Forecast 🔮")
    with st.container():
        st.write("Générer des predictions.")
        if st.button("Predictions"):
            try:
                with st.spinner("Predictions..."):
                    # dataframe de la prediction
                    result1 = predict_depart(_df=df, _model = model_classif1, _scaler = scaler_classif1, **dpt_clss1_freq).reset_index() 
                    result2 = predict_depart(_df=df, _model = model_classif2, _scaler = scaler_classif2, **dpt_clss2_freq).reset_index() 
                    result_copy_ = result1.merge(result2, on="date_ind")
                    nbr_total_dpt = len(result_copy_.columns.values.tolist())
                    date_ = []
                    count_inci_ = []
                    list_dpt = []
                    for  ind, row in result_copy_.iterrows():                       
                        # nom depart incident
                        df_tmp_ = result_copy_.iloc[[ind]]
                        # date
                        date_.append(df_tmp_.at[ind, "date_ind"])
                    
                        col_list = (df_tmp_ == 1 ).any()
                        df_tmp_= df_tmp_.loc[: , col_list]
                        noms_ = [col for col in df_tmp_.columns.values.tolist()]
                        list_dpt.append(noms_)
                        # nombre dpt
                        count_inci_.append(len(noms_))

                    # dictionnaire pour stocker le nombre de dpt atteint d'incident par date
                    tmp_dict_ = {
                        "DATE":date_,
                        "NOMBRE_DEPART_INCIDENT": count_inci_,
                        "PROPORTION(%)": [(val/nbr_total_dpt)*100 for val in count_inci_],
                        "LISTE_DEPART_AFFECTE":list_dpt
                    }
                   
                    # dataframe du nombre de dpt atteint d'incident par date
                    df_dept_incident_ = pd.DataFrame(tmp_dict_)
                
                    st.success('Prediction generated sucessfully')
                    st.markdown("Le nombre de depart atteint par les incidents sont : ")
                    st.dataframe(df_dept_incident_)
                    
                  
            except:
                    st.warning("No model found.. ")


if page == "Prediction Nombre Incident":
                     
    st.title("""Prediction du nombre d'incident quotidien par Direction 🧙🏻""")
    st.markdown("""Cette application vous permet de faire les prédictions du nombre d'incident pour un jour donné à l'aide des données metéos.""")

    # legacy_caching.clear_cache()
    df =  pd.DataFrame()   

    st.subheader('1. Chargement de Données 🏋️')
    st.write("Importer les predicteurs meteos en fichier csv.")
    st.markdown("""**NB** : Les prédicteurs sont : DAY, TEMP, DEW_POINT, HUMIDITY, WIND_SPEED, WIND_GUST, PRESSURE.""")

    input = st.file_uploader("Upload your file here...", type=["csv","xlsx","xls"])

    try :
        if input:
            with st.spinner('Loading Data...'):
                df = load_data(input)
                st.write("ploting data : ")
                st.dataframe(df)
            
    
    except:
        if input is None:
            st.write("Vous devez telecharger un fichier CVS comportant les predicteurs")

       
    st.subheader("2. Forecast 🔮")
    with st.container():
      
        # Selection de la Direction
        
        list_direction = ['DRABO', 'DRAN', 'DRAS', 'DRYOP']
        direction_input = st.selectbox("Selectionnez la Direction :", list_direction)

        st.write("Générer des predictions.")
        if st.button('Prediction'):
            # Donnees aggregé
            #st.write("Data après transformation...")
            input_agg_data = transformation(df)
            # st.dataframe(input_agg_data)

            try:
                with st.spinner("Predictions..."):

                    if direction_input == list_direction[0]:
                        input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_drabo)
                        predictions = predict(input_scale_df, direction_input , model_drabo)

                    elif direction_input == list_direction[1]:
                        input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dran)
                        predictions = predict(input_scale_df, direction_input, model_dran)

                    elif direction_input == list_direction[2]:
                        input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dras)
                        predictions = predict(input_scale_df, direction_input, model_dras)

                    else:
                        input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dryop)
                        predictions = predict(input_scale_df, direction_input, model_dryop)
                    
                    st.success('Prediction generated sucessfully')
                    st.write(f"Nombre Incident: ")
                    st.dataframe(predictions)
            except:
                    st.warning("No model found.. ")
            # with st.spinner("Predictions..."):

            #         if direction_input == list_direction[0]:
            #             input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_drabo)
            #             predictions = predict(input_scale_df, direction_input , model_drabo)

            #         elif direction_input == list_direction[1]:
            #             input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dran)
            #             predictions = predict(input_scale_df, direction_input, model_dran)

            #         elif direction_input == list_direction[2]:
            #             input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dras)
            #             predictions = predict(input_scale_df, direction_input, model_dras)

            #         else:
            #             input_scale_df = prep_data_for_nb_incident(input_agg_data, scaler_dryop)
            #             predictions = predict(input_scale_df, direction_input, model_dryop)
                    
            #         st.success('Prediction generated sucessfully')
            #         st.write(f"Nombre Incident: ")
            #         st.dataframe(predictions)

if page == "Performance des modèles":
    with st.container():
        df_performance_dict_1 = {
            "Accuracy" : [0.74],
            "Precision": [0.73],
            "Rappel": [0.75],
            "F1-score": [0.74]
        }
        df_performance_1 = pd.DataFrame(df_performance_dict_1)
        st.title("""Performance des différents modèles 🧙🏻""")
        st.subheader('1. Performance du modèle du statut incident sur les départs 🏋️')
        st.markdown("""Etant donné qu'il s'agit d'un problème de classification, c'est à dire d'incident ou pas sur un depart bien donné, les métriques qui ont été utilisés pour mesurer la performance du modèle sont : """)
        st.markdown("""**NB : Accuracy, Precision, Rappel, F1-score** """)
        st.markdown("""**Les performances sont** """ )
        st.dataframe(df_performance_1)


    with st.container():
        df_performance_dict_2 = {
            "Direction" : ["DRAN", "DRAS", "DRABO", "DRYOP"],
            "MAE" : [5.94, 4.79, 2.36, 3.87],
            "RMSE": [9.48, 8.47, 3.37, 6.42],
            "Moyenne des observations": [8, 5, 3, 5]
        }
        df_performance_2 = pd.DataFrame(df_performance_dict_2)
        st.subheader('2. Performance du modèle du nombre incident par  direction 🏋️')
        st.markdown("""Etant donné qu'il s'agit d'un problème de regression, c'est à dire du nombre d'incident pour une direction bien donnée, les métriques qui ont été utilisés pour mesurer la performance du modèle sont : """)
        st.markdown("""**NB : Erreur Moyenne Absolue(MAE), Erreur moyennne quadratique(RMSE)** """)
        st.markdown("""**Les performances sont** """ )
        st.dataframe(df_performance_2)

	

	