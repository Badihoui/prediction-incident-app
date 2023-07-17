import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Mapping de la fréquence des departements dans la base d'entrainement
dpt_clss2_freq = {
    '2DEPART ANYAMA': 0.15009861932938856,
    '2DEPART DABOU': 0.12978303747534517,
    '2DEPART ANDOKOI': 0.1280078895463511,
    '1DEPART AKENDJE': 0.12071005917159763,
    '2DEPART PK44': 0.09526627218934912,
    '2DEPART KOBAKRO': 0.0932938856015779,
    '2DEPART EBIMPE': 0.0863905325443787,
    '2DEPART ATTINGUIE': 0.07159763313609467,
    '2DEPART IDC': 0.04181459566074951,
    '2DEPART CIMOD': 0.028796844181459565,
    '2DEPART PRESTIGE': 0.014792899408284023,
    '2DEPART LIMARK 3': 0.014398422090729782,
    '2DEPART LIMARK 1': 0.013609467455621301,
    '2DEPART OYAK': 0.009270216962524655,
    '2DEPART SONGON': 0.0021696252465483235
 }

dpt_clss1_freq = {
    '1DEPART BINGERVILLE': 0.027380702123186884,
    '1DEPART UNIWAX': 0.019287365986966575,
    '1DEPART SANTE': 0.015766239226403196,
    '1DEPART MUTASIR': 0.015608576834139163,
    '1DEPART 15KV DOKUI 2': 0.015450914441875132,
    '1DEPART ABOBO 2': 0.0152932520496111,
    '1DEPART 748B': 0.014820264872819004,
    '1DEPART SIR': 0.014084507042253521,
    '1DEPART OCTAZ': 0.013769182257725457,
    '1DEPART SOTRAPIM': 0.01350641160395207,
    '1DEPART KKF': 0.012770653773386589,
    '1DEPART CEMOI': 0.012560437250367878,
    '1DEPART NIANGON': 0.012245112465839815,
    '1DEPART AVOCATIER': 0.012034895942821106,
    '1DEPART NOBOU': 0.011194029850746268,
    '1DEPART COTIERE': 0.011194029850746268,
    '1DEPART MAROC': 0.011141475719991592,
    '1DEPART SIPIM 5': 0.010983813327727559,
    '1DEPART 634': 0.010931259196972883,
    '1DEPART GESTOCI': 0.010931259196972883,
    '1DEPART AZITO': 0.010878705066218205,
    '1DEPART PK 18': 0.010668488543199496,
    '1DEPART ZOO': 0.010615934412444818,
    '1DEPART BANCO': 0.010510826150935463,
    '1DEPART DCH': 0.010458272020180787,
    '1DEPART SYNATRESOR': 0.010458272020180787,
    '1DEPART 350': 0.010248055497162076,
    '1DEPART SODECI': 0.0101955013664074,
    '1DEPART UDEC': 0.009985284843388691,
    '1DEPART 226': 0.009932730712634013,
    '1DEPART CITE BLANCHE': 0.009669960058860626,
    '1DEPART DJOROGOBITE': 0.009617405928105949,
    '1DEPART 740': 0.009564851797351271,
    '1DEPART 944': 0.00940718940508724,
    '1DEPART FROID': 0.009249527012823208,
    '1DEPART MACACI': 0.009091864620559177,
    '1DEPART YSICO': 0.009091864620559177,
    '1DEPART BINGERVILLE 2': 0.00882909396678579,
    '1DEPART BEAGO': 0.008776539836031112,
    '1DEPART LEM': 0.008566323313012403,
    '1DEPART 101': 0.008356106789993694,
    '1DEPART OCCITANES': 0.008303552659239016,
    '1DEPART KOUMASSI': 0.008198444397729661,
    '1DEPART RAN': 0.008145890266974985,
    '1DEPART ABLE': 0.007988227874710952,
    '1DEPART CITE SIR': 0.007883119613201598,
    '1DEPART ESSO': 0.007725457220937566,
    '1DEPART STADE': 0.007672903090182888,
    '1DEPART FRAT-MAT': 0.007620348959428211,
    '1DEPART KESSY': 0.007357578305654824,
    '1DEPART ARCADES': 0.00725247004414547,
    '1DEPART AKOUEDO': 0.007199915913390792,
    '1DEPART EBRAH': 0.007147361782636115,
    '1DEPART INTERBAT': 0.007094807651881438,
    '1DEPART 638': 0.007094807651881438,
    '1DEPART ABOBO 3': 0.007042253521126761,
    '1DEPART 118': 0.006832036998108052,
    '1DEPART LOKOA': 0.006832036998108052,
    '1DEPART COCOTERAIE': 0.006779482867353374,
    '1DEPART PRODOMO': 0.006674374605844019,
    '1DEPART DANGA': 0.006306495690561278,
    '1DEPART 205': 0.006201387429051923,
    '1DEPART SICOGI': 0.0061488332982972465,
    '1DEPART 271B': 0.0061488332982972465,
    '1DEPART MOINEAUX': 0.006096279167542569,
    '1DEPART 902': 0.006096279167542569,
    '1DEPART INJS': 0.006043725036787891,
    '1DEPART ATTOBAN': 0.0059386167752785365,
    '1DEPART SIPOREX': 0.0059386167752785365,
    '1DEPART CIMINTER 2': 0.00588606264452386,
    '1DEPART MACA': 0.00588606264452386,
    '1DEPART SETU': 0.00588606264452386,
    '1DEPART USA': 0.005833508513769182,
    '1DEPART SAVANE': 0.005780954383014505,
    '1DEPART ABOBO 1': 0.005780954383014505,
    '1DEPART 704': 0.005675846121505151,
    '1DEPART BLOHORN': 0.005623291990750473,
    '1DEPART OPT': 0.005570737859995796,
    '1DEPART AMARA': 0.005518183729241118,
    '1DEPART VGE': 0.005307967206222409,
    '1DEPART ASSA': 0.0052554130754677315,
    '1DEPART ANONO': 0.0052554130754677315,
    '1DEPART ADJOUFFOU': 0.005150304813958377,
    '1DEPART ADJAME': 0.005045196552449022,
    '1DEPART LBI': 0.0049926424216943455,
    '1DEPART CEMOI 2': 0.004834980029430313,
    '1DEPART ALLIODAN': 0.0047824258986756355,
    '1DEPART 255': 0.0047824258986756355,
    '1DEPART 703': 0.004677317637166281,
    '1DEPART 2 PLATEAUX': 0.004677317637166281,
    '1DEPART AERIA': 0.004572209375656926,
    '1DEPART REMBLAIS': 0.004572209375656926,
    '1DEPART 847': 0.004572209375656926,
    '1DEPART LOGECO': 0.004572209375656926,
    '1DEPART CALMETTE': 0.004572209375656926,
    '1DEPART STAR 11': 0.0045196552449022495,
    '1DEPART POTHY': 0.004414546983392895,
    '1DEPART CAMELIA': 0.004414546983392895,
    '1DEPART CAGEOT': 0.004414546983392895,
    '1DEPART 7122': 0.004361992852638217,
    '1DEPART HEVEAS': 0.00430943872188354,
    '1DEPART 624': 0.00430943872188354,
    '1DEPART MAURITANIE': 0.00430943872188354,
    '1DEPART VERSANT 2': 0.004256884591128863,
    '1DEPART MARCHE': 0.004204330460374185,
    '1DEPART PHOENIX': 0.004204330460374185,
    '1DEPART 271A': 0.004151776329619508,
    '1DEPART 1121': 0.004151776329619508,
    '1DEPART 639': 0.004151776329619508,
    '1DEPART COCODY': 0.0040992221988648304,
    '1DEPART 702': 0.004046668068110154,
    '1DEPART 210': 0.004046668068110154,
    '1DEPART POLYMERE': 0.003836451545091444,
    '1DEPART AGIP': 0.003836451545091444,
    '1DEPART 193': 0.003836451545091444,
    '1DEPART LANGEVIN': 0.003678789152827412,
    '1DEPART 169': 0.003626235022072735,
    '1DEPART ALLIODAN 2': 0.0035736808913180576,
    '1DEPART BESSIKOI': 0.0035736808913180576,
    '1DEPART MEECI': 0.0035736808913180576,
    '1DEPART DJIBI': 0.0035211267605633804,
    '1DEPART GMA': 0.003416018499054026,
    '1DEPART 813': 0.0032583561067899935,
    '1DEPART SICOMED': 0.0032583561067899935,
    '1DEPART KOWEIT': 0.0032058019760353162,
    '1DEPART SCA': 0.0032058019760353162,
    '1DEPART COSMOS': 0.003153247845280639,
    '1DEPART 1117': 0.0031006937145259617,
    '1DEPART SOLIC 5': 0.0030481395837712844,
    '1DEPART PROCACI': 0.00294303132226193,
    '1DEPART 323': 0.00294303132226193,
    '1DEPART CHOCODI': 0.0028904771915072526,
    '1DEPART SOTRALCI': 0.002785368929997898,
    '1DEPART PMC': 0.0026277065377338657,
    '1DEPART 1132': 0.002470044145469834,
    '1DEPART SOTACI 1': 0.0024174900147151566,
    '1DEPART WEST': 0.0024174900147151566,
    '1DEPART WASSAKARA': 0.002312381753205802,
    '1DEPART 109': 0.0022598276224511248,
    '1DEPART CITARTS': 0.0022072734916964475,
    '1DEPART COPRIM': 0.00215471936094177,
    '1DEPART MMCI': 0.0021021652301870925,
    '1DEPART INFS': 0.0019445028379230607,
    '1DEPART CSP': 0.001839394576413706,
    '1DEPART CIMAF 2': 0.001839394576413706,
    '1DEPART SODEFOR': 0.0017342863149043515,
    '1DEPART JUSTICE': 0.0017342863149043515,
    '1DEPART AVODIRE': 0.0016817321841496743,
    '1DEPART FEINDJE': 0.0016817321841496743,
    '1DEPART ATL': 0.0015766239226403195,
    '1DEPART AGHIEN': 0.0015766239226403195,
    '1DEPART SUD NIANGON': 0.0013138532688669329,
    '1DEPART CIMAF 1': 0.0013138532688669329,
    '1DEPART PLASTICA': 0.0012612991381122556,
    '1DEPART CIPROCO': 0.0005780954383014505,
    '1DEPART ANANI 2': 0.00036787891528274123
    }

def add_date_cols(_df, _date_col="date_ind"):

    cols_selected = ['TEMP', 	'DEW_POINT', 	'HUMIDITY',	'WIND_SPEED', 	'WIND_GUST', 	'PRESSURE', 	'Heure', 	'Jour', 	'mois', 	'jour-semaine']
    _df[_date_col] = pd.to_datetime(_df[_date_col])
    _df['Heure'] = _df[_date_col].dt.hour
    _df['Jour'] = _df[_date_col].dt.day
    _df['mois'] = _df[_date_col].dt.month
    _df['jour-semaine'] = _df[_date_col].dt.dayofweek + 1
    _df = _df.sort_values(_date_col)
    _df = _df.set_index(_date_col)
    _df = _df[cols_selected]

    return _df

@st.cache_data          
def load_data(file=input):
    
    df_input = pd.DataFrame()  
    try:
   
        df_input = pd.read_excel(file)
    except:
        df_input = pd.read_csv(file,engine='python', encoding='utf-8',
                                parse_dates=True,
                                infer_datetime_format=True)
    return df_input

def polynomial_features(_data,_cols,_degree=3):
    
    for col in _cols:
        for i in range(2,_degree+1):
            _data[col+"^{}".format(i)] = _data[col]**i

    return _data

def encode_cycle_var(_data, _cols):

    max_cycle = {"Heure":23, "Jour":30, "mois":12, "jour-semaine":7}
    for col in _cols :
        _data[col + '_sin'] = np.sin(2 * np.pi * _data[col]/max_cycle[col])
        _data[col + '_cos'] = np.cos(2 * np.pi * _data[col]/max_cycle[col])
    _data = _data.drop(columns=_cols)
   
    return _data

def final_transform(_df,_depart_freq, _scaler):

    _df["depart_frequence"] = _depart_freq
    var_scale = ["TEMP", 	"DEW_POINT", 	"HUMIDITY", 	"WIND_SPEED", 	"WIND_GUST", 	"PRESSURE"]
    var_cycle = ["Jour","mois","jour-semaine","Heure"]
    _df = polynomial_features(_df, var_scale)
    _df = encode_cycle_var(_df, var_cycle)
    var_cycle_cols = list(_df.iloc[:,-len(var_cycle)*2:].columns)
    X_test_scaled = _scaler.transform(_df.drop(columns=var_cycle_cols))
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=_df.drop(columns=var_cycle_cols).columns,index=_df.index)
    X_test_scaled = pd.concat([X_test_scaled,_df[var_cycle_cols]], axis=1)
   
    return X_test_scaled

def predict_depart(_df,_model , _scaler, **kwargs):
    df_temp_ = add_date_cols(_df)
    data_predict = pd.DataFrame()
    for key, value in kwargs.items():
        # print("value : ", value)
        data_predict[key] = _model.predict(final_transform(df_temp_, value, _scaler))
    data_predict.index = df_temp_.index
    return data_predict



#------------------------LES FOCNTIONS UTILES POUR LE Prob de Reg---------------------#
     
def transformation(df,mesure="max"):
    df = df.groupby(["DAY"]).agg({"TEMP":mesure,"DEW_POINT": mesure, "HUMIDITY":mesure ,"WIND_SPEED":mesure,"PRESSURE":mesure,"WIND_GUST": mesure})
    df["Mois"] = pd.to_datetime(df.index).month
    return df

def prep_data_for_nb_incident(_data, _scaler):
    cols_selected = ['TEMP', 'DEW_POINT', 'HUMIDITY', 'WIND_SPEED', 'PRESSURE', 'WIND_GUST', 'Mois']
    input_df_scaled = pd.DataFrame(_scaler.transform(_data[cols_selected]), columns=cols_selected)
    input_df_scaled.index = _data.index
    return input_df_scaled

def predict(_data,direction, _model):
    
    prediction = pd.DataFrame()
    prediction[direction] = (_model.predict(_data)).astype(int)
    prediction.index = _data.index
    return prediction

