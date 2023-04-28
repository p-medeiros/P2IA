import streamlit as st
import pandas as pd
import pydeck as pdk
import pickle
import tensorflow
import keras
import sklearn

rede_neural  = pickle.load(open("rede_neural.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))


# Transformações
education_transform = {
  "Below College": 1,
  "College": 2,
  "Bachelor": 3,
  "Master": 4,
  "Doctor": 5
}

levels = {
  'Low': 1,
  'Medium': 2,
  'High': 3,
  'Very High': 4,
}

quality = {
  'Bad': 1,
  'Good': 2,
  'Better': 3,
  'Best': 4,
}

boolean = {
  'Yes': 1,
  'No': 0
}

def escalonameto(a, b):
  return 1.0 if a == b else 0

#
st.header("Parâmetros")

#Dados Pessoais
age = st.slider('Idade', min_value=18, max_value=100)
gender = st.selectbox('Genero', ['Female', 'Male'])
marital_status = st.selectbox('Estado civil', ["Single","Married","Divorced"])

#Informações Profissionais
job_role = st.selectbox('Area de trabalho', ['Sales Executive','Research Scientist','Laboratory Technician',
'Manufacturing Director','Healthcare Representative','Manager','Sales Representative' 'Research Director',
'Human Resources'])
job_level = st.slider('Hieraquia no trabalho', min_value=1, max_value=5)
departament = st.selectbox('Departamento', ['Sales', 'Research & Development', 'Human Resources'])
business_travel = st.selectbox('Viagens de trablaho', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
overtime = st.selectbox('Hora extra', ['No', 'Yes'])

#Informações de Trabalho
monthly_income = st.number_input('Salario mensal', min_value=0, step=1)
total_working_years = st.number_input('Anos de trablaho', min_value=0, step=1)
num_companies_worked = st.number_input('Numero de empresas trabalhadas', min_value=0, step=1)
years_at_company = st.number_input('Anos na empresa', min_value=0, step=1)
years_in_current_role = st.number_input('Anos no cargo', min_value=0, step=1)
years_since_last_promotion = st.number_input('Anos desde a ultima promoção', min_value=0, step=1)
years_with_current_manager = st.number_input('Anos com o atual gerente', min_value=0, step=1)
training_times_last_year = st.number_input('Tempo de treino no ultimo ano', min_value=0, step=1)
stock_option_level = st.slider('Opção de Mercado de valores', min_value=0, max_value=3)

#Informações de Educação
education = st.selectbox('Educação', ["Below College","College","Bachelor","Master","Doctor"])
education_field = st.selectbox('Campo de estudo', ['Life Sciences','Other','Medical', 'Marketing', 'Technical Degree', 'Human Resources'])

#Satisfação
job_satisfaction = st.selectbox('Satisfação com o trabalho', ["Low", "Medium", "High", "Very High"])
environment_satisfaction = st.selectbox('Satisfação com o ambiente', ["Low", "Medium", "High", "Very High"])
relationship_satisfaction = st.selectbox('Satisfação com a relação', ["Low", "Medium", "High", "Very High"])
work_life_balance = st.selectbox('Equilibrio de vida / trabalho', ["Bad","Good","Better","Best"])
job_involvement = st.selectbox('Envolvimento com o trabalho', ["Low", "Medium", "High", "Very High"])
distance_from_home = st.number_input('Distancia do trabalho', step=1, min_value=0)

# Definindo DataFrame
data = {
  'Age': [age],
  'DistanceFromHome': [distance_from_home],
  'Education': [education_transform[education]],
  'EnvironmentSatisfaction': [levels[environment_satisfaction]],
  'Gender': [1 if gender == 'Female' else 0],
  'JobInvolvement': [levels[job_involvement]],
  'JobLevel': [job_level],
  'JobSatisfaction': [levels[job_satisfaction]],
  'MonthlyIncome': [monthly_income],
  'NumCompaniesWorked': [num_companies_worked],
  'OverTime': [boolean[overtime]],
  'RelationshipSatisfaction': [levels[relationship_satisfaction]],
  'StockOptionLevel': [stock_option_level],
  'TotalWorkingYears': [total_working_years],
  'TrainingTimesLastYear': [training_times_last_year],
  'WorkLifeBalance': [quality[work_life_balance]],
  'YearsAtCompany': [years_at_company],
  'YearsInCurrentRole': [years_in_current_role],
  'YearsSinceLastPromotion': [years_since_last_promotion],
  'YearsWithCurrManager': [years_with_current_manager],
  'MaritalStatus_Divorced': [escalonameto('Divorced', marital_status)],
  'MaritalStatus_Married': [escalonameto('Married', marital_status)],
  'MaritalStatus_Single': [escalonameto('Single', marital_status)],
  'JobRole_Healthcare Representative': [escalonameto('Healthcare Representative', job_role)],
  'JobRole_Human Resources': [escalonameto('Human Resources', job_role)],
  'JobRole_Laboratory Technician': [escalonameto('Laboratory Technician', job_role)],
  'JobRole_Manager': [escalonameto('Manager', job_role)],
  'JobRole_Manufacturing Director': [escalonameto('Manufacturing Director', job_role)],
  'JobRole_Research Director': [escalonameto('Research Director', job_role)],
  'JobRole_Research Scientist': [escalonameto('Research Scientist', job_role)],
  'JobRole_Sales Executive': [escalonameto('Sales Executive', job_role)],
  'JobRole_Sales Representative': [escalonameto('Sales Representative', job_role)],
  'EducationField_Human Resources': [escalonameto('Human Resources', education_field)],
  'EducationField_Life Sciences': [escalonameto('Life Sciences', education_field)],
  'EducationField_Marketing': [escalonameto('Marketing', education_field)],
  'EducationField_Medical': [escalonameto('Medical', education_field)],
  'EducationField_Other': [escalonameto('Other', education_field)],
  'EducationField_Technical Degree': [escalonameto('Technical Degree', education_field)],
  'Department_Human Resources': [escalonameto('Human Resources', departament)],
  'Department_Research & Development': [escalonameto('Research & Development', departament)],
  'Department_Sales': [escalonameto('Sales', departament)],
  'BusinessTravel_Non-Travel': [escalonameto('Non-Travel', business_travel)],
  'BusinessTravel_Travel_Frequently': [escalonameto('Travel_Frequently', business_travel)],
  'BusinessTravel_Travel_Rarely': [escalonameto('Travel_Rarely', business_travel)],
}

df = pd.DataFrame(data)
X = scaler.transform(df)

result = "Deixará a empresa" if rede_neural.predict(X) > 0.5 else "Não deixará a empresa"
color = "red" if rede_neural.predict(X) > 0.5 else "green"


# MAIN
st.sidebar.title("Modelo de previsão do RH")
st.sidebar.markdown("Esse Funcionário :"+color+"["+ result +"].")
st.sidebar.dataframe(df.T)
