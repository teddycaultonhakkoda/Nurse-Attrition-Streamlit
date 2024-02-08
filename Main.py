from snowflake.snowpark.session import Session
import streamlit as st
import logging ,sys
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pickle
import numpy as np


# Generate LLM response
def generate_response(data_file, input_query):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
  df = data_file
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response)

@st.cache_data()
def db_connection():
    try:
        connection_parameters = {
            "account": st.secrets["account"],
            "user": st.secrets["user"],
            "password": st.secrets["pass"],
            "role": 'DATA_SCIENCE',
            "warehouse": 'DS_WAREHOUSE',
            "database": 'HEALTHCARE',
            "schema": 'NURSE_ATTRITION'
        }

        # Create and Verify Session
        session = Session.builder.configs(connection_parameters).create()
        session.add_packages("snowflake-snowpark-python", "pandas", "numpy")

        queried_table = session.sql('SELECT * FROM HEALTHCARE.NURSE_ATTRITION.EMPLOYEES_MERGED')
        queried_table = queried_table.to_pandas()
    except Exception as e:
        st.error(f'incorrect credentials or account {e}')

    return queried_table


tab1, tab2 = st.tabs(["Virtual Analyst", "Churn Prediction"])

with tab1:
    st.title('Data Conversational Tool')

    openai_api_key = st.secrets["api"]
    queried_table = db_connection()

    st.dataframe(queried_table)


    query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')

    # App logic
    if query_text is 'Enter query here ...':
        query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')

    generate_response(queried_table, query_text)

with tab2:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    queried_table = db_connection()

    salary = st.text_input("Salary")
    months_after_college = st.text_input("How many months after college did they get this job")
    tenure = st.text_input("what is their tenure in days?")
    sex = st.text_input("enter 1 if this person is a male")
    job = st.selectbox('select a role', ('nurse', 'social', 'occupational', 'technologist'))
    if len(salary) > 0 & len(months_after_college) > 0 and len(tenure) > 0 and len(sex) > 0:
        if job == 'nurse':
            nurse = 1
            occupational = 0
            social = 0
            technologist = 0
        elif job == 'social':
            nurse = 0
            occupational = 0
            social = 1
            technologist = 0
        elif job == 'occupational':
            nurse = 0
            occupational = 1
            social = 0
            technologist = 0
        elif job == 'technologist':
            nurse = 0
            occupational = 0
            social = 0
            technologist = 1
        salary_tenure = salary*tenure

        input = np.array([[salary, months_after_college, tenure, sex, nurse, occupational, social, technologist, salary_tenure]])
        outcome = model.predict(input)*100
        print(outcome)
        average_salary = np.average(queried_table["SALARY"][queried_table["MAPPED_ROLE_CLEAN"] == job])
        print(average_salary)

    st.text(f"According to the model, this employee has a {outcome}% liklihood of churn this year, anything higher than 12.7% should be looked into")
    st.text(f"The average salary for someone with this role is {average_salary}, take this into consideration")




    
