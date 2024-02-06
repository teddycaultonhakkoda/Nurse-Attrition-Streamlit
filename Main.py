from snowflake.snowpark.session import Session
import streamlit as st
import logging ,sys
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


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



st.title('Data Conversational Tool')

openai_api_key = st.secrets["api"]
queried_table = db_connection()

st.dataframe(queried_table)


query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')

# App logic
if query_text is 'Enter query here ...':
    query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')

generate_response(queried_table, query_text)




    
