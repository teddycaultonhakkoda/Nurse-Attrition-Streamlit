import os
from dotenv import load_dotenv
import snowflake.connector as sfc
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
_ = load_dotenv()
class Config():
    def __init__(self):
        # self.openai_base = os.getenv('OPENAI_API_BASE')
        # self.openai_key = os.getenv('OPENAI_API_KEY')
        # self.openai_type = os.getenv('OPENAI_API_TYPE')
        # self.openai_version = os.getenv('OPENAI_API_VERSION')
        # self.openai_deployment_name = os.getenv('OPENAI_API_DEPLOYMENT_NAME')
        self.sf_username = os.getenv('SECRET_SF_USERNAME')
        self.sf_password = os.getenv('SECRET_SF_PASSWORD')
        self.sf_account = os.getenv('SECRET_SF_ACCOUNT')
        self.llm = AzureOpenAI(
            openai_api_base = os.getenv('OPENAI_API_BASE'),
            openai_api_version= os.getenv('OPENAI_API_VERSION'),
            openai_api_key = os.getenv('OPENAI_API_KEY'),
            openai_api_type = os.getenv('OPENAI_API_TYPE'),
            deployment_name = os.getenv('OPENAI_API_DEPLOYMENT_NAME')
        )
        self.chat_model = AzureChatOpenAI(
            openai_api_base = os.getenv('OPENAI_API_BASE'),
            openai_api_version= os.getenv('OPENAI_API_VERSION'),
            openai_api_key = os.getenv('OPENAI_API_KEY'),
            openai_api_type = os.getenv('OPENAI_API_TYPE'),
            deployment_name = os.getenv('OPENAI_API_DEPLOYMENT_NAME')
        )
    def create_sf_conn(self,
                       wh='HAKKODATA_WH',
                       r='FR_HAKKODA_ALL_EMPLOYEES',
                       keep_alive=False):
        conn = sfc.connect(
                user = self.sf_username,
                password = self.sf_password,
                account = self.sf_account,
                warehouse = wh,
                role = r,
                client_session_keep_alive=keep_alive)
        return conn
    def close_sf_conn(self, conn):
        conn.close()