import pickle

df_sentiment_score=pickle.load(open(''))

from pyathena import connect
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('')
AWS_ACCESS_KEY = config.get('aws', 'aws_access_key')
AWS_SECRET_KEY = config.get('aws', 'aws_secret_key')

client_id = config.get("uTp5HipGLhBhPv1S7oxk8A")
client_secret = config.get("AcT-azqNdIT3onTltclkX4Q3ISUDDQ")