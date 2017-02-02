from flask import render_template
from flask import request
from flaskapp import app
from flaskapp import nlp_optimized
from flaskapp import url_analysis

from flaskapp import predict
from flaskapp.model import model_it
import pandas

# DEPRECATE
# from sqlalchemy import create_engine
# import psycopg2
# import pandas as pd

import helpful_functions

# --------------------------------------------------------------------------------------------------------
# SQL - IMPORTS
# DEPRECATE FOR NOW
# user = 'Pipjak'  # add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
# con = None
# con = psycopg2.connect(database=dbname, user=user, port=5433)
# --------------------------------------------------------------------------------------------------------

# NLP  IMPORTS
golden_fake_vector = helpful_functions.safely_open('goldenFakeVector', True)
golden_true_vector = helpful_functions.safely_open('goldenTrueVector', True)

# KNOWN URL IMPORTS
metadata_fake = helpful_functions.safely_open('fake_urls_metadata', True)
metadata_true = helpful_functions.safely_open('true_urls_metadata', True)

# MODEL IMPORT
model = helpful_functions.safely_open('AdaBoost2017_2_1_15_49_1', False)

# counter
counter = 0

@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html")

    # originally you had:
    # return render_template("index.html",
    #                       title='Home', user={'nickname': 'Pipjaky'})

# --------------------------------------------------------------------------------------------------------
#  @app.route('/db')
#  def birth_page():
#     sql_query = """
#                 SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
# ;
#                 """
#     query_results = pd.read_sql_query(sql_query, con)
#     births = ""
#    print(query_results[:10])
#    for i in range(0, 10):
#        births += query_results.iloc[i]['birth_month']
#        births += "<br>"
#    return births
# --------------------------------------------------------------------------------------------------------

# @app.route('/db_fancy')
# def cesareans_page_fancy():
#    sql_query = """
#               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                """
#    query_results = pd.read_sql_query(sql_query, con)
#    births = []
#    for i in range(0, query_results.shape[0]):
#        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'],
#                           birth_month=query_results.iloc[i]['birth_month']))
#
#    return render_template('cesareans.html', births=births)
# --------------------------------------------------------------------------------------------------------

@app.route('/input')
def predictor_input():

    return render_template("input.html")


@app.route('/output')
def prediction_output():
    # pull the 'url_of_article' from input field and store it
    url_to_analyse = request.args.get('url_of_article')

    # pull 'text_of_article' from input field and store it
    text_to_analyse = request.args.get('text_of_article')

    if counter == 0:
        the_result = ''
    else:
        the_result = 'please fill up both fields'

    if (text_to_analyse != '') and (url_to_analyse != ''):
        the_result = model_it(text_to_analyse, url_to_analyse, golden_fake_vector, golden_true_vector, model,
                              nlp_optimized, url_analysis, predict, pandas,  metadata_fake, metadata_true,
                              helpful_functions)
        global counter
        counter += 1

    return render_template("output.html", the_result=the_result)
