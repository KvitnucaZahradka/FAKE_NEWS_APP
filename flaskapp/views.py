from flask import render_template
from flask import request
from flaskapp import app
from flaskapp import nlp_optimized
from flaskapp import url_analysis

from flaskapp import predict
from flaskapp.model import model_it
import pandas
import json

from urllib.request import quote, unquote, urlparse

import math



import numpy as np



# DEPRECATE
# from sqlalchemy import create_engine
# import psycopg2
# import pandas as pd

from flaskapp import helpful_functions

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

# TLDS IMPORTS
tlds_domain_suffixes = helpful_functions.load_list_of_tls_domains("tlds")

# NLP  IMPORTS
golden_fake_vector = helpful_functions.safely_open('goldenFakeVector', True)
golden_true_vector = helpful_functions.safely_open('goldenTrueVector', True)

# KNOWN URL IMPORTS
metadata_fake = helpful_functions.safely_open('fake_urls_metadata', True)
metadata_true = helpful_functions.safely_open('true_urls_metadata', True)

# MODEL IMPORT
#model = helpful_functions.safely_open('AdaBoost2017_2_5_22_59_2', False)
model = helpful_functions.safely_open('AdaBoost2017_2_9_0_35_2', False)

# KEY VALUES
key_values = ['Upper Case', 'Upper Case Dens.', 'Readability', 'Polarity', 'Subjectivity', 'Quest. and Ex.',
             'Quest. and Ex. Dens.', 'Profanity', 'Profanity Dens.', 'Avg. Cosine sim. to True',
              'Avg. Cosine sim. to False', 'Lex. Diversity', 'Creation Year']


@app.route('/')
@app.route('/index')
def index():
    #return render_template("input.html")

    # originally you had:
    # return render_template("index.html", title='Home', user={'nickname': 'Pipjaky'})
    return render_template("index.html")
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


@app.route('/examples')
def examples():

    return render_template("examples.html")


@app.route('/about_model')
def about_model():

    return render_template("about_model.html")



@app.route('/output')
def prediction_output():
    # pull the 'url_of_article' from input field and store it
    url_to_analyse = request.args.get('url_of_article')

    # pull 'text_of_article' from input field and store it
    text_to_analyse = request.args.get('text_of_article')

    the_result = []

    if (text_to_analyse != '') and (url_to_analyse != ''):

        # pre-parsing using urllib
        url_to_analyse = unquote(url_to_analyse)

        g = urlparse(url_to_analyse)

        # crucial parsing element
        if g.netloc != '':
            url_to_analyse = g.netloc
        elif g.path != '':
            url_to_analyse = "https://" + g.path
            g = urlparse(url_to_analyse)
            url_to_analyse = g.netloc

        # final parse by my build-in function
        parse_url = helpful_functions.extract_domain_name(tlds_domain_suffixes, url_to_analyse)
        # parse_url = tld.get_tld(url_to_analyse, as_object=True, fail_silently=True)
        print(url_to_analyse)
        if parse_url!="":
            url_to_analyse = parse_url  # parsing the URL
            print(url_to_analyse)
            the_result = model_it(text_to_analyse, url_to_analyse, golden_fake_vector, golden_true_vector, model,
                              nlp_optimized, url_analysis, predict, pandas, metadata_fake, metadata_true,
                              helpful_functions)
        else:
            the_result = ['Wrong URL, please fix URL']

    if len(the_result)==0:
        return render_template("output.html")
    elif len(the_result)==1:
        return render_template("output.html", the_result=the_result[0])
    else:
        return render_template("output.html", the_result=the_result[0], mu0=the_result[1], mu1=the_result[2],
                           mu2=the_result[3], mu3=the_result[4], mu4=the_result[5], mu5=the_result[6],
                           mu6=the_result[7], mu7=the_result[8], mu8=the_result[9], mu9=the_result[10],
                           mu10=the_result[11], mu11=the_result[12], mu12=the_result[13])



# KEY VALUES
key_values = ['Upper Case', 'Upper Case Dens.', 'Readability', 'Polarity', 'Subjectivity', 'Quest. and Ex.',
             'Quest. and Ex. Dens.', 'Profanity', 'Profanity Dens.', 'Avg. Cosine sim. to True',
              'Avg. Cosine sim. to False', 'Lex. Diversity', 'Source Relevance (based on creat. year)']

influence = ['negative', 'negative', 'negative', 'negative', 'negative', 'negative',
             'negative', 'negative', 'negative', 'positive',
              'negative', 'positive', 'negative']


mu_values = ['mu0', 'mu1', 'mu2', 'mu3', 'mu4', 'mu5', 'mu6', 'mu7', 'mu8', 'mu9', 'mu10', 'mu11', 'mu12']


importances = [0.02818359,  0.02950097,  0.03561894,  0.04241241,  0.01857136,
        0.00571765,  0.1757425,  0.00800361,  0.00818607,  0.05985107,
        0.07807531,  0.04264085,  0.46749567]



@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/gdata")
@app.route("/gdata/<float:mu0>/<float:mu1>/<float:mu2>/<float:mu3>/<float:mu4>/<float:mu5>/<float:mu6>/<float:mu7>/<float:mu8>/<float:mu9>/<float:mu10>/<float:mu11>/<float:mu12>")
def gdata():

    mu0 = float(request.args.get('mu0'))
    mu1 = float(request.args.get('mu1'))
    mu2 = float(request.args.get('mu2'))
    mu3 = float(request.args.get('mu3'))
    mu4 = float(request.args.get('mu4'))
    mu5 = float(request.args.get('mu5'))
    mu6 = float(request.args.get('mu6'))
    mu7 = float(request.args.get('mu7'))
    mu8 = float(request.args.get('mu8'))
    mu9 = float(request.args.get('mu9'))
    mu10 = float(request.args.get('mu10'))
    mu11 = float(request.args.get('mu11'))
    mu12 = float(request.args.get('mu12'))

    req = [math.log(abs(mu0)+1), math.log(abs(mu1)+1), math.log(abs(mu2)+1), math.log(abs(mu3)+1),
           math.log(abs(mu4)+1), math.log(abs(mu5)+1), math.log(abs(mu6)+1), math.log(abs(mu7)+1),
           math.log(abs(mu8)+1), math.log(abs(mu9)+1), math.log(abs(mu10)+1), math.log(abs(mu11)+1),
           math.log(abs(mu12)+1)]

    orig_req = [mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12]


    # turn nlp_dic to [{"key": value, ..}]
    #    dic_to_dump = helpful_functions.turn_nlp_to_dump(key_values, )

    # print(dic_to_dump)
    #    dump = json.dumps([dic_to_dump])

    """
    On request, this returns a list of ``ndata`` randomly made data points.
    about the mean mux,muy

    :param ndata: (optional)
        The number of data points to return.

    :returns data:
        A JSON string of ``ndata`` data points.

    """


    # creating dictionary to dump
    res = []
    count = 0
    for i in key_values:
        res.append({"_id": i, "inf": influence[count], "x": i, "y": req[count], "z": orig_req[count]})
        count += 1

    print(res)
    return json.dumps(res)

