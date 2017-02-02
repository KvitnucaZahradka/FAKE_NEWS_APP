
def model_it(text_to_analyse, url_to_analyse, golden_fake_vector, golden_true_vector, model,
             nlp_optimized, url_analysis, predict, pandas, metadata_fake, metadata_true, helpful_functions):

    if type(text_to_analyse) == str:
        # nlp and url analysis
        nlp = nlp_optimized.nlp_optimized(text_to_analyse, golden_fake_vector, golden_true_vector)

        # print('PRINTING URL TO ANALYSE ' + url_to_analyse)

        url = url_analysis.url_analysis(url_to_analyse, metadata_fake, metadata_true, helpful_functions)

        nlp_dic = nlp.get_nlp_dictionary()

        url_dict = url.get_url_dictionary()

        # print(url_dict)

        # here you add a nlp "dictionary". I am doing that to have as a result the same structures
        nlp_dic[list(nlp_dic.keys())[0]].extend(list(url_dict.values())[0])

        # print(nlp_dic)

        predicted = predict.predict(nlp_dic, model, pandas)
        res = predicted.predict()

        return res
    else:
        return 'check your input'
