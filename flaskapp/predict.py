
class predict:

    # FIELDS
    __numeric_dictionary = {}
    __model = None
    __pandas = None

    # CONSTRUCTOR
    def __init__(self, numeric_dictionary, model, pandas):
        if type(numeric_dictionary) == dict:
            self.__reset()
            self.__numeric_dictionary = numeric_dictionary
            self.__model = model
            self.__pandas = pandas
        else:
            raise ValueError

    # HELPFUL METHODS
    def __reset(self):
        self.__numeric_dictionary = {}

    def __create_x(self):
        dfX = self.__pandas.DataFrame(self.__numeric_dictionary)

        return dfX.transpose()


    def predict(self):
        # res = self.__model.predict(self.__create_x())
        res = self.__model.predict_proba(self.__create_x())
        print(res[0])
        print(self.__create_x())

        if res[0][0]>=0.6:
            return 'fake with probability ' + str(res[0][0])
        elif res[0][0]<=0.4:
            return 'true with probability ' + str(res[0][1])
        else:
            return 'inconclusive, fake with probability ' + str(res[0][0]) + ', true with probability ' + str(res[0][1])
