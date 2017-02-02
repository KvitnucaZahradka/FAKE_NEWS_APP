
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
        res = self.__model.predict(self.__create_x())

        if res == 1:
            return 'not a fake'
        else:
            return 'fake'