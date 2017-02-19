import re, math
import textblob.tokenizers as tt

from profanity import profanity

from textblob import TextBlob
from textstat.textstat import textstat as ts

from collections import Counter
from functools import partial


# remember you have to have golenFake vectors and goldenTrue vectors pre-computed

class nlp_optimized:

    __text = ''
    __result = []
    __word = re.compile(r'\w+')

    # CONSTRUCTOR
    def __init__(self, text, golden_fake_vector, golden_true_vector):

        if (type(text) == str) and (type(golden_fake_vector) == list) \
                and(type(golden_true_vector) == list):
            self.__reset()

            self.__golden_fake_vector = golden_fake_vector
            self.__golden_true_vector = golden_true_vector
            self.__text = text
        else:
            raise ValueError

    # HELPFUL FUNCTIONS
    def __reset(self):
        self.__text = ''
        self.__result = []

    # this function cleans up the fake text
    @staticmethod
    def __clean_text(text):
        text = text.replace('\n', ' ')
        text = text.split('.')

        text.append('')
        text = '.'.join(text)

        return text

    def __calculate_fun_results(self, fun):
        self.__result.extend(fun(self.__text))

    # the tt.word_tokenize(text) is just the generator and is not present after you used it once
    def __tokenize_text(self, text):
        self.__tokenized_text = list(tt.word_tokenize(text))

    # NLP FUNCTIONS:

    # function calculates the total number of capitalized words in a text and density
    def __check_upper_case_words_and_density(self, text):
        final_number_of_upper = 0
        length = float(len(self.__tokenized_text))
        for word in self.__tokenized_text:
            if word.isupper():
                final_number_of_upper += 1
        return [final_number_of_upper, final_number_of_upper/length]

    ## calculating readability of the text (so far only Dale-Chall readability implemented)
    @staticmethod
    def __readability_of_text(text, score="dale_chall"):
        try:
            if type(score) == str:
                if score == "dale_chall":
                    readability = ts.dale_chall_readability_score(text)
                    return [readability]
                else:
                    print('Other scores are not supported yet. You wanted: ' + score + " we have only dale_chall")
            else:
                raise ValueError
        except ValueError:
            print("the score should be of type str. You put " + str(type(score)))
            raise

    def __calculate_quest_and_ex_and_density(self, text):
        length = float(len(self.__tokenized_text))
        q_and_ex = len(list(filter(lambda x: re.match('\?|!', x), text)))
        return [q_and_ex, q_and_ex/length]

    # total vulgarity of text, DEPRECATED, because "profanity" package was not working under PyCharm
    def __vulgar_and_density(self, text):
        length = float(len(self.__tokenized_text))
        prof = 0
        for word in self.__tokenized_text:
            if profanity.contains_profanity(word):
                prof += 1
        return [prof, prof/length]

    # calculate polarity and subjectivity
    @staticmethod
    def __polarity_and_subjectivity(text):
        blob = TextBlob(text)
        return [blob.sentiment[0], blob.sentiment[1]]

    # this calculates the cosine similarities of text to goldenFake and goldenTrue
    def __calculate_avg_cosine_similarity(self, fake, text):
        if (type(fake) == bool) and fake and (type(text) == str):
            vect_text = self.__text_to_vector(text)

            return [sum([self.__get_cosine(vect_text, vec) for vec in self.__golden_fake_vector])\
                    / float(len(self.__golden_fake_vector))]

        elif (type(fake) == bool) and (not fake) and (type(text) == str):
            vect_text = self.__text_to_vector(text)
            return [sum([self.__get_cosine(vect_text, vec) for vec in
                         self.__golden_true_vector]) / float(len(self.__golden_true_vector))]
        else:
            raise ValueError


    ## this function turns the text into a vector
    def __text_to_vector(self, text):
        text = self.__relevant_words(text)
        words = self.__word.findall(text)
        return Counter(words)

    ## this function picks up the relevant words from the text == words of type 'NN' and 'JJ' in nltk language
    @staticmethod
    def __relevant_words(text):
        blob = TextBlob(text)
        tags = blob.tags
        return " ".join([t[0] for t in tags if ((t[1] == "NN") or (t[1] == "JJ"))])

    ## this function calculates cosine similarities between vec1 and vec2
    @staticmethod
    def __get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    # function returne yules lexical diversity of the text
    def __get_yules(self, text):
        """
        Returns a tuple with Yule's K and Yule's I.
        (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
        International Journal of Applied Linguistics, Vol 10 Issue 2)
        In production this needs exception handling. (what kind of exceptions??)
        """
        tokens = self.__tokenized_text
        token_counter = Counter(tok.upper() for tok in tokens)

        m1 = sum(token_counter.values())
        m2 = sum([freq ** 2 for freq in token_counter.values()])

        if m2 != m1:
            i = (m1 * m1) / (m2 - m1)
            # k = 1 / i * 10000
        # HOW TO CORRECTLY HANDLE THIS???
        else:
            i = 0

        return [i]

    # NLP CALCULATIONS

    # application of the list of functions on our dictionary
    def __calculate_nlp(self):

        # putting the list of functions by hand, that you want to apply:
        list_of_functions = [self.__check_upper_case_words_and_density,
                             self.__readability_of_text, self.__polarity_and_subjectivity,
                             self.__calculate_quest_and_ex_and_density, self.__vulgar_and_density,
                             partial(self.__calculate_avg_cosine_similarity, True),
                             partial(self.__calculate_avg_cosine_similarity, False),
                             self.__get_yules]

        # keep the text in self.__tokenized text
        self.__tokenize_text(self.__text)

        for fun in list_of_functions:
            self.__calculate_fun_results(fun)


    def get_nlp_dictionary(self):
        self.__calculate_nlp()

        return {'result': self.__result}


