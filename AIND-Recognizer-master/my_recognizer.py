import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    for word_select in [i for i,word in enumerate(test_set.wordlist)]:
        word_loc = dict()
        for key, model in models.items():
            try:
                X,lengths = test_set.get_item_Xlengths(word_select)
                word_score = model.score(X,lengths)
            except:
                word_score = float("-inf")
            word_loc[key] = word_score

        probabilities.append(word_loc)
        guesses.append(max(word_loc, key=word_loc.get))
    # return probabilities, guesses
    return (probabilities, guesses)
