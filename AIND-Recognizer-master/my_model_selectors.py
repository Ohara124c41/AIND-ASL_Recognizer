import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        optimal_score = float("inf")
        optimal_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                optimal_model = self.base_model(n)
                log_L = model.score(self.X, self.lengths)
                datum = SUM(self.lengths)
                hyper = (n**2) + (2*n*datum) - 1
                BIC = (-2*log_L) + (hyper*np.log10(datum))
                if BIC < optimal_score:
                    optimal_score = BIC
                    optimal_model = updated_model

            except:
                pass

        return optimal_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores

        ordered_pairs = [self.hwords[key] for key in self.hwords.keys() if key != self.this_word]
        optimal_DIC = float("-inf")
        optimal_model = None

        for n in range(self.min_n_components, self.max_n_components +1):
            try:
                model = self.base_model(n)
                log_L_X = model.score(self.X, self.lengths)
                log_L_O = [model.score(X[0],X[1]) for X in [ord for ord in ordered_pairs]]
                DIC = log_L_X - sum(log_L_O)/(len(log_L_O)-1)
                if DIC > optimal_DIC:
                    optimal_model = model
                    optimal_DIC = DIC
            except:
                pass

        return optimal_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        split_method = KFold()
        optimal_model = None
        optimal_score = float("-inf")

        for n in range(self.min_n_components, self.max_n_components+1):
            SUM_score = 0
            try:
                if len(self.sequences) > 2:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(cv_train_idx,self.sequences)
                        X_test, length_test = combine_sequences(cv_test_idx,self.sequences)

                        model = self.base_model(n)
                        SUM_score += model.score(X_test,length_test)

                    mean_score = SUM_score/3
                else:
                    model = self.base_model(n)
                    mean_score = model.score(self.X, self.lengths)

                if mean_score > optimal_score:
                    optimal_score = mean_score
                    optimal_model = model

            except:
                pass

        return optimal_model
