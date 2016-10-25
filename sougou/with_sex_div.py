__author__ = 'jjzhu'
import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
import numpy as np

import logging
import logging.config
import platform
import os

def logger_conf():
    """
    load basic logger configure
    :return: configured logger
    """
    if platform.system() is 'Windows':
                logging.config.fileConfig(os.path.abspath('./')+'\\conf\\logging.conf')
    elif platform.system() is 'Linux':
                logging.config.fileConfig(os.path.abspath('./')+'/conf/logging.conf')
    logger = logging.getLogger('simpleLogger')
    return logger


class SougouNBC():
    def __init__(self, seg=True, save_file_name='./data/sougou/result/sougou_result_nbc_no_seg.csv'):
        # self.seg_need = ['n', 'v', 'e', 'j', 'l']
        print(os.path.abspath('./'))
        self.mid_result_path = './data/sougou/result/nbc.no.seg.txt'
        self.my_logger = logger_conf()
        self.my_logger.info('init SougouNBC')
        self.train_file_name = './data/2W.TRAIN.pro.seg.jieba' if seg else './data/2W.TRAIN.pro.no.seg.jieba'
        self.gender_train = './data/2W.TRAIN.pro.seg.jieba' if seg else './data/2W.TRAIN.pro.no.seg.jieba'
        self.female_train = './data/2W.TRAIN.pro.seg.female.jieba'
        self.male_train = './data/2W.TRAIN.pro.seg.male.jieba'
        self.test_file_name = './data/2W.TEST.pro.jieba'
        self.stop_word_file_name = './extra_dict/stop_words_ch.txt'
        self.age_model_save_path = './model/age_ber_no_seg.model'
        self.gender_model_save_path = './model/gender_ber_no_seg.model'
        self.edu_model_save_path = './model/edu_ber_no_seg.model'
        self.save_file_name = save_file_name

        self.all_words = set()
        self.stop_words = []
        self.load_stop_word()
        # self.age_input, self.gender_input, self.edu_input = self.get_data()
        self.gender_input, self.male_age_input, self.male_edu_input,\
        self.female_age_input, self.female_edu_input = self.get_data()

        self.age_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=0.2)), ])
        self.gender_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=0.35)), ])
        self.edu_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=0.21)), ])

    def load_stop_word(self):
        self.my_logger.info('load stop word from file %s' % self.stop_word_file_name)
        start_time = datetime.datetime.now()
        with open(self.stop_word_file_name, 'r', encoding='utf-8') as s_w_file:
            for line in s_w_file:
                self.stop_words.append(line.strip())
        end_time = datetime.datetime.now()
        self.my_logger.info('load stop word completed. cost %d seconds' % (end_time-start_time).seconds)

    def get_data(self):
        self.my_logger.info('start get train data from %s' % self.train_file_name)
        start_time = datetime.datetime.now()
        male_age_input = []
        female_age_input = []
        gender_input = []
        male_edu_input = []
        female_edu_input = []
        temp_list = []
        with open(self.gender_train, mode='r', encoding='utf-8') as train_file:
            for line in train_file:
                line = line.strip()
                split_r = line.split(',')
                all_word = split_r[4]
                if split_r[2] != '0':
                    gender_input.append((all_word, split_r[2]))
                temp_list.clear()
        self.my_logger.info('start get male train data from %s' % self.male_train)
        with open(self.male_train, 'r', encoding='utf-8') as male_f:
            for line in male_f:
                line = line.strip()
                split_r = line.split(',')
                all_word = split_r[4]
                if split_r[1] != 0:
                    male_age_input.append((all_word, split_r[1]))
                if split_r[3] != 0:
                    male_edu_input.append((all_word, split_r[3]))
        self.my_logger.info('start get female train data from %s' % self.female_train)
        with open(self.female_train, 'r', encoding='utf-8') as female_f:
            for line in female_f:
                line = line.strip()
                split_r = line.split(',')
                all_word = split_r[4]
                if split_r[1] != 0:
                    female_age_input.append((all_word, split_r[1]))
                if split_r[3] != 0:
                    female_edu_input.append((all_word, split_r[3]))
        end_time = datetime.datetime.now()
        self.my_logger.info('load train data complete. cost %d' % (end_time-start_time).seconds)
        return gender_input, male_age_input, male_edu_input, female_age_input, female_edu_input

    def get_train_data(self, data_input):
        train_data_ = [elem[0] for elem in data_input]
        train_target_ = [elem[1] for elem in data_input]
        return train_data_, train_target_

    def train(self):
        self.my_logger.info('training age classify model')
        start_time = datetime.datetime.now()
        age_train_data, age_train_target = self.get_train_data(self.age_input)
        self.age_ber_nbc.fit(age_train_data, age_train_target)
        self.my_logger.info('train completed')
        self.my_logger.info('training gender classify model')
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        self.gender_ber_nbc.fit(gender_train_data, gender_train_target)
        self.my_logger.info('train completed')
        self.my_logger.info('training edu classify model')
        edu_train_data, edu_train_target = self.get_train_data(self.edu_input)
        self.edu_ber_nbc.fit(edu_train_data, edu_train_target)
        end_time = datetime.datetime.now()
        self.my_logger.info('model train completed. cost %d' % (end_time-start_time).seconds)

    def get_test_data(self):
        start_time = datetime.datetime.now()
        self.my_logger.info('load test data from %s' % self.test_file_name)
        temp_list = []
        pre_data = []
        with open(self.test_file_name, mode='r', encoding='utf-8') as test_file:
            for line in test_file:
                split_r = line.strip().split(' ')
                pre_data.append((split_r[0], ' '.join(split_r[1:])))
                temp_list.clear()
        end_time = datetime.datetime.now()
        self.my_logger.info('load test data(%s) complete.cost %s seconds' %
                            (self.test_file_name, str((end_time-start_time).seconds)))
        return pre_data

    def classify(self):
        self.my_logger.info('start classify')
        start_time = datetime.datetime.now()
        result = []
        pre_data = self.get_test_data()
        user_ids = [elem[0] for elem in pre_data]
        input_data = [elem[1] for elem in pre_data]
        age_predict = self.age_ber_nbc.predict(input_data)
        gender_predict = self.gender_ber_nbc.predict(input_data)
        edu_predict = self.edu_ber_nbc.predict(input_data)
        self.my_logger.info('classify complete')
        self.my_logger.info('start save predict result')

        result_file = open(self.save_file_name, 'w', encoding='utf-8')
        for id_, age, gender, edu in zip(user_ids, age_predict, gender_predict, edu_predict):
            result.append('%s %s %s %s\n' % (str(id_), str(age), str(gender), str(edu)))
            if len(result) > 1000:
                result_file.writelines(result)
                result_file.flush()
                self.my_logger.info('write result, total %d' % len(result))
                result.clear()
        if len(result) != 0:
            result_file.writelines(result)
            result_file.flush()
            self.my_logger.info('write result, total %d' % len(result))
            result.clear()
        result_file.close()
        end_time = datetime.datetime.now()
        self.my_logger.info('predict result saved（%s）.cost %s secound ' %
                            (self.save_file_name, str((end_time-start_time).seconds)))

    def validation2(self):
        mid_result = []

        age_train_data, age_train_target = self.get_train_data(self.age_input)
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        edu_train_data, edu_train_target = self.get_train_data(self.edu_input)
        import numpy as np
        self.my_logger.info('start cross validation')
        self.my_logger.info('validation age models')
        age_result = cross_val_score(self.age_ber_nbc, age_train_data, age_train_target, cv=10)
        self.my_logger.info('use:%s alpha:%s age:%s' % ('BernoulliNB', '0.2', str(age_result)))
        mid_result.append('use:%s alpha:%s mrean:%s age:%s\n' % ('BernoulliNB', '0.2',
                                                                 str(np.mean(age_result)), str(age_result)))

        self.my_logger.info('validation edu models')
        edu_result = cross_val_score(self.edu_ber_nbc, edu_train_data, edu_train_target, cv=10)
        self.my_logger.info('use:%s alpha:%s age:%s' % ('BernoulliNB', '0.2', str(edu_result)))
        mid_result.append('use:%s alpha:%s mrean:%s age:%s\n' % ('BernoulliNB', '0.35',
                                                                 str(np.mean(edu_result)), str(edu_result)))
        self.my_logger.info('validation gender models')
        gender_result = cross_val_score(self.gender_ber_nbc, gender_train_data, gender_train_target, cv=10)
        self.my_logger.info('use:%s alpha:%s age:%s' % ('BernoulliNB', '0.2', str(gender_result)))
        mid_result.append('use:%s alpha:%s mrean:%s age:%s\n' % ('BernoulliNB', '0.2',
                                                                 str(np.mean(gender_result)), str(gender_result)))

        self.my_logger.info('save mid result (%s)' % self.mid_result_path)
        mid_result_file = open(self.mid_result_path, 'w', encoding='utf-8')
        mid_result_file.writelines(mid_result)
        mid_result_file.flush()
        mid_result_file.close()
        self.my_logger.info('end')

    def print_grid_search_info(self, pip, param):
        self.my_logger.info('grid search.....')
        self.my_logger.info('pipeline: ' + str([name for name, _ in pip.steps]))
        self.my_logger.info('parameters:')
        self.my_logger.info(param)

    def print_best_params(self, params, gs):
        """
        print best parameters
        :param params: grid search parameters
        :param gs: grid search result
        :return: null
        """
        self.my_logger.info('Best score: %0.6f' % gs.best_score_)
        self.my_logger.info('Best parameters set:')
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(params.keys()):
            self.my_logger.info('\t%s: %r' % (param_name, best_parameters[param_name]))

    def validation(self):
        mid_result_file = open(self.mid_result_path, 'w', encoding='utf-8')
        mid_result = []

        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        male_age_train_data, male_age_train_target = self.get_train_data(self.male_age_input)
        male_edu_train_data, male_edu_train_target = self.get_train_data(self.male_edu_input)
        female_age_train_data, female_age_train_target = self.get_train_data(self.female_age_input)
        female_edu_train_data, female_edu_train_target = self.get_train_data(self.female_edu_input)

        model_params = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__use_idf': (True, False),
            'vect__norm': ('l1', 'l2'),
            'clf__alpha': [_ for _ in np.arange(0.1, 0.45, 0.01)],
        }

        self.my_logger.info('grid search male age model')
        male_age_model_pip = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB()), ])
        male_age_grid_search = GridSearchCV(male_age_model_pip, param_grid=model_params, n_jobs=-1, verbose=1)
        self.print_grid_search_info(male_age_model_pip, model_params)
        male_age_grid_search.fit(male_age_train_data, male_age_train_target)
        self.print_best_params(model_params, male_age_grid_search)

        self.my_logger.info('grid search female age model')
        female_age_model_pip = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB()), ])
        female_age_grid_search = GridSearchCV(female_age_model_pip, param_grid=model_params, n_jobs=-1, verbose=1)
        self.print_grid_search_info(female_age_model_pip, model_params)
        male_age_grid_search.fit(female_age_train_data, female_age_train_target)
        self.print_best_params(model_params, female_age_grid_search)

        self.my_logger.info('grid search male edu model')
        male_edu_model_pip = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB()), ])
        male_edu_grid_search = GridSearchCV(male_edu_model_pip, param_grid=model_params, n_jobs=-1, verbose=1)
        self.print_grid_search_info(male_edu_model_pip, model_params)
        male_age_grid_search.fit(male_edu_train_data, male_edu_train_target)
        self.print_best_params(model_params, male_edu_grid_search)

        self.my_logger.info('grid search female edu model')
        female_edu_model_pip = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB()), ])
        female_edu_grid_search = GridSearchCV(female_edu_model_pip, param_grid=model_params, n_jobs=-1, verbose=1)
        self.print_grid_search_info(female_edu_model_pip, model_params)
        male_age_grid_search.fit(female_edu_train_data, female_edu_train_target)
        self.print_best_params(model_params, female_edu_grid_search)

        self.my_logger.info('grid search gender model')
        gender_model_pip = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB()), ])
        gender_grid_search = GridSearchCV(gender_model_pip, param_grid=model_params, n_jobs=-1, verbose=1)
        self.print_grid_search_info(gender_model_pip, model_params)
        male_age_grid_search.fit(gender_train_data, gender_train_target)
        self.print_best_params(model_params, gender_grid_search)


        self.my_logger.info('save mid result (%s)' % self.mid_result_path)

        mid_result_file.writelines(mid_result)
        mid_result_file.flush()
        mid_result_file.close()
        self.my_logger.info('end')

    def test_gender(self):
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        gender_result = cross_val_score(self.gender_ber_nbc, gender_train_data, gender_train_target, cv=5)
        self.my_logger.info(str(gender_result))

    def model_save(self):
        from sklearn.externals import joblib
        self.my_logger.info('save age model, target path： %s' % self.age_model_save_path)
        joblib.dump(self.age_ber_nbc, self.age_model_save_path)
        self.my_logger.info('save gender model, target path： %s' % self.gender_model_save_path)
        joblib.dump(self.gender_ber_nbc, self.gender_model_save_path)
        self.my_logger.info('save edu model, target path： %s' % self.edu_model_save_path)
        joblib.dump(self.edu_ber_nbc, self.edu_model_save_path)

    def start(self):
        self.train()
        # self.model_save()
        self.classify()

if __name__ == '__main__':
    sougou = SougouNBC()
    sougou.validation()
    # sougou.start()