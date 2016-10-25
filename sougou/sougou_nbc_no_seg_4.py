__author__ = 'jjzhu'
import jieba
import jieba.posseg as pseg
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_predict, ShuffleSplit, cross_val_score
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import logging
import logging.config


def logger_conf():
    import platform
    import os
    if platform.system() is 'Windows':
                logging.config.fileConfig(os.path.abspath('./')+'\\conf\\logging.conf')
    elif platform.system() is 'Linux':
                logging.config.fileConfig(os.path.abspath('./')+'/conf/logging.conf')
    logger = logging.getLogger('simpleLogger')
    return logger


class SougouNBC():
    def __init__(self, save_file_name='./data/sougou/result/sougou_result_no_seg.csv'):
        # self.seg_need = ['n', 'v', 'e', 'j', 'l']
        self.mid_result_path = './data/sougou/result/nbc.no.seg.4.txt'
        self.my_logger = logger_conf()
        self.my_logger.info('init SougouNBC')
        self.train_file_name = './data/user_tag_query.2W.TRAIN'
        self.test_file_name = './data/user_tag_query.2W.TEST'
        self.stop_word_file_name = './extra_dict/stop_words_ch.txt'
        self.age_model_save_path = './model/age_ber_no_seg.model'
        self.gender_model_save_path = './model/gender_ber_no_seg.model'
        self.edu_model_save_path = './model/edu_ber_no_seg.model'
        self.save_file_name = save_file_name
        self.my_logger.info('start get train data from %s' % self.train_file_name)
        self.all_words = set()
        self.stop_words = []
        self.load_stop_word()
        self.age_input, self.gender_input, self.edu_input = self.get_data(self.train_file_name)

        self.age_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
        self.gender_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
        self.edu_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])

        self.age_ber_nbc = Pipeline([('vect', HashingVectorizer()), ('clf', BernoulliNB(alpha=0.3)), ])
        self.gender_ber_nbc = Pipeline([('vect', HashingVectorizer()), ('clf', BernoulliNB(alpha=0.35)), ])
        self.edu_ber_nbc = Pipeline([('vect', HashingVectorizer()), ('clf', BernoulliNB(alpha=0.3)), ])


    def load_stop_word(self):
        with open(self.stop_word_file_name, 'r', encoding='utf-8') as s_w_file:
            for line in s_w_file:
                self.stop_words.append(line.strip())

    def get_data(self, train_f_n):

        age_input = []
        gender_input = []
        edu_input = []
        temp_list = []
        rewrite_file_name = './data/sougou/rewrite4.csv'
        unused_words_file_name = './data/sougou/unused_words4.csv'
        rewrite_file = open(rewrite_file_name, 'w', encoding='utf-8')
        unused_file = open(unused_words_file_name, 'w', encoding='utf-8')
        rewrite_context = []
        unused_context = []
        with open(train_f_n, mode='r', encoding='utf-8') as train_file:
            for line in train_file:
                line = line.strip()
                split_r = line.split('\t')
                pesg_words = jieba.cut(','.join(split_r[4:]))
                for pesg_word in pesg_words:
                    if pesg_word not in self.stop_words \
                            and len(pesg_word) > 1:
                        temp_list.append(str(pesg_word))
                    else:
                        unused_context.append('%s \n' % pesg_word)
                all_word = ' '.join(temp_list)
                age_input.append((all_word, split_r[1]))
                gender_input.append((all_word, split_r[2]))
                edu_input.append((all_word, split_r[3]))
                '''
                if split_r[1] != '0':
                    age_input.append((all_word, split_r[1]))
                if split_r[2] != '0':
                    gender_input.append((all_word, split_r[2]))
                if split_r[3] != '0':
                    edu_input.append((all_word, split_r[3]))
                '''
                rewrite_context.append(' '.join(temp_list) + '\n')
                temp_list.clear()
        self.my_logger.info('rewrite word cut result to file: %s' % rewrite_file_name)
        rewrite_file.writelines(rewrite_context)
        self.my_logger.info('rewrite complete')
        self.my_logger.info('rewrite unused words  to file: %s' % unused_words_file_name)
        unused_file.writelines(unused_context)
        self.my_logger.info('rewrite complete')
        return age_input, gender_input, edu_input

    def get_train_data(self, data_input):
        train_data_ = [elem[0] for elem in data_input]
        train_target_ = [elem[1] for elem in data_input]
        return train_data_, train_target_

    def train(self):
        self.my_logger.info('training age classify model')
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
        self.my_logger.info('model train completed')

    def classify(self):
        self.my_logger.info('start classify')
        pre_data = []
        temp_list = []
        result = []

        with open(self.test_file_name, mode='r', encoding='utf-8') as test_file:
            for line in test_file:
                split_r = line.strip().split('\t')
                words = jieba.cut(','.join(split_r[1:]))
                for w in words:
                    if w not in self.stop_words and len(w) > 1:  #
                        temp_list.append(str(w))
                if len(temp_list) > 0:
                    pre_data.append((split_r[0], ' '.join(temp_list)))
                else:
                    result.append('%s %s %s %s\n' % (str(split_r[0]), '0', '0', '0'))
                    self.my_logger.warn('%s %s %s %s\n\t%s' % (str(split_r[0]), '0', '0', '0', str(words)))
                temp_list.clear()
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
        self.my_logger.info('predict result saved（%s） ' % self.save_file_name)

    def validation(self):

        mid_result = []
        age_train_data, age_train_target = self.get_train_data(self.age_input)
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        edu_train_data, edu_train_target = self.get_train_data(self.edu_input)
        import numpy as np
        self.my_logger.info('start cross validation')
        clf_name = ['BernoulliNB', 'MultinomialNB']
        cv = ShuffleSplit(n=len(age_train_data), n_iter=10, test_size=0.3, random_state=0)
        self.my_logger.info('validation age models')
        age_models = {i: Pipeline([('vect', HashingVectorizer()), ('clf', BernoulliNB(alpha=i)), ])
                      for i in np.arange(0.01, 1, 0.01)}
        age_sort_models = sorted(age_models.items(), key=lambda elem: elem[0])
        for item in age_sort_models:
            age_result = cross_val_score(item[1], age_train_data, age_train_target, cv=10)
            self.my_logger.info('use:%s alpha:%s age:%s' % (clf_name[1], item[0], str(age_result)))
            mid_result.append('use:%s alpha:%s mrean:%s age:%s\n' % (clf_name[1], item[0],
                                                                     str(np.mean(age_result)), str(age_result)))
        self.my_logger.info('validation edu models')
        edu_models = {str(i): Pipeline([('vect', HashingVectorizer()), ('clf', MultinomialNB(alpha=i)), ])
                      for i in np.arange(0.01, 1, 0.01)}
        edu_sort_models = sorted(edu_models.items(), key=lambda elem: elem[0])

        for item in edu_sort_models:
            edu_result = cross_val_score(item[1], edu_train_data, edu_train_target, cv=10)
            self.my_logger.info('use:%s alpha:%s mrean:%s edu:%s' % (clf_name[1], item[0],
                                                                     str(np.mean(edu_result)), str(edu_result)))
            mid_result.append('use:%s alpha:%s mrean:%s edu:%s\n' % (clf_name[1], item[0],
                                                                     str(np.mean(edu_result)), str(edu_result)))

        '''
        self.my_logger.info('validation gender models')
        gender_models = {str(i): Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=i)), ])
                         for i in np.arange(0.1, 1, 0.05)}
        for alp, gender_model in gender_models.items():
            gender_result = cross_val_score(gender_model, gender_train_data, gender_train_target, cv=10)
            self.my_logger.info('use:%s alpha:%s mrean:%s gender:%s' % (clf_name[1], alp,
                                                                        str(np.mean(gender_result)), str(gender_result)))
            mid_result.append('use:%s alpha:%s mrean:%s gender:%s\n' % (clf_name[1], alp,
                                                                        str(np.mean(gender_result)), str(gender_result)))
        self.my_logger.info('validation edu models')
        edu_models = {str(i): Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=i)), ])
                      for i in np.arange(0.1, 1, 0.05)}
        for alp, edu_model in edu_models.items():
            edu_result = cross_val_score(edu_model, edu_train_data, edu_train_target, cv=10)
            self.my_logger.info('use:%s alpha:%s mrean:%s edu:%s' % (clf_name[1], alp,
                                                                     str(np.mean(edu_result)), str(edu_result)))
            mid_result.append('use:%s alpha:%s mrean:%s edu:%s\n' % (clf_name[1], alp,
                                                                     str(np.mean(edu_result)), str(edu_result)))
        '''
        # mean_result = []
        # for a, g, e in zip(age_result, gender_result, edu_result):
        #     mean_result.append(np.mean([a, g, e]))
        # self.my_logger.info('mean result: %s' % str(mean_result))
        self.my_logger.info('save mid result (%s)' % self.mid_result_path)
        mid_result_file = open(self.mid_result_path, 'r', encoding='utf-8')
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