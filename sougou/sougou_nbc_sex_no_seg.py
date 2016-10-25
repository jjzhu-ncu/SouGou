__author__ = 'jjzhu'
"""
剔除了所有标签未知的数据
按写别划分：
    性别
    年龄+性别
    学历+性别
"""
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
    def __init__(self, save_file_name='./data/sougou/result/result_sex_no_seg.csv'):
        # self.seg_need = ['n', 'v', 'e', 'j', 'l']
        self.my_logger = logger_conf()
        self.my_logger.info('init SougouNBC')
        self.train_file_name = './data/user_tag_query.2W.TRAIN'
        self.test_file_name = './data/user_tag_query.2W.TEST'
        self.stop_word_file_name = './extra_dict/stop_words_ch.txt'
        self.age_model_save_path = './model/age_ber_sex_no_seg.model'
        self.gender_model_save_path = './model/gender_ber_sex_no_seg.model'
        self.edu_model_save_path = './model/edu_ber_sex_no_seg.model'



        self.save_file_name = save_file_name
        self.my_logger.info('start get train data from %s' % self.train_file_name)
        self.all_words = set()
        self.stop_words = []
        self.load_stop_word()
        self.age_input, self.gender_input, self.edu_input = self.get_data(self.train_file_name)

        self.age_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
        self.gender_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
        self.edu_mul_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])

        self.age_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=1.0)), ])
        self.gender_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=1.0)), ])
        self.edu_ber_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=1.0)), ])

        self.age_gs_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', GaussianNB()), ])
        self.gender_gs_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', GaussianNB()), ])
        self.edu_gs_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', GaussianNB()), ])

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
        no_sex = True
        with open(train_f_n, mode='r', encoding='utf-8') as train_file:
            for line in train_file:
                line = line.strip()
                split_r = line.split('\t')
                pesg_words = jieba.cut(','.join(split_r[4:]), HMM=True)
                for pesg_word in pesg_words:
                    if pesg_word not in self.stop_words \
                            and len(pesg_word) > 1:
                        temp_list.append(str(pesg_word))
                    else:
                        unused_context.append('%s \n' % pesg_word)
                all_word = ' '.join(temp_list)

                if split_r[2] != '0':
                    gender_input.append((all_word, split_r[2]))
                    no_sex = False
                if split_r[1] != '0' and not no_sex:
                    age_input.append((all_word, split_r[2]+'_'+split_r[1]))

                if split_r[3] != '0' and not no_sex:
                    edu_input.append((all_word, split_r[2]+'_'+split_r[3]))
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
                words = jieba.cut(','.join(split_r[1:]), HMM=True)
                for w in words:
                    if w not in self.stop_words:  # and len(w) > 1
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
        self.my_logger.info('predict result saved ')

    def validation(self):

        age_train_data, age_train_target = self.get_train_data(self.age_input)
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        edu_train_data, edu_train_target = self.get_train_data(self.edu_input)
        import numpy as np
        self.my_logger.info('start cross validation')
        clf_name = ['MultinomialNB', 'BernoulliNB']
        cv = ShuffleSplit(n=len(age_train_data), n_iter=10, test_size=0.3, random_state=0)

        age_result_ber = cross_val_score(self.age_ber_nbc, age_train_data, age_train_target, cv=5)
        gender_result_ber = cross_val_score(self.gender_ber_nbc, gender_train_data, gender_train_target, cv=5)
        edu_result_ber = cross_val_score(self.edu_ber_nbc, edu_train_data, edu_train_target, cv=5)

        age_result_mul = cross_val_score(self.age_mul_nbc, age_train_data, age_train_target, cv=5)
        edu_result_mul = cross_val_score(self.edu_mul_nbc, edu_train_data, edu_train_target, cv=5)

        self.my_logger.info('use %s: age:%s' % (clf_name[0], str(age_result_mul)))
        self.my_logger.info('use %s: edu:%s' % (clf_name[0], str(edu_result_mul)))

        self.my_logger.info('use %s: age:%s' % (clf_name[1], str(age_result_ber)))
        self.my_logger.info('use %s: gender:%s' % (clf_name[1], str(gender_result_ber)))
        self.my_logger.info('use %s: edu:%s' % (clf_name[1], str(edu_result_ber)))
        mean_result = []
        for a, g, e in zip(age_result_ber, gender_result_ber, edu_result_ber):
            mean_result.append(np.mean([a, g, e]))
        self.my_logger.info('mean result: %s' % str(mean_result))
        # for age_clf_name, age_clf in zip(clf_name, [self.age_mul_nbc, self.age_ber_nbc]):
        #     for gender_clf_name, gender_clf in zip(clf_name, [self.gender_mul_nbc,
        #                                                       self.gender_ber_nbc]):
        #         for edu_clf_name, edu_clf in zip(clf_name, [self.edu_mul_nbc,
        #                                                     self.edu_ber_nbc]):
        #             self.my_logger.info('start cross validation: age:%s, gender:%s, edu:%s' %
        #                                 (age_clf_name, gender_clf_name, edu_clf_name))
        #             age_result = cross_val_score(age_clf, age_train_data, age_train_target, cv=5)
        #             gender_result = cross_val_score(gender_clf, gender_train_data, gender_train_target, cv=5)
        #             edu_result = cross_val_score(edu_clf, edu_train_data, edu_train_target, cv=5)
        #             self.my_logger.info('use %s: age:%s' % (age_clf_name, str(age_result)))
        #             self.my_logger.info('use %s: gender:%s' % (gender_clf_name, str(gender_result)))
        #             self.my_logger.info('use %s: edu:%s' % (edu_clf_name, str(edu_result)))
        #             mean_result = []
        #             for a, g, e in zip(age_result, gender_result, edu_result):
        #                 mean_result.append(np.mean([a, g, e]))
        #             self.my_logger.info('mean result: %s' % str(mean_result))
        #             mean_result.clear()
        self.my_logger.info('end')

    def test_gender(self):
        gender_train_data, gender_train_target = self.get_train_data(self.gender_input)
        gender_result = cross_val_score(self.gender_gs_nbc, gender_train_data, gender_train_target, cv=5)
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
        self.model_save()
        # self.classify()

if __name__ == '__main__':
    sougou = SougouNBC()
    sougou.validation()

    # sougou.start()