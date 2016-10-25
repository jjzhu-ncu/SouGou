__author__ = 'jjzhu'

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import jieba
import os
import random

train_file_name = './data/user_tag_query.2W.TRAIN.small'
test_file_name = './data/user_tag_query.2W.TEST.small'
stop_word_file_name = './extra_dict/stop_words_ch.txt'

def load_stop_word():
    stop_words = []
    with open(stop_word_file_name, 'r', encoding='utf-8') as s_w_file:
        for line in s_w_file:
            stop_words.append(line.strip())
    return stop_words

sw = load_stop_word()

def get_dataset():

    age_input = []
    gender_input = []
    edu_input = []
    temp_list = []
    with open(train_file_name, mode='r', encoding='utf-8') as train_file:
        for line in train_file:
            line = line.strip()
            split_r = line.split('\t')
            words = jieba.cut(','.join(split_r[4:]), HMM=True)
            for w in words:
                if w not in sw:
                    temp_list.append(str(w))
            all_word = ' '.join(temp_list)
            age_input.append((all_word, split_r[1]))
            gender_input.append((all_word, split_r[2]))
            edu_input.append((all_word, split_r[3]))
            temp_list.clear()
    return age_input, gender_input, edu_input


def train_and_test_data(data_):
    filesize = len(data_)  # int(0.7 * len(data_))
    # 训练集和测试集的比例为7:3
    train_data_ = [each[0] for each in data_[:filesize]]
    train_target_ = [each[1] for each in data_[:filesize]]
    # test_data_ = [each[0] for each in data_[filesize:]]
    # test_target_ = [each[1] for each in data_[filesize:]]

    return train_data_, train_target_
age_input, gender_input, edu_input = get_dataset()
age_train_data, age_train_target = train_and_test_data(age_input)
gender_train_data, gender_train_target = train_and_test_data(gender_input)
edu_train_data, edu_train_target = train_and_test_data(edu_input)

age_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
gender_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
edu_nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])

print(age_train_data)
age_nbc.fit(age_train_data, age_train_target)    # 训练我们的多项式模型贝叶斯分类器
print(gender_train_data)
gender_nbc.fit(gender_train_data, gender_train_target)    # 训练我们的多项式模型贝叶斯分类器
print(edu_train_data)
edu_nbc.fit(edu_train_data, edu_train_target)    # 训练我们的多项式模型贝叶斯分类器


def classify():
        pre_data = []
        temp_list = []
        with open(test_file_name, mode='r', encoding='utf-8') as test_file:
            for line in test_file:
                split_r = line.strip().split('\t')
                for w in jieba.cut(split_r[1]):
                    if w not in sw:
                        temp_list.append(str(w))

                    else:
                        pass
                if len(temp_list) > 0:
                    pre_data.append((split_r[0], ' '.join(temp_list)))
                temp_list.clear()
        user_ids = [elem[0] for elem in pre_data]
        input_data = [elem[1] for elem in pre_data]
        return user_ids, input_data

user_ids, input_data = classify()
age_predict = age_nbc.predict(input_data)  # 在测试集上预测结果
gender_predict = gender_nbc.predict(input_data)  # 在测试集上预测结果
edu_predict = edu_nbc.predict(input_data)  # 在测试集上预测结果
count = 0                                      # 统计预测正确的结果个数
print(edu_predict)

# for left, right in zip(predict, test_target):
#       if left == right:
#             count += 1
# print(count/len(test_target))
