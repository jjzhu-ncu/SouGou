# -*- coding: utf-8 -*-
__author__ = 'jjzhu'

import jieba
import jieba.analyse
import gensim.models
import jieba.posseg as pseg

import datetime
start = datetime.datetime.now()
jieba.load_userdict('./extra_dict/dict.txt.big')
train_file = open('./data/user_tag_query.2W.TRAIN', encoding='utf-8')
test_file = open('./data/user_tag_query.2W.TEST', encoding='utf-8')
jieba.analyse.set_idf_path("./extra_dict/idf.txt.big")
jieba.analyse.set_stop_words('./extra_dict/stop_words_ch.txt')
count = 0
age_dict = {0: '未知', 1: '0-18岁', 2: '19-23', 3: '24-30', 4: '31-40', 5: '41-50', 6: '51-999'}
gender_dict = {0: '未知', 1: '男', 2: '女'}
edu_dict = {0: '未知', 1: '博士', 2: '硕士', 3: '大学生', 4: '高中', 5: '初中', 6: '小学'}
age_gender_edu_input = {}
all_input = list()
age_gender_edu_keywords = {}
age_input = {}
gender_input = {}
edu_input = {}
age_keywords = {}
gender_keywords = {}
edu_keywords = {}

def get_stop_words(stop_words_fn):
    with open(stop_words_fn, 'rb') as f:
        stop_words_set = {line.strip('\r\t').decode('utf-8') for line in f}
    return stop_words_set


def sentence2words(sentence, stop_words=False, stop_words_set=None):
    """
    split a sentence into words based on jieba
    """
    # seg_words is a generator
    seg_words = jieba.cut(sentence)
    if stop_words:
        words = [word for word in seg_words if word not in stop_words_set and word != ' ']
    else:
        words = [word for word in seg_words]
    return words


class Sentences(object):
    def __init__(self, train_file):

        stop_words_fn = './extra_dict/stop_words_ch.txt'
        self.stopWords_set = get_stop_words(stop_words_fn)
        self.fns = train_file

    def __iter__(self):
        with open(self.fns) as f:
            for line in f:
                fields = line.split('\t')
                yield sentence2words(','.join(fields[4:]), True, self.stopWords_set)


def prepare_for_word2vec():
    # for line in train_file.readlines():
    #     split_r = line.split('\t')
    #     for w in split_r[4:]:
    #         print(w)
    #         # print([i for i in jieba.cut(w)])
    #         all_input.append([i for i in jieba.cut(w)])
    with open('./test.txt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            print(line)
            k = line.split(' ')[0]
            v = line.split(' ')[1].split(',')
            age_gender_edu_keywords[k] = v


def proc_train_data():
    for line in train_file.readlines():
        split_r = line.split('\t')

        age_input[split_r[1]] = age_input.get(split_r[1], []) + split_r[4:]
        gender_input[split_r[2]] = gender_input.get(split_r[2], []) + split_r[4:]
        edu_input[split_r[3]] = edu_input.get(split_r[3], []) + split_r[4:]
        # for w in split_r[4:]:
        #     all_input.append(jieba.cut(w))
    # result = open('./data/keywords.txt', 'w')
    for input_, keywords_ in zip([age_input, gender_input, edu_input], [age_keywords, gender_keywords,  edu_keywords]):

        for key, value in input_.items():
                sentence = ','.join(value)
                split_k = key.split('_')
                print(','.join(split_k))

                keywords = jieba.analyse.extract_tags(sentence, allowPOS=('ns', 'n', 'vn', 'v'))
                print(','.join(keywords))
                keywords_[key] = keywords
            # keywords = jieba.analyse.textrank(sentence)
            # print('textrank:'+','.join(keywords))


def word_2_vec():
    num_features = 200
    min_word_count = 10
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = gensim.models.Word2Vec(
        all_input,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        sample=sample,
        window=context,
        iter=epoch,
    )
    model.save('word_2_vec_model')
    # model.load()
    return model


def analyze():
    result = {}
    for user in test_file:
        print('*' * 100)
        field = user.split('\t')
        print(field)
        user_id = field[0]
        user_query = ','.join(field[1:])
        print(user_query)

        user_keywords = jieba.analyse.extract_tags(','.join(user_query), allowPOS=('ns', 'n', 'vn', 'v'))
        max_sim = -1
        most_sim = ''
        for k, v in age_gender_edu_keywords.items():
            print(k, v)
            print(user_keywords)
            curr_all_sim = 0
            for a_v in v:
                for u_v in user_keywords:
                    curr_all_sim += m.similarity(a_v, u_v)
            if len(v)*len(user_keywords) == 0:
                continue
            if curr_all_sim/(len(v)*len(user_keywords)) > max_sim:
                most_sim = k
        result[user_id] = most_sim
        split_k = most_sim.split(',')
        print(most_sim)
        if most_sim == '':
            continue
        print(str(user_id)+'\n' +
              str(user_keywords) + '\n' +
              age_dict[int(split_k[0])]+'\n' +
              gender_dict[int(split_k[1])]+'\n' +
              edu_dict[int(split_k[2])]+'\n')


def calc_sim(c_k, d_k):
        max_sim = -1
        most_sim = '0'
        # mid_result_file.write(str(user_id)+'\n')
        for k, v in c_k.items():
            curr_all_sim = 0
            curr_count = 0
            for a_v in v:
                for u_v in d_k:
                    # print('-' * 10)
                    try:
                        sim = m.similarity(a_v, u_v)
                        curr_all_sim += sim
                        curr_count += 1
                    except KeyError:
                        continue
            # mid_result_file.write('\t %s--%f\n' % (k, curr_all_sim/curr_count))

            if curr_count == 0:
                continue
            if curr_all_sim/curr_count > max_sim:
                max_sim = curr_all_sim/curr_count
                most_sim = k
        return most_sim


def test():
    from operator import itemgetter
    result_file = open('./result-1.csv', mode='w', encoding='utf-8')
    # mid_result_file = open('./mid_result.csv', mode='w', encoding='utf-8')
    NUM = 20
    pos_need = ['n', 'v']
    word_counts = {}
    result = {}
    for user in test_file:

        print('*' * 100)
        word_counts.clear()
        field = user.strip().split('\t')
        user_id = field[0]
        query = ','.join(field[1:])
        words = pseg.cut(query)
        for w in words:
            if len(w.word) < 2:
                continue
            if w.flag[:1] in pos_need:
                print(w.word)
                if w.word in word_counts:
                    word_counts[w.word] += 1
                else:
                    word_counts[w.word] = 1

        sorted_word = sorted(word_counts.items(), key=lambda elem: elem[1], reverse=True)
        user_keywords = []
        for i in range(NUM if NUM < len(sorted_word) else len(sorted_word)):
            user_keywords.append(sorted_word[i][0])
        age_most_sim = calc_sim(age_keywords, user_keywords)
        gender_most_sim = calc_sim(gender_keywords, user_keywords)
        edu_most_sim = calc_sim(edu_keywords, user_keywords)
        # mid_result_file.write('\t\t final %s--%s\n' % (str(user_id), most_sim))
        # mid_result_file.flush()
        result_file.write('%s %d %d %d\n' % (str(user_id), int(age_most_sim), int(gender_most_sim), int(edu_most_sim)))
        result_file.flush()

    result_file.close()
    # mid_result_file.close()

proc_train_data()
# print all_input[:3]
# m = word_2_vec()
m = gensim.models.Word2Vec.load('word_2_vec_model')
test()
end = datetime.datetime.now()
print('total cost %d' % (end-start).seconds)

# nrfg nr n nrt ns v nz vn nl nz nt