# -*- encoding: utf-8 -*-
__author__ = 'jjzhu'


import chardet
import re
import jieba
import gensim
import datetime
# file = open("D:\\GoogleDownload\\news_tensite_xml.full\\news_tensite_xml.dat", "rb")
count = 0
pattern = re.compile(u'<content>(.*?)</content>')
start = datetime.datetime.now()
end = datetime.datetime.now()
content_re = re.compile(u'<content>(.*?)</content>')
stop_words_fn = './extra_dict/stop_words_ch.txt'


def precess(sentence):
    if sentence.strip() == '':
        return ''
    try:
        code = chardet.detect(sentence)['encoding']
        return sentence.decode(code if code is not None else 'GB2312')
    except UnicodeDecodeError:
        return sentence.decode('GB18030')


def get_stop_words(stop_words_fn):
    '''
    :param stop_words_fn:
    :return:
    '''
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


class MySentences(object):
    def __init__(self, train_file):
        stop_words_fn = './extra_dict/stop_words_ch.txt'
        start = datetime.datetime.now()
        self.stopWords_set = get_stop_words(stop_words_fn)
        end = datetime.datetime.now()
        print('load stop words cost %ds' % (end-start).seconds)
        self.pattern = re.compile(u'<content>(.*?)</content>')

        self.fns = train_file

        print('load train data cost %ds' % (end-start).seconds)

    def __iter__(self):
            with open(self.fns, 'r') as f:
                for line in f:
                    if line.startswith('<content>'):
                        content = self.pattern.findall(line)
                        if len(content) != 0:
                            process_sec = precess(content[0].strip())
                            print(process_sec)
                            yield sentence2words(process_sec, True, self.stopWords_set)
def get_sentence(path):
    stop_words_set = get_stop_words(stop_words_fn)
    sentences = []
    with open(path, 'r') as f:
                for line in f:
                    if line.startswith('<content>'):
                        content = content_re.findall(line)
                        if len(content) != 0:
                            process_sec = precess(content[0].strip())
                            print(process_sec)
                            sentences.append(sentence2words(process_sec, True, stop_words_set))
    return sentences


def train_save(list_csv, model_fn):
    start = datetime.datetime.now()
    sentences = get_sentence(list_csv)
    end = datetime.datetime.now()
    print('load sentence cost %ds' % (end-start).seconds)
    num_features = 200
    min_word_count = 10
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = gensim.models.Word2Vec(
        sentences,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        sample=sample,
        window=context,
        iter=epoch,
    )
    model.save(model_fn)
    model.load()
    return model


def main():
    model = train_save('./extra_dict/train.dat', 'word2vec_model_CA')

    # get the word vector
    for w in model.most_similar(u'互联网'):
        print w[0], w[1]

    print model.similarity(u'网络', u'互联网')

    country_vec = model[u"国家"]
    print country_vec


def test():
    c = 0
    with open('./extra_dict/train.dat') as fi:
        for line in fi:
            print precess(line)
            if c > 1000:
                break
            c += 1
if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print('total cost %d s' % (end - start).seconds)