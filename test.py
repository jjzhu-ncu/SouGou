__author__ = 'jjzhu'
import jieba
import jieba.posseg as pseg


def prepare_for_word2vec():
    train_file = open('./sougou_result_seg.csv', encoding='utf-8')
    to_file = open('gbk_sougou_result_seg.csv', encoding='gbk', mode='w')
    for line in train_file:
        line_ = line.encode('utf-8').decode('gbk')
        split_field = line_.split(' ')
        new_line = split_field[0] + ' ' + split_field[1].split('_')[1] + ' ' +\
                   split_field[2] + ' ' + split_field[3].split('_')[1]
        to_file.write(new_line)
    to_file.flush()
    to_file.close()
    train_file.close()


def clear_data():
    exception_dict = {'1': ['1', '2', '3'], '2': ['1']}

    source_file = open('./data/user_tag_query.2W.TRAIN', 'r', encoding='utf-8')
    for line in source_file:
        split_r = line.strip().split('\t')
        if split_r[1] == '0' or split_r[2] == '0' or split_r[3] == '0':
            # print(line)
            continue
        if split_r[3] in exception_dict.get(split_r[1], []):
            print(line)


def load_stop_word():
    stop_words_ = []
    with open('./extra_dict/stop_words_ch.txt', 'r', encoding='utf-8') as s_w_file:
        for line in s_w_file:
            stop_words_.append(line.strip())
    return stop_words_


def prepare_train_data():
    train_f_n = './data/user_tag_query.2W.TRAIN'
    target = './data/2W.TRAIN.PRO.NO.SEG.jieba'
    unsuccessful = './data/2W.TRAIN.UNS.NO.SEG.jieba'

    stop_words = load_stop_word()
    temp_list = []
    unused_context = []
    rewrite_context = []
    with open(train_f_n, mode='r', encoding='utf-8') as train_file:
            for line in train_file:
                line = line.strip()
                split_r = line.split('\t')
                pesg_words = jieba.cut(','.join(split_r[4:]))
                for pesg_word in pesg_words:
                    if pesg_word not in stop_words \
                            and len(pesg_word) > 1:
                        temp_list.append(str(pesg_word))

                if len(temp_list) == 0:
                    unused_context.append(line+'\n')
                # all_word = ' '.join(temp_list)
                rewrite_context.append(','.join(split_r[:4])+','+' '.join(temp_list) + '\n')
                temp_list.clear()
    open(target, 'w', encoding='utf-8').writelines(rewrite_context)
    if len(unused_context) != 0:
        open(unsuccessful, 'w', encoding='utf-8').writelines(unused_context)

def prepare_train_data_seg():
    stop_words = load_stop_word()
    seg_need = ['n', 'v', 'e', 'j', 'l', 't', 'i', 'b', 's', 'a', 'r', 'd', 'z']
    save_path = './data/2W.TRAIN.pro.seg.jieba'
    unsuccessful = './data/2W.TRAIN.uns.jieba'
    temp_list = []
    unsuccessful_context = []
    save_file = open(save_path, 'w', encoding='utf-8')
    unsuccessful_file = open(unsuccessful, 'w', encoding='utf-8')
    with open('./data/user_tag_query.2W.TRAIN', mode='r', encoding='utf-8') as train_file:
            for line in train_file:
                line = line.strip()
                split_r = line.split('\t')
                pesg_words = pseg.cut(','.join(split_r[4:]))
                for pesg_word in pesg_words:
                        if pesg_word.word not in stop_words \
                                and len(pesg_word.word) > 1 \
                                and pesg_word.flag[0] in seg_need:
                            temp_list.append(str(pesg_word.word))
                        else:
                            unsuccessful_context.append('%s %s\n' % (pesg_word.word, pesg_word.flag))
                save_file.writelines([','.join(split_r[:4])+','+' '.join(temp_list)+'\n'])
                unsuccessful_file.writelines(['\t'.join(unsuccessful_context)])
                temp_list.clear()
                unsuccessful_context.clear()
            unsuccessful_file.flush()
            unsuccessful_file.close()
            save_file.flush()
            save_file.close()


def prepare_test_data():
    stop_words = load_stop_word()
    save_path = './data/2W.TEST.pro.jieba'
    unsuccessful = './data/2W.TEST.uns.jieba'
    save_file = open(save_path, 'w', encoding='utf-8')
    unsuccessful_file = open(unsuccessful, 'w', encoding='utf-8')
    temp_list = []
    with open('./data/user_tag_query.2W.TEST', mode='r', encoding='utf-8') as test_file:
            for line in test_file:
                temp_list.clear()
                split_r = line.strip().split('\t')
                words = jieba.cut(','.join(split_r[1:]))
                for w in words:
                    if w not in stop_words and len(w) > 1:  # and len(w) > 1
                        temp_list.append(str(w))
                if len(temp_list) != 0:
                    save_file.writelines([split_r[0]+' '+' '.join(temp_list)+'\n'])
                else:
                    print('-'*100)
                    unsuccessful_file.writelines([line])
    save_file.flush()
    unsuccessful_file.flush()
    save_file.close()
    unsuccessful_file.close()


def delete_blank_line():
    target = './data/sougou/unused_words1.d2.csv'
    no_blank_lines = []
    with open('./data/sougou/unused_words1.d.csv', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) != 0 and len(line.split(' ')[0]) > 1:
                no_blank_lines.append(line)
    open(target, 'w', encoding='utf-8').writelines(no_blank_lines)

import datetime
start = datetime.datetime.now()

prepare_train_data_seg()
end = datetime.datetime.now()
print('%s minute' % str((end-start).seconds/60))


