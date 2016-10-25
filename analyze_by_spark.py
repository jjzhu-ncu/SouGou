
import jieba.posseg as pseg
words = pseg.cut('lol英雄联盟')
for word in words:
     print('%s %s' % (word.word, word.flag))
     '''
    '''
