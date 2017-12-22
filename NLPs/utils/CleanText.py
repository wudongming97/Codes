import unidecode

class T_:
    keep_line_lens = [5, 100]
    keep_word_lens = [5, 25]



f1 = lambda x: len(x) >= T_.keep_line_lens[0] and len(x) < T_.keep_line_lens[1] # 保留指定长度字符的行
f2 = lambda x: len(x.split()) >= T_.keep_word_lens[0] and len(x.split()) < T_.keep_word_lens[1] #保留指定token个数范围的行

F_ = [f1, f2]

if __name__ == '__main__':
    raw_file = '../data/europarl-v7.en'
    target_file = '../data/europarl-v7.txt'
    with open(raw_file,'r',encoding='UTF-8') as sf:
        with open(target_file,'a') as tf:
            for line in sf:
                line = unidecode.unidecode(line)
                if f1(line) and f2(line):
                    tf.write(line)





