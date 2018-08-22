import torch as T

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')


def split_file_by_lines(f_name, *args):
    cur_line = 0
    i = 0
    with open(f_name) as f:
        lines = f.readlines()
        L = len(lines)
        assert sum(args) <= L
        for arg in args:
            with open(f_name + str(i), 'w') as wf:
                wf.writelines(lines[cur_line: cur_line + arg])
            cur_line += arg
            i += 1

        with open(f_name + str(i), 'w') as wf:
            wf.writelines(lines[cur_line: L])


if __name__ == '__main__':
    file = '../../Datasets/WMT17/neu2017/NEU_cn.txt'
    split_file_by_lines(file, 100000, 50000)
    file = '../../Datasets/WMT17/neu2017/NEU_en.txt'
    split_file_by_lines(file, 100000, 50000)

    # reference = 'Two person be in a small race car drive by a green hill .'
    # output = 'Two person be in race uniform in a street car .'
    # bleu([reference], [output])
