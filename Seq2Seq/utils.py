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


def tensor_to_sentences(idxs, itos):
    idxs = idxs.detach().cpu().numpy()
    batch_size = idxs.shape[1]
    seq_len = idxs.shape[0]
    outputs = []
    for batch in range(batch_size):
        one_batch = []
        for seq in range(seq_len):
            ss = itos[idxs[seq, batch]]
            if ss == '<s>':
                continue
            if ss == '</s>':
                break
            one_batch.append(ss)
        one_batch = ' '.join(one_batch)
        outputs.append(one_batch)

    return outputs


if __name__ == '__main__':
    file = '../../Datasets/WMT17/neu2017/NEU_cn.txt'
    split_file_by_lines(file, 200000, 200000)
    file = '../../Datasets/WMT17/neu2017/NEU_en.txt'
    split_file_by_lines(file, 200000, 200000)
