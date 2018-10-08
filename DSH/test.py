from dataloader import test_iter
from network import *
from utils import *

if __name__ == '__main__':
    # load model
    saved_model = 'best_dsh_sg.pth'
    model = get_network().to(DEVICE)
    model.load_state_dict(torch.load(save_dir + saved_model))

    # mean topk 评价标准
    top_k = mean_topk(model, test_iter)
    ##############################################
    top1, top2, top5, top10, mrr = 0, 0, 0, 0, 0
    for tt in top_k:
        if tt == 0:
            top1 += 1
        if tt < 2:
            top2 += 1
        if tt < 5:
            top5 += 1
        if tt < 10:
            top10 += 1
        mrr += 1 / (tt + 1)
    mrr = mrr / len(top_k)
    print('[test] top1: %d, top2: %d, top5: %d, top10: %d, mrr: %.3f' % (top1, top2, top5, top10, mrr))
