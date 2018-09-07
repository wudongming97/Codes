from dataloader import test_iter, train_iter, test_16W_iter
from network import *
from utils import *

if __name__ == '__main__':
    # load model
    saved_model = 'best_dsh_nsg.pth'
    model = get_network().to(DEVICE)
    model.load_state_dict(torch.load(save_dir + saved_model))

    # mean topk 评价标准
    top_k = mean_topk(model, test_iter, test_16W_iter)
    print('[test] top_k: %.3f' % (sum(top_k) / len(top_k)))
    print(",".join(str(i) for i in top_k))
    top_k = mean_topk(model, train_iter)
    print('[Train] top_k: %.3f' % (sum(top_k) / len(top_k)))
