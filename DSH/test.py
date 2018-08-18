from utils import *


# test
def test(model, data_iter):
    with torch.no_grad():
        model.eval()
        pb = []
        nb = []
        for pos, neg, _, _ in data_iter:
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)
            pb.append(model(pos))
            nb.append(model(neg, False))

        pb = torch.cat(pb)
        nb = torch.cat(nb)

        pb_len = pb.size(0)
        distance_mx = torch.zeros(pb_len, pb_len, dtype=torch.float, device=DEVICE)
        for i in range(pb_len):
            for j in range(pb_len):
                distance_mx[i][j] = ((pb[i] - nb[j]) ** 2).mean()

        top_k = [0] * pb_len
        for i in range(pb_len):
            for j in range(pb_len):
                if distance_mx[i][j] < distance_mx[i][i]:
                    top_k[i] += 1

        return top_k


# if __name__ == '__main__':
    # saved_model = '20.pth'
# model = get_network().to(DEVICE)
# model.load_state_dict(torch.load(save_dir + saved_model))
# top_k = test(model, test_iter)
# print('[test] top_k: %.3f' % (sum(top_k) / len(top_k)))
# print(",".join(str(i) for i in top_k))
# top_k = test(model, train_as_test_iter)
# print('[Train] top_k: %.3f' % (sum(top_k) / len(top_k)))
