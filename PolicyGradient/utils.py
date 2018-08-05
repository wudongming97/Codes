import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(v_list):
    return list(map(lambda x: torch.tensor(x).to(DEVICE), v_list))


def test_policy(net, env, vis=False):
    with torch.no_grad():
        rewards = []
        state = env.reset()
        if vis: env.render()
        is_done = False
        while not is_done:
            dist, _ = net(state)
            action = dist.sample()
            state, reward, is_done, _ = env.step(action.item())
            if vis: env.render()
            rewards.append(reward)
    return sum(rewards)
