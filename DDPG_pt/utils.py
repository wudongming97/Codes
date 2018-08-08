import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(var):
    return var.cpu().data.numpy()


def to_tensor(v_list):
    return list(map(lambda x: torch.tensor(x).float().to(DEVICE), v_list))


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def test_policy(actor, env, vis=False, n_episodes=2, max_len=500):
    returns = []
    for i_episode in range(n_episodes):
        state = env.reset()
        if vis: env.render()
        episode_return = 0
        for t in range(max_len):
            action = actor.action(state)
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if vis: env.render()
            if done:
                returns.append(episode_return)
                break
    return sum(returns) / len(returns)
