import gym
import time
import numpy as np
import multiprocessing as mp

N_KID = 10
N_GENERATION = 5000
LR = 0.05
SIGMA = 0.05
N_CORE = mp.cpu_count() - 1  # number of processing

CONFIG = [
    # n_feature, n_action means observations and actions in gym, eval_threshold ?
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][1]  # choose a game


def sign(k_id):
    return -1 if k_id % 2 == 0 else 1


def build_net():
    def linear(n_in, n_out):
        w = np.random.randn(n_in * n_out).astype(np.float32) * 0.1  # why multiply 0.1?
        b = np.random.randn(n_out).astype(np.float32) * 0.1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 30)
    s1, p1 = linear(30, 20)
    s2, p2 = linear(20, CONFIG['n_action'])
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


class SGD(object):
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum  # the momentum method

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1 - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):
    pa, start = [], 0
    for i, shape in enumerate(shapes):  # may be there is a simple way
        n_w, n_b = shape[0] * shape[1], shape[1]
        pa = pa + [params[start:start + n_w].reshape(shape),
                   params[start + n_w:start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return pa


def get_action(params, x, continuous_a):  # need to test manually
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not continuous_a[0]:
        return np.argmax(x, axis=1)[0]
    else:
        return continuous_a[1] * np.tanh(x)[0]


def get_reward(shapes, params, env, ep_max_step, continuous_a, seed_and_id=None,):  # excuse me ?
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    pa = params_reshape(shapes, params)  # could be deleted?
    # run episode
    s = env.reset()
    ep_r = 0
    for step in range(ep_max_step):
        a = get_action(pa, s, continuous_a)
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky ???
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1:
            r = 0
        ep_r += r
        if done:
            break
    return ep_r


def train(net_shapes, net_params, optimizer, utility, pool):
    # random seeds are used to adjust parallel work
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)  # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID * 2)]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]  # need to test manually

    # update distribution
    cumulative_update = np.zeros_like(net_params)  # need to test manually
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)

    gradients = optimizer.get_gradients(cumulative_update/(2 * N_KID * SIGMA))
    return net_params + gradients, rewards


if __name__ == "__main__":
    # fitness shaping, similar to wi in CMA-ES
    base = N_KID * 2
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    net_shapes, net_params = build_net()
    env = gym.make(CONFIG['game']).unwrapped
    optimizer = SGD(net_params, LR)

    # multi-processing
    pool = mp.Pool(processes=N_CORE)
    mar = None
    for g in range(N_GENERATION):
        t0 = time.time()
        net_params, kid_rewards = train(net_shapes, net_params, optimizer, utility, pool)  # shapes NEAT

        # get the reward after evolution
        net_r = get_reward(net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None, )
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r
        print(
            'Gen: ', g,
            '| Net_R: %.1f' % mar,
            '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            '| Gen_T: %.2f' % (time.time() - t0), )  # excuse me ?
        if mar >= CONFIG['eval_threshold']:
            break

    # test
    print("\nTESTING...")
    p = params_reshape(net_shapes, net_params)
    while True:
        s = env.reset()
        for _ in range(CONFIG['ep_max_step']):
            env.render()
            a = get_action(p, s, CONFIG['continuous_a'])
            s, _, done, _ = env.step(a)
            if done:
                break
