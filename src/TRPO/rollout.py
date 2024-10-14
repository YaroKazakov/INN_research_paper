import numpy as np
import torch
import scipy.signal

def TakeFlatParamsFrom(model):
    params = [torch.ravel(param.detach()) for param in model.parameters()]
    flat_params = torch.cat(params)
    return flat_params

def SetFlatParamsTo(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.shape)))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].reshape(param.shape)
        )
        prev_ind += flat_size

def GetCumulativeReturns(r, gamma=1):
    """
    Computes cumulative discounted rewards given immediate rewards
    G_i = r_i + gamma*r_{i+1} + gamma^2*r_{i+2} + ...
    Also known as R(s,a).
    """
    r = np.array(r)
    #assert r.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], r[::-1], axis=0)[::-1]

def rollout(env, agent, max_pathlength=2500, n_timesteps=50000):
    """
    Generate rollouts.
    :param: env - environment in which we will make actions to generate rollouts.
    :param: act - the function that can return policy and action given observation.
    :param: max_pathlength - maximum size of one path that we generate.
    :param: n_timesteps - total sum of sizes of all pathes we generate.
    """
    paths = []
    n_actions = env.action_space.n

    total_timesteps = 0
    while total_timesteps < n_timesteps:
        obervations, actions, rewards, action_probs = [], [], [], []
        obervation, _ = env.reset()
        for _ in range(max_pathlength):
            action, policy = agent.act(n_actions,obervation)
            obervations.append(obervation)
            actions.append(action)
            action_probs.append(policy)
            obervation, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total_timesteps += 1
            if terminated or truncated or total_timesteps >= n_timesteps:
                path = {
                    "observations": np.array(obervations),
                    "policy": np.array(action_probs),
                    "actions": np.array(actions),
                    "rewards": np.array(rewards),
                    "cumulative_returns": GetCumulativeReturns(rewards),
                }
                paths.append(path)
                break
    return paths


def get_loss(agent, observations, actions, cumulative_returns, old_probs):
    """
    Computes TRPO objective
    :param: observations - batch of observations [timesteps x state_shape]
    :param: actions - batch of actions [timesteps]
    :param: cumulative_returns - batch of cumulative returns [timesteps]
    :param: old_probs - batch of probabilities computed by old network [timesteps x num_actions]
    :returns: scalar value of the objective function
    """
    batch_size = observations.shape[0]
    probs_all = agent.get_probs(observations)

    probs_for_actions = probs_all[torch.arange(batch_size), actions]
    old_probs_for_actions = old_probs[torch.arange(batch_size), actions]

    # Compute surrogate loss, aka importance-sampled policy gradient

    loss = - torch.mean(cumulative_returns * probs_for_actions / old_probs_for_actions)

    assert loss.ndim == 0
    return loss


def get_kl(agent, observations, actions, cumulative_returns, old_probs):
    """
    Computes KL-divergence between network policy and old policy
    :param: observations - batch of observations [timesteps x state_shape]
    :param: actions - batch of actions [timesteps]
    :param: cumulative_returns - batch of cumulative returns [timesteps] (we don't need it actually)
    :param: old_probs - batch of probabilities computed by old network [timesteps x num_actions]
    :returns: scalar value of the KL-divergence
    """
    batch_size = observations.shape[0]
    log_probs_all = agent.get_log_probs(observations)
    old_log_probs = torch.log(old_probs + 1e-10)

    kl = torch.sum(old_probs*(old_log_probs - log_probs_all))/batch_size

    assert kl.ndim == 0
    assert (kl > -0.0001).all() and (kl < 10000).all()
    return kl

def get_entropy(agent, observations):
    """
    Computes entropy of the network policy
    :param: observations - batch of observations
    :returns: scalar value of the entropy
    """

    observations = torch.tensor(observations, dtype=torch.float32)

    log_probs_all = agent.get_log_probs(observations)
    probs_all = torch.exp(log_probs_all)

    entropy = (-probs_all * log_probs_all).sum(dim=1).mean(dim=0)

    assert entropy.ndim == 0
    return entropy

def linesearch(f, x: torch.Tensor, fullstep: torch.Tensor, max_kl: float, max_backtracks: int = 10, backtrack_coef: float = 0.5):
    """
    Linesearch finds the best parameters of neural networks in the direction of fullstep contrainted by KL divergence.
    :param: f - function that returns loss, kl and arbitrary third component.
    :param: x - old parameters of neural network.
    :param: fullstep - direction in which we make search.
    :param: max_kl - constraint of KL divergence.
    :returns:
    """
    loss, _, = f(x)
    for stepfrac in backtrack_coef**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_loss, kl = f(xnew)
        if kl <= max_kl and new_loss < loss:
            x = xnew
            loss = new_loss
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    This method solves system of equation Ax=b using an iterative method called conjugate gradients
    :f_Ax: function that returns Ax
    :b: targets for Ax
    :cg_iters: how many iterations this method should do
    :residual_tol: epsilon for stability
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.sum(r*r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.sum(p*z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = torch.sum(r*r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def f_Ax(x):
    return torch.ravel(torch.tensor(A, dtype=torch.float32) @ x.reshape(-1, 1))

def update_step(agent, observations, actions, cumulative_returns, old_probs, max_kl):
    """
    This function does the TRPO update step
    :param: observations - batch of observations
    :param: actions - batch of actions
    :param: cumulative_returns - batch of cumulative returns
    :param: old_probs - batch of probabilities computed by old network
    :param: max_kl - controls how big KL divergence may be between old and new policy every step.
    :returns: KL between new and old policies and the value of the loss function.
    """

    # Here we prepare the information
    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
    old_probs = torch.tensor(old_probs, dtype=torch.float32)

    # Here we compute gradient of the loss function
    loss = get_loss(agent, observations, actions, cumulative_returns, old_probs)
    grads = torch.autograd.grad(loss, agent.parameters())
    loss_grad = torch.cat([torch.ravel(grad.detach()) for grad in grads])

    def Fvp(v):
        # Here we calculate Fx to solve Fx = g using conjugate gradients
        # In fact, we use several techniques to efficiently calculate

        kl = get_kl(agent, observations, actions, cumulative_returns, old_probs)

        grads = torch.autograd.grad(kl, agent.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.reshape(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, agent.parameters())
        flat_grad_grad_kl = torch.cat([torch.ravel(grad) for grad in grads]).detach()

        return flat_grad_grad_kl + v * 0.1

    def get_loss_kl(params):
        # Helper for linear search
        SetFlatParamsTo(agent, params)
        return [
            get_loss(agent, observations, actions, cumulative_returns, old_probs),
            get_kl(agent, observations, actions, cumulative_returns, old_probs),
        ]

    # Here we solve the Fx=g system using conjugate gradients
    stepdir = conjugate_gradient(Fvp, -loss_grad, 10)

    # Here we compute the initial vector to do linear search
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    # Here we get the start point
    prev_params = TakeFlatParamsFrom(agent)

    # Here we find our new parameters
    new_params = linesearch(get_loss_kl, prev_params, fullstep, max_kl)
    # And we set it to our network
    SetFlatParamsTo(agent, new_params)

    return get_loss_kl(new_params)

