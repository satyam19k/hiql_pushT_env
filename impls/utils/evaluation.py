import jax
import numpy as np
from utils_dw import save_video_from_rollout
def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id,
    task_list,
    goal_state_actor,
    seeds,
    save_path,
    data_preprocessor,
    eval_temperature=0,
    
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    '''Taking intial and final state from task'''

    obs_0 = task_list[task_id-1]["obs_0"]
    state_0 = task_list[task_id-1]["state_0"]
    obs_g = task_list[task_id-1]["obs_g"]
    state_g = task_list[task_id-1]["state_g"]


    obs_0, state_0 = env.prepare([seeds[task_id-1]], [state_0])
    _,_ = env.reset()

    '''Setting goal_pose for the env using last state of the task '''
    env.set_task_goal([state_g[2:5]])
    
    done = False
    step = 0
    max_steps = 50
    state = state_0[0]
    e_visual_rollout=[]

    while not done and step < max_steps:
        action = actor_fn(observations=state, goals=goal_state_actor, temperature=eval_temperature)
        action = np.expand_dims(action, axis=0)  
        next_obs_visual, reward, done,_, info = env.step_once(action)
        '''Updating state for next step'''
        state = info["state"]  
        e_visual_rollout.append(next_obs_visual["visual"])
        step += 1

    # Evaluate final state
    eval_results = env.eval_state([state_g], [state])
    successes = eval_results['success']

    visual_dists = np.linalg.norm(next_obs_visual["visual"] - obs_g["visual"], axis=1)
    mean_visual_dist = np.mean(visual_dists)

    eval_results['visual_dists'] = mean_visual_dist

    e_visuals = [e_visual_rollout]

    e_visuals = data_preprocessor.transform_obs_visual(e_visuals)


    e_visuals = e_visuals[: 1]
    goal_visual = [obs_g["visual"][: 1]]
    goal_visual = data_preprocessor.transform_obs_visual(goal_visual)

    correction = 0.3
    
    '''Saving video and image of the rollout'''
    save_video_from_rollout(e_visuals,goal_visual,successes,correction,task_id,save_path)


    return eval_results



