
from env.venv import SubprocVectorEnv
import gym
import numpy as np
import torch
from dataset_utils import create_preprocessor,sample_traj_segment_from_dset

def prepare_tasks(dset,traj_len,n_evals,eval_seed,env_name):
    env = SubprocVectorEnv(
                [
                    lambda: gym.make(
                        "pusht",100, {'with_velocity': True, 'with_target': True}
                    )
                    for _ in range(n_evals)
                ]
            )
    
    
    data_preprocessor = create_preprocessor(env_name)
    
    '''Sampling random trajectory from the dataset'''
    states, actions = sample_traj_segment_from_dset(traj_len, n_evals, dset)

    init_state = [x[0] for x in states]
    init_state = np.array(init_state)

    obs_0,state_0 = env.prepare(eval_seed, init_state)

    actions = torch.stack(actions)
    exec_actions = data_preprocessor.denormalize_actions(actions)

    rollout_obses_gt, rollout_states_gt = env.rollout(eval_seed, init_state, exec_actions.numpy())

    obs_0 = {
    key: np.expand_dims(arr[:, 0], axis=1)
    for key, arr in rollout_obses_gt.items()
}
    obs_g = {
        key: np.expand_dims(arr[:, -1], axis=1)
        for key, arr in rollout_obses_gt.items()
    }
    state_0 = init_state  # (b, d)
    state_g = rollout_states_gt[:, -1]


    task_list = []
    for i in range(n_evals):
        task_list.append({
            "obs_0": {key: obs_0[key][i] for key in obs_0},
            "state_0": state_0[i],
            "obs_g": {key: obs_g[key][i] for key in obs_g},
            "state_g": state_g[i]})
    return task_list