import torch
import pickle
from flax.core.frozen_dict import freeze
import random
import numpy as np
from preprocessor import Preprocessor

def prepare_dataset(data_path, dset_type, env):
    if env == "pusht":
        data_path = data_path + dset_type

        ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
        ACTION_STD = torch.tensor([0.2019, 0.2002])

        action_scale = 100.0

        with open(data_path + "seq_lengths.pkl", "rb") as f:
            seq_lengths = pickle.load(f)

        states = torch.load(data_path + "states.pth")
        vel = torch.load(data_path + "velocities.pth")
        actions = torch.load(data_path + "rel_actions.pth")

        states = states.float()
        actions = actions.float()
        vel = vel.float()

        states_0 = states[0,:seq_lengths[0],:] 
        vel_0 = vel[0,:seq_lengths[0],:]
        actions_0 = actions[0,:seq_lengths[0],:]
        actions_0 = actions_0 / action_scale

        observations = torch.cat([states_0.float(), vel_0.float()], dim=-1)
        actions_dt = (actions_0 - ACTION_MEAN) / ACTION_STD

        terminals = torch.zeros(observations.shape[0], dtype=torch.float32)
        valids = torch.ones(observations.shape[0], dtype=torch.float32)
        terminals[-2:] = 1.0
        valids[-1] = 0.0

        for i in range(1,states.shape[0]):
            seq_len = seq_lengths[i]

            states_0 = states[0,:seq_len,:] 
            vel_0 = vel[0,:seq_len,:]

            actions_0 = actions[0,:seq_len,:]
            actions_0 = actions_0 / action_scale

            observations_i = torch.cat([states_0.float(), vel_0.float()], dim=-1)
            actions_i = (actions_0 - ACTION_MEAN) / ACTION_STD

            observations = torch.cat((observations, observations_i), dim=0)
            actions_dt = torch.cat((actions_dt, actions_i), dim=0)

            terminals_i = torch.zeros(observations_i.shape[0], dtype=torch.float32)
            valids_i = torch.ones(observations_i.shape[0], dtype=torch.float32)
            terminals_i[-2:] = 1.0
            valids_i[-1] = 0.0

            terminals = torch.cat((terminals, terminals_i), dim=0)
            valids = torch.cat((valids, valids_i), dim=0)
        
        print(observations.shape)
        print(actions_dt.shape)

        data_dict = {
            'terminals': terminals.detach().cpu().numpy(),
            'valids': valids.detach().cpu().numpy(),
            'observations': observations.detach().cpu().numpy(),
            'actions': actions_dt.detach().cpu().numpy()
        }
        return freeze(data_dict) 
    else:
        raise ValueError("Environment not supported")




def sample_traj_segment_from_dset(traj_len,n_evals,dset):
    states = []
    actions = []

    arr = dset['valids']
    arr=np.insert(arr, 0, 0)
    
    zero_indices = [i for i, x in enumerate(arr) if x == 0]

    distances_with_index = [
        [zero_indices[i+1] - zero_indices[i], zero_indices[i]] 
        for i in range(len(zero_indices) - 1)
    ]


    valid_traj = [
        distances_with_index[i][1]
        for i in range(len(distances_with_index))
        if distances_with_index[i][0] >= traj_len
    ]
    if len(valid_traj) == 0:
        raise ValueError("No trajectory in the dataset is long enough.")

    for _ in range(n_evals):

        traj_id = np.random.randint(low=0, high=len(distances_with_index), dtype=np.int32)
        max_offset = distances_with_index[traj_id][0] - traj_len

        offset = np.random.randint(low=0, high=max_offset, dtype=np.int32)
        index = distances_with_index[traj_id][1]
        state = dset['observations'][index+offset : index+offset + traj_len]
        act = dset['actions'][index+offset : index+offset + traj_len-1]
        act = torch.from_numpy(act)
        actions.append(act)
        states.append(state)

    return states, actions


def create_preprocessor(env):
    if env == "pusht":
        ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
        ACTION_STD = torch.tensor([0.2019, 0.2002])
        STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
        STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
        PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
        PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])
        data_preprocessor = Preprocessor(
                action_mean=ACTION_MEAN,
                action_std=ACTION_STD,
                state_mean=STATE_MEAN,
                state_std=STATE_STD,
                proprio_mean=PROPRIO_MEAN,
                proprio_std=PROPRIO_STD,
                transform="datasets.img_transforms.default_transform",
            )
        return data_preprocessor
    else:
        raise ValueError("Environment not supported")
    



def sample_goal_state_actor(train_dataset,val_dataset):
    train_or_val = random.randint(0, 1)
    if train_or_val == 0:
        goal_state = train_dataset['observations'][random.randint(0, len(train_dataset['observations']) - 1)]
    else:
        goal_state = val_dataset['observations'][random.randint(0, len(val_dataset['observations']) - 1)]
    return goal_state