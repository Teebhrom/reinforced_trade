import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random

REF_MAX_SCORE = {
    'halfcheetah' : 12135.0,
    'walker2d' : 4592.3,
    'hopper' : 3234.3,
    'stocktrading' : 200
}

REF_MIN_SCORE = {
    'halfcheetah' : -280.178953,
    'walker2d' : 1.629008,
    'hopper' : -20.272305,
    'stocktrading' : 0
}

##REF for stock trading



## calculated from d4rl datasets

D4RL_DATASET_STATS = {
        'halfcheetah-medium-v2': {
                'state_mean':[-0.06845773756504059, 0.016414547339081764, -0.18354906141757965,
                              -0.2762460708618164, -0.34061527252197266, -0.09339715540409088,
                              -0.21321271359920502, -0.0877423882484436, 5.173007488250732,
                              -0.04275195300579071, -0.036108363419771194, 0.14053793251514435,
                              0.060498327016830444, 0.09550975263118744, 0.06739100068807602,
                              0.005627387668937445, 0.013382787816226482
                ],
                'state_std':[0.07472999393939972, 0.3023499846458435, 0.30207309126853943,
                             0.34417077898979187, 0.17619241774082184, 0.507205605506897,
                             0.2567007839679718, 0.3294812738895416, 1.2574149370193481,
                             0.7600541710853577, 1.9800915718078613, 6.565362453460693,
                             7.466367721557617, 4.472222805023193, 10.566964149475098,
                             5.671932697296143, 7.4982590675354
                ]
            },
        'halfcheetah-medium-replay-v2': {
                'state_mean':[-0.12880703806877136, 0.3738119602203369, -0.14995987713336945,
                              -0.23479078710079193, -0.2841278612613678, -0.13096535205841064,
                              -0.20157982409000397, -0.06517726927995682, 3.4768247604370117,
                              -0.02785065770149231, -0.015035249292850494, 0.07697279006242752,
                              0.01266712136566639, 0.027325302362442017, 0.02316424623131752,
                              0.010438721626996994, -0.015839405357837677
                ],
                'state_std':[0.17019015550613403, 1.284424901008606, 0.33442774415016174,
                             0.3672759234905243, 0.26092398166656494, 0.4784106910228729,
                             0.3181420564651489, 0.33552637696266174, 2.0931615829467773,
                             0.8037433624267578, 1.9044333696365356, 6.573209762573242,
                             7.572863578796387, 5.069749355316162, 9.10555362701416,
                             6.085654258728027, 7.25300407409668
                ]
            },
        'halfcheetah-medium-expert-v2': {
                'state_mean':[-0.05667462572455406, 0.024369969964027405, -0.061670560389757156,
                              -0.22351515293121338, -0.2675151228904724, -0.07545716315507889,
                              -0.05809682980179787, -0.027675075456500053, 8.110626220703125,
                              -0.06136331334710121, -0.17986927926540375, 0.25175222754478455,
                              0.24186332523822784, 0.2519369423389435, 0.5879552960395813,
                              -0.24090635776519775, -0.030184272676706314
                ],
                'state_std':[0.06103534251451492, 0.36054104566574097, 0.45544400811195374,
                             0.38476887345314026, 0.2218363732099533, 0.5667523741722107,
                             0.3196682929992676, 0.2852923572063446, 3.443821907043457,
                             0.6728139519691467, 1.8616976737976074, 9.575807571411133,
                             10.029894828796387, 5.903450012207031, 12.128185272216797,
                             6.4811787605285645, 6.378620147705078
                ]
            },
        'walker2d-medium-v2': {
                'state_mean':[1.218966007232666, 0.14163373410701752, -0.03704913705587387,
                              -0.13814310729503632, 0.5138224363327026, -0.04719110205769539,
                              -0.47288352251052856, 0.042254164814949036, 2.3948874473571777,
                              -0.03143199160695076, 0.04466355964541435, -0.023907244205474854,
                              -0.1013401448726654, 0.09090937674045563, -0.004192637279629707,
                              -0.12120571732521057, -0.5497063994407654
                ],
                'state_std':[0.12311358004808426, 0.3241879940032959, 0.11456084251403809,
                             0.2623065710067749, 0.5640279054641724, 0.2271878570318222,
                             0.3837319612503052, 0.7373676896095276, 1.2387926578521729,
                             0.798020601272583, 1.5664079189300537, 1.8092705011367798,
                             3.025604248046875, 4.062486171722412, 1.4586567878723145,
                             3.7445690631866455, 5.5851287841796875
                ]
            },
        'walker2d-medium-replay-v2': {
                'state_mean':[1.209364652633667, 0.13264022767543793, -0.14371201395988464,
                              -0.2046516090631485, 0.5577612519264221, -0.03231537342071533,
                              -0.2784661054611206, 0.19130706787109375, 1.4701707363128662,
                              -0.12504704296588898, 0.0564953051507473, -0.09991033375263214,
                              -0.340340256690979, 0.03546293452382088, -0.08934258669614792,
                              -0.2992438077926636, -0.5984178185462952
                ],
                'state_std':[0.11929835379123688, 0.3562574088573456, 0.25852200388908386,
                             0.42075422406196594, 0.5202291011810303, 0.15685082972049713,
                             0.36770978569984436, 0.7161387801170349, 1.3763766288757324,
                             0.8632221817970276, 2.6364643573760986, 3.0134117603302,
                             3.720684051513672, 4.867283821105957, 2.6681625843048096,
                             3.845186948776245, 5.4768385887146
                ]
            },
        'walker2d-medium-expert-v2': {
                'state_mean':[1.2294334173202515, 0.16869689524173737, -0.07089081406593323,
                              -0.16197483241558075, 0.37101927399635315, -0.012209027074277401,
                              -0.42461398243904114, 0.18986578285694122, 3.162475109100342,
                              -0.018092676997184753, 0.03496946766972542, -0.013921679928898811,
                              -0.05937029421329498, -0.19549426436424255, -0.0019200450042262673,
                              -0.062483321875333786, -0.27366524934768677
                ],
                'state_std':[0.09932824969291687, 0.25981399416923523, 0.15062759816646576,
                             0.24249176681041718, 0.6758718490600586, 0.1650741547346115,
                             0.38140663504600525, 0.6962361335754395, 1.3501490354537964,
                             0.7641991376876831, 1.534574270248413, 2.1785972118377686,
                             3.276582717895508, 4.766193866729736, 1.1716983318328857,
                             4.039782524108887, 5.891613960266113
                ]
            },
        'hopper-medium-v2': {
                'state_mean':[1.311279058456421, -0.08469521254301071, -0.5382719039916992,
                              -0.07201576232910156, 0.04932365566492081, 2.1066856384277344,
                              -0.15017354488372803, 0.008783451281487942, -0.2848185896873474,
                              -0.18540096282958984, -0.28461286425590515
                ],
                'state_std':[0.17790751159191132, 0.05444620922207832, 0.21297138929367065,
                             0.14530418813228607, 0.6124444007873535, 0.8517446517944336,
                             1.4515252113342285, 0.6751695871353149, 1.5362390279769897,
                             1.616074562072754, 5.607253551483154
                ]
            },
        'hopper-medium-replay-v2': {
                'state_mean':[1.2305138111114502, -0.04371410980820656, -0.44542956352233887,
                              -0.09370097517967224, 0.09094487875699997, 1.3694725036621094,
                              -0.19992674887180328, -0.022861352190375328, -0.5287045240402222,
                              -0.14465883374214172, -0.19652697443962097
                ],
                'state_std':[0.1756512075662613, 0.0636928603053093, 0.3438323438167572,
                             0.19566889107227325, 0.5547984838485718, 1.051029920578003,
                             1.158307671546936, 0.7963128685951233, 1.4802359342575073,
                             1.6540331840515137, 5.108601093292236
                ]
            },
        'hopper-medium-expert-v2': {
                'state_mean':[1.3293815851211548, -0.09836531430482864, -0.5444297790527344,
                              -0.10201650857925415, 0.02277466468513012, 2.3577215671539307,
                              -0.06349576264619827, -0.00374026270583272, -0.1766270101070404,
                              -0.11862941086292267, -0.12097819894552231
                ],
                'state_std':[0.17012375593185425, 0.05159067362546921, 0.18141433596611023,
                             0.16430604457855225, 0.6023368239402771, 0.7737284898757935,
                             1.4986555576324463, 0.7483318448066711, 1.7953159809112549,
                             2.0530025959014893, 5.725032806396484
                ]
            },
    }



def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)

            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()[0]
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg


                # Note that DT refer last context_len of history to infer next action
                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()


                running_state, running_reward, done, _, _ = env.step(act.cpu().numpy())

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results

def get_total_asset_list(model, device, context_len, env, rtg_target, rtg_scale,
                    max_test_ep_len=1000,
                    state_mean=None, state_std=None):

    eval_batch_size = 1  # required for forward pass
    
    initial_asset = 1e6
    reward_scale = 2**11

    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    
    total_asset_list = [initial_asset]

    with torch.no_grad():
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)

            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()[0]
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg


                # Note that DT refer last context_len of history to infer next action
                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()


                running_state, running_reward, done, _, _ = env.step(act.cpu().numpy())
                
                
                total_asset = initial_asset + running_reward * reward_scale 
                total_asset_list.append(total_asset)
                initial_asset = total_asset
                if done:
                    break
                
    return total_asset_list

class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # # normalize states
        # for traj in self.trajectories:
        #     traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std


    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask
