import os
import pandas as pd
import pickle as pkl
import torch
import numpy as np
from load_dataset import collect_preference_data

from preference_tools.oracle import HumanRewardNetwork, HumanCritic

"""
data_all/
│
├── <participant_id>/                       # Data for each participant
│   ├── [VR, 2D_TD, 2D_FPV]/                # Data for VR modality
│   │   ├── preferred/                      # Preferred trajectories
│   │   │   ├── metrics/                    # Metrics for preferred trajectories
│   │   │   │   ├── <metric_name>           # List of metric values
│   │   │   ├── data/                       # Preferred trajectory data
│   │   │   │   ├── trajectory              # Trajectory data dict
│   │   ├── rejected/                       # Rejected trajectories
│   │   │   ├── metrics/                    # Metrics for rejected trajectories
│   │   │   │   ├── <metric_name>           # List of metric values
│   │   │   ├── data/                       # Rejected trajectory data
│   │   │   │   ├── trajectory              # Trajectory data file
│   │   ├── id                              # List of trajectory IDs
│   │   ├── preference                      # List of user preferences
"""


def convert_user_study_data_to_reward_model_dataset(reward_model, data, modality, participant_skip_ids=[]):
    for participant_key in data:
        if participant_key in participant_skip_ids:
            continue

        participant = data[participant_key]
        for condition_key in participant:
            condition = participant[condition_key]
            preferred_trajectories = condition["preferred"]["data"]
            unpreferred_trajectories = condition["rejected"]["data"]
            n_trajectories = len(preferred_trajectories)
            if modality != condition_key:
                continue
            for i in range(n_trajectories):
                pref_traj = preferred_trajectories[i]
                unpref_traj = unpreferred_trajectories[i]

                pref_obs = []
                for j in range(len(pref_traj["states"])):
                    obs = pref_traj["states"][j]["task_obs"]
                    obs_scan = pref_traj["states"][j]["scan"]
                    act = pref_traj["actions"][j]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    obs_scan = torch.tensor(obs_scan, dtype=torch.float32)
                    act_tensor = torch.tensor(act, dtype=torch.float32)
                    traj_tensor = torch.cat((obs_tensor, obs_scan, act_tensor), dim=0)
                    pref_obs.append(traj_tensor)
                pref_traj = torch.stack(pref_obs)

                unpref_obs = []
                for j in range(len(unpref_traj["states"])):
                    obs = unpref_traj["states"][j]["task_obs"]
                    obs_scan = unpref_traj["states"][j]["scan"]
                    act = unpref_traj["actions"][j]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    obs_scan = torch.tensor(obs_scan, dtype=torch.float32)
                    act_tensor = torch.tensor(act, dtype=torch.float32)
                    traj_tensor = torch.cat((obs_tensor, obs_scan, act_tensor), dim=0)
                    unpref_obs.append(traj_tensor)
                unpref_traj = torch.stack(unpref_obs)

                label = [1,0]
                reward_model.add_pairs(pref_traj, unpref_traj, label)
    return reward_model

if __name__ == "__main__":
    participant_skip_ids = [27]  # removed from dataset due to technical issues during data collection as reported in paper

    user_study_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './resources/dataset/user_feedback_data')
    metrics_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './resources/dataset/dataset_2024_11_25__12_24_23/query_dataset_metrics.csv')

    metrics = ["length", "time", "steps", "speed_average", "angle_accumulation", "turning_points", "min_dist_human", "area_under_path"]
    units = ["m", "s", "\#", "m/s", "rad", "\#", "m", "m$^2$"]

    data = collect_preference_data(user_study_data_path, skip_ids=participant_skip_ids, metrics=metrics, metrics_csv=metrics_csv_path)

    for modality in ["VR", "2D_FPV", "2D_TD"]:
        print("Modality: " + modality)
        reward_model = HumanCritic(
            obs_size=[66],
            action_size=2,
            training_epochs=200,
            hidden_sizes=[256,256,256]
        )
        reward_model = convert_user_study_data_to_reward_model_dataset(reward_model, data, modality, participant_skip_ids)
        reward_model.train()
        reward_model.save_reward_model(os.path.join("resources", "reward_models", "{}-rewardmodel".format(modality)))