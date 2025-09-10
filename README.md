# rlhf_prefnav_interface_study
Code repository for the publication "The Impact of VR and 2D Interfaces on Human Feedback in Preference-Based Robot Learning" by Jorge de Heuvel, Daniel Marta, Simon Holk, Iolanda Leite, and Maren Bennewitz, in Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

## Setup  

### Prerequisites  
- In order to use the script `setup.sh`, [Conda](https://docs.conda.io/en/latest/) must be installed.  

### Steps  
1. Simply create a new conda environment and install all dependencies by running:  
    ```bash
    bash setup.sh
    ```
2. Download the dataset: https://cloud.vi.cs.uni-bonn.de/index.php/s/3CMM2ZA2YCLgpLw
3. Unzip in `./resources/dataset`. See below for folder structure.

### Setup on Mac OS
During the time of code upload, a compiler issue due to an updated clang version on MacOS prevented installation of igibson and pybullet.
The issue is described here: https://github.com/bulletphysics/bullet3/issues/4712
If available, use a Linux system.

## Project Files To Get Started
- **load_dataset.py:** Collects and processes user preference data from the study, organizing it by modality (VR, 2D top-down, 2D first-person view).
- **train_reward_model.py:** Trains the reward models from the dataset.
- **train_policy_aligned.py:** Is a training script to train the navigation policies. 
Needs to be called with the configuration file for the corresponding policy, such as
  - VR: `python train_policy_aligned.py --yaml 25_02_28__alignment_vr.yml`
  - 2D_FPV: `python train_policy_aligned.py --yaml 25_02_28__alignment_fpv.yml`
  - 2D_TD: `python train_policy_aligned.py --yaml 25_02_28__alignment_td.yml`
  - BL: `python train_policy_aligned.py --yaml 25_02_28__alignment_bl.yml`

## Reward Models
The trained reward models used for the experimental results of the paper can be found in `./resources/reward_models`.

## Dataset Folder Structure
This dataset in `dataset_<date>/` contains human preference queries for robot navigation collected across three modalities: Virtual Reality (VR), 2D Top-Down (2D_TD), and 2D First-Person View (2D_FPV). Each trial presents two trajectory options (Query A and Query B), with corresponding data files for robot states, reinforcement learning rollouts, and video recordings.

The `user_feedback_data/` folder contains anonymized participant information and feedback logs from the study. 
It records the randomized query order and user preferences across the three interface modalities: VR, 2D Top-Down (2D_TD), and 2D First-Person View (2D_FPV).

```
dataset_<date>/
│
├── Trial_<id>/                             # Each trial corresponds to one preference query
│   ├── Query_A_0_Robot.pkl                 # Robot state information for Query A
│   ├── Query_A_state_action_reward.hdf5    # RL rollout data for Query A (state, action, reward)
│   ├── Query_A_state_action_reward.pkl     # Pickle version of rollout data for Query A
│   ├── Query_A_fpv_video.mp4               # First-person view video for Query A
│   ├── Query_A_top_down_video.mp4          # Top-down view video for Query A
│   │
│   ├── Query_B_0_Robot.pkl                 # Robot state information for Query B
│   ├── Query_B_state_action_reward.hdf5    # RL rollout data for Query B (state, action, reward)
│   ├── Query_B_state_action_reward.pkl     # Pickle version of rollout data for Query B
│   ├── Query_B_fpv_video.mp4               # First-person view video for Query B
│   ├── Query_B_top_down_video.mp4          # Top-down view video for Query B
│   │
│   ├── Query_AB_Plot.png                   # Visualization comparing Query A and Query B
│
├── user_feedback_data/                     # Feedback data from participants
│   ├── participant_list_anonymous.csv           # Anonymized list of participant IDs
│   │
│   ├── randomization_id_<pid>_block_2D_FPV.csv  # Randomized query order and feedback for participant <pid>, 2D first-person view block
│   ├── randomization_id_<pid>_block_2D_TD.csv   # Randomized query order and feedback for participant <pid>, 2D top-down view block
│   ├── randomization_id_<pid>_block_VR.csv      # Randomized query order and feedback for participant <pid>, VR block
│   │
│   ├── user_study_results_id_<pid>_block_2D_FPV.csv  # Recorded responses for participant <pid>, 2D first-person block
│   ├── user_study_results_id_<pid>_block_2D_TD.csv   # Recorded responses for participant <pid>, 2D top-down block
│   ├── user_study_results_id_<pid>_block_VR.csv      # Recorded responses for participant <pid>, VR block
```

## Loaded Dataset Structure (dictionary)
Once you run `python load_dataset.py`, the `data` dictionary will have the structure below.
Note that trials (`<trial_id>`) refers to sequence of pre-recorded queries (`<query_id>`) that the participant was prompted with in each modality.
```
data_all/
│
├── <participant_id>/                       # Data for each participant
│   ├── [VR, 2D_TD, 2D_FPV]/                # Data for VR modality
│   │   ├── preferred/                      # Preferred trajectories
│   │   │   ├── metrics/                    # Metrics for preferred trajectories
│   │   │   │   ├── <metric_name>           # Dict of metrics for each trajectory/trial
│   │   │   ├── data/                       # Preferred trajectory data
│   │   │   │   ├── <trial_id>/             # Trial (numbered)
│   │   │   │   │   ├── states              # List of states in trajectoy (ordered_dict)
│   │   │   │   │   ├── actions             # List of actions
│   │   │   │   │   ├── rewards             # List of rewards
│   │   │   │   │   ├── next_states         # List of next_states
│   │   │   │   │   ├── dones               # List of dones
│   │   │   │   │   └── infos               # List of infos
│   │   ├── rejected/                       # Rejected trajectories
│   │   │   ├── metrics/                    # Metrics for rejected trajectories
│   │   │   │   ├── <metric_name>           # Dict of metrics for each trajectory/trial
│   │   │   ├── data/                       # Rejected trajectory data
│   │   │   │   ├── <trial_id>/             # Trial (numbered)
│   │   │   │   │   ├── states              # List of states in trajectoy (ordered_dict)
│   │   │   │   │   ├── actions             # List of actions
│   │   │   │   │   ├── rewards             # List of rewards
│   │   │   │   │   ├── next_states         # List of next_states
│   │   │   │   │   ├── dones               # List of dones
│   │   │   │   │   └── infos               # List of infos
│   │   ├── id                              # List of query_ids for each trial
│   │   ├── preference                      # List of user preferences for the trials
```


## State & Action Space
The **state space** ordered_dicts contain the keys
- `task_obs`: The task-specific observation comprises of the goal and human position (both 2D-polar coordinates, robot centric), and the current robot velocity (linear and angular). `task_obs` is not normalized.
- `scan`: The min-pooled 2D lidar scan with resolution 60, normalized to [0, 1] down from [0, 6] meter.

The 2D action space contains:
- linear velocity in [0, 1] m/s
- angular velocity in [-pi, +pi] rad/s