import os
import pandas as pd
import pickle as pkl

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

def collect_preference_data(user_study_data_path, skip_ids, metrics, metrics_csv=None, modalities=["VR", "2D_TD", "2D_FPV"]):
    data_all = dict()
    participant_list = pd.read_csv(os.path.join(user_study_data_path, "participant_list_anonymous.csv"))
    data_metrics = pd.read_csv(metrics_csv)

    for participant_id in list(participant_list["Participant ID"]):

        if participant_id in skip_ids:
            continue

        data_all[participant_id] = dict()

        for modality in modalities:
            data_all[participant_id][modality] = dict()
            data_all[participant_id][modality]["preferred"] = dict()
            data_all[participant_id][modality]["rejected"] = dict()
            data_all[participant_id][modality]["preferred"]["metrics"] = dict()
            data_all[participant_id][modality]["rejected"]["metrics"] = dict()
            data_all[participant_id][modality]["preferred"]["data"] = list()
            data_all[participant_id][modality]["rejected"]["data"] = list()
            data_all[participant_id][modality]["id"] = list()
            data_all[participant_id][modality]["preference"] = list()

            path = os.path.join(user_study_data_path, "user_study_results_id_{}_block_{}.csv".format(participant_id, modality))
            data = pd.read_csv(path)

            metrics_initialized = list()
            for idx, row in data.iterrows():

                # Skip learning trials
                if row['TrialNumber'] == -1:
                    continue

                preferred = row['Preference']
                rejected = "B" if preferred == "A" else "A"

                # Load pre-computed metrics from csv file
                row_metrics = data_metrics.loc[data_metrics['id'] == row['id']]
                for m, metric in enumerate(metrics):
                    if metric not in metrics_initialized:
                        data_all[participant_id][modality]['preferred']["metrics"][metric] = list()
                        data_all[participant_id][modality]['rejected']["metrics"][metric] = list()
                        metrics_initialized.append(metric)

                    data_all[participant_id][modality]['preferred']["metrics"][metric].append(
                        row_metrics[metric + str(f"_{preferred}").lower()].item()
                    )
                    data_all[participant_id][modality]['rejected']["metrics"][metric].append(
                        row_metrics[metric + str(f"_{rejected}").lower()].item()
                    )

                # Fixing the path reference to the local repo
                data_path = os.path.join(row["data_path"])
                parts = data_path.rstrip(os.sep).split(os.sep)  # Split into components
                last_three_dirs = os.sep.join(parts[-2:])
                data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/dataset", last_three_dirs)

                # Load episode data of trajectories
                with open(os.path.join(data_path, "Query_A_state_action_reward.pkl"), 'rb') as file:
                    trajectory_a = pkl.load(file)

                with open(os.path.join(data_path, "Query_B_state_action_reward.pkl"), 'rb') as file:
                    trajectory_b = pkl.load(file)

                if preferred == "A":
                    data_all[participant_id][modality]['preferred']["data"].append(trajectory_a)
                    data_all[participant_id][modality]['rejected']["data"].append(trajectory_b)
                else:
                    data_all[participant_id][modality]['preferred']["data"].append(trajectory_b)
                    data_all[participant_id][modality]['rejected']["data"].append(trajectory_a)

                data_all[participant_id][modality]["id"].append(row['id'])
                data_all[participant_id][modality]["preference"].append(preferred)
    return data_all


if __name__ == "__main__":
    participant_skip_ids = [27]  # removed from dataset due to technical issues during data collection as reported in paper

    user_study_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './resources/dataset/user_feedback_data')
    metrics_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './resources/dataset/dataset_2024_11_25__12_24_23/query_dataset_metrics.csv')

    metrics = ["length", "time", "steps", "speed_average", "angle_accumulation", "turning_points", "min_dist_human", "area_under_path"]
    units = ["m", "s", "\#", "m/s", "rad", "\#", "m", "m$^2$"]

    data = collect_preference_data(user_study_data_path, skip_ids=participant_skip_ids, metrics=metrics, metrics_csv=metrics_csv_path)
    data = data