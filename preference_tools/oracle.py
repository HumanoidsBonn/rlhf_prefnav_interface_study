import collections
import numpy as np
import torch
from torch import nn


import random


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        if j < len(sizes) - 2:
            layers += [nn.Dropout(0.5 if j > 0 else 0.2)]
    return nn.Sequential(*layers)


class HumanRewardNetwork(nn.Module):
    def __init__(self, obs_size, hidden_sizes=(64, 64)):
        super(HumanRewardNetwork, self).__init__()

        self.linear_relu_stack = mlp([obs_size] + list(hidden_sizes) + [1], activation=nn.LeakyReLU)
        self.tanh = nn.Tanh()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.tanh(logits)


class HumanCritic:
    LEARNING_RATE = 0.0003
    BUFFER_SIZE = 1e5
    BATCH_SIZE = 10

    def __init__(self,
                 obs_size=3,
                 action_size=2,
                 maximum_segment_buffer=1000000,
                 maximum_preference_buffer=3500,
                 training_epochs=10,
                 batch_size=32,
                 hidden_sizes=(64, 64),
                 traj_k_lenght=100,
                 weight_decay=0.0,
                 learning_rate=0.0003,
                 env_name=None,
                 seed=12345,
                 epsilon=0.1):
        print("created")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ===BUFFER===
        self.segments = [None] * maximum_segment_buffer  # lists are very fast for random access
        self.pairs = [None] * maximum_preference_buffer
        self.critical_points = [None] * maximum_segment_buffer
        self.maximum_segment_buffer, self.maximum_preference_buffer, self.maximum_critical_points_buffer = maximum_segment_buffer, maximum_preference_buffer, maximum_segment_buffer
        self.segments_index, self.pairs_index, self.critical_points_index = 0, 0, 0
        self.segments_size, self.pairs_size, self.critical_points_size = 0, 0, 0
        self.segments_max_k_len = traj_k_lenght

        # === MODEL ===
        self.obs_size = obs_size
        self.action_size = action_size
        self.SIZES = hidden_sizes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.init_model()  # creates model

        # === DATASET TRAINING ===
        self.training_epochs = training_epochs  
        self.batch_size = batch_size 

        self.writer = None  # SummaryWriter(working_path + reward_model_name)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.updates = 0

        self.env_name = env_name
        self.epsilon = epsilon


    def init_model(self, delete=False):
        # ==MODEL==
        if delete:
            del self.reward_model
            del self.optimizer
        self.reward_model = HumanRewardNetwork(self.obs_size[0] + self.action_size, self.SIZES)
        # ==OPTIMIZER==
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)



    def add_pairs(self, o0, o1, preference):
        self.pairs[self.pairs_index] = [o0, o1, preference]
        self.pairs_size = min(self.pairs_size + 1, self.maximum_preference_buffer)
        self.pairs_index = (self.pairs_index + 1) % self.maximum_preference_buffer


    def train(self, meta_data=None, epochs_override=-1, loss_threshold=15):
        epochs = epochs_override if epochs_override != -1 else self.training_epochs
        losses = collections.deque(maxlen=10)

        self.reward_model.train(True)

        o1, o2, prefs = self.get_all_preference()

        num_pairs = len(o1)
        print("num pairs: " + str(num_pairs))

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            running_accuracy = 0.0

            indices = torch.randperm(num_pairs)

            for step_idx, i in enumerate(indices):
                self.optimizer.zero_grad()

                traj1 = o1[i]
                traj2 = o2[i]
                pref_vec = prefs[i]

                r1 = self.reward_model(traj1)
                r2 = self.reward_model(traj2)

                rs1 = r1.mean()
                rs2 = r2.mean()

                rss = torch.stack([rs1, rs2], dim=0)

                preds = torch.softmax(rss, dim=0)  # shape [2]

                label = torch.argmax(pref_vec)
                correct = (torch.argmax(preds) == label).float().item()
                running_accuracy += correct

                loss_pref = -torch.log(preds[label] + 1e-8)

                # Optionally add L1 regularization
                #l1_lambda = 0.001
                #l1_norm = sum(p.abs().sum() for p in self.reward_model.parameters())
                #loss = loss_pref + l1_lambda * l1_norm
                loss = loss_pref

                # Backprop
                loss.backward()
                self.optimizer.step()

                running_loss += loss.detach().item()

                if meta_data is not None:
                    meta_data['loss'].append(loss.detach().cpu().item())
                    meta_data['accuracy'].append(correct)

                reporting_interval = (epochs // 10) if epochs >= 10 else 1
                if epoch % reporting_interval == 0 and step_idx == (num_pairs - 1):
                    print(
                        f"Epoch {epoch}, loss on last pair = {float(loss):.4f}, "
                        f"Accuracy so far in this epoch = "
                        f"{running_accuracy / (step_idx + 1):.4f}"
                    )

            episode_loss = running_loss / num_pairs
            episode_accuracy = running_accuracy / num_pairs

            if self.writer:
                self.writer.add_scalar("reward/loss", episode_loss, self.updates)
                self.writer.add_scalar("reward/accuracy", episode_accuracy, self.updates)

            self.updates += 1

            losses.append(episode_loss)

        self.reward_model.train(False)
        return meta_data


    def save_reward_model(self, env_name=""):
        torch.save(self.reward_model.state_dict(), env_name)

    def load_reward_model(self, env_name="LunarLanderContinuous-v2"):
        print("loading:" + "models/reward_model/" + env_name)
        self.reward_model.load_state_dict(torch.load(env_name))

    def get_all_preference(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1 = []
        obs2 = []
        prefs = []
        for pair in pairs:
            obs1.append(pair[0])
            obs2.append(pair[1])
            prefs.append(pair[2])
        prefs = torch.tensor(prefs).type(torch.float32)
        return obs1, obs2, prefs

   