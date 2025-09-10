import os
import cv2
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from PIL import Image

import igibson
# from igibson.utils.assets_utils import get_scene_path
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from tqdm import tqdm
import pickle as pkl
from datetime import datetime
import matplotlib.patches as patches

from igibson_rlhf.utils import get_date_time_str


class Trajectory():
    """
    Data to be added as a single trajectory to the same plot.  
    """

    def __init__(self, robot_poses=None, robot_speeds=None, human_poses=None, human_orns=None, scene_id=None,
                 robot_orientations=None, full_robot_pose=False):
        """
        Single Trajectory to add to the Trajectory Plotter.
        :param robot_poses:  (T, 3) shape np.array giving the poses of the robot.
        :param robot_speeds:  (T) shape np.array giving the speed of the robot.  
                                If None, robot set to constant speed of 1.0
        :param human_poses:   (T, 3) shape np.array giving the poses of the huamn.
                               If (3) shape array, the human will be statically placed at the x-y pose.
                               If None, Human will not be plotted.
        :param human_orns:     If (T, 4) shape np.array, orientation of the huamn in Quaternion.
                               If (T, 3) shape array,  orientation of the huamn in Euler (yaw as [2] index).
                               If (T) shape array,  only the yaw of the human.
                               If "float" shape array,  static yaw of the human.
                               If None, no orientation will be shown
        """
        self.full_robot_pose = full_robot_pose
        if robot_poses is None:
            # Initialize empty trajectory
            self.steps = 0
            self.robot_poses = None
        else:
            self.steps = len(robot_poses)
            self.robot_poses = robot_poses[:, :2] if not self.full_robot_pose else robot_poses  # Drop z-dim

        self.robot_orientations = robot_orientations

        if robot_speeds is not None:
            self.robot_speeds = robot_speeds
        else:
            self.robot_speeds = np.ones(self.steps)

        self.human_in_fov = None

        # Create self.human_poses
        # TODO Extend for multiple humans
        self.human_poses_list = list()
        self.show_human = True
        if human_poses is None:
            self.show_human = False
            self.human_poses = np.empty((self.steps, 2)) # dummy data
        elif len(human_poses.shape) == 2:
            self.human_poses =  human_poses[:, :2] # Drop z-dim
        elif len(human_poses) == 3:
            self.human_poses = np.ones((self.steps, 2))
            self.human_poses[:, 0] *= human_poses[0]
            self.human_poses[:, 1] *= human_poses[1]
        else:
            raise RuntimeError("Invalid Human Poses Shape {}".format(human_poses.shape))

        # Create self.human_orns
        # TODO Extend for multiple humans
        self.human_orn_list = list()
        self.show_human_orn = True
        if human_orns is None:
            self.show_human_orn = False
            self.human_yaw = np.empty(self.steps) # dummy data
        elif isinstance(human_orns, float):
            # Just a yaw input
            self.human_yaw = np.ones(self.steps) # dummy data
            self.human_yaw *= human_orns
        elif len(human_poses.shape) == 1:
            self.human_yaw = human_orns
        elif human_poses.shape[1] == 3:
            # Euler Input
            self.human_yaw = human_orns[:, 2]
        elif human_poses.shape[1] == 4:
            # Quaternion Input
            orn_eul = np.array([p.getEulerFromQuaternion(human_orns[i]) for i in range(self.steps)])
            self.human_yaw = orn_eul[:, 2]
        else:
            raise RuntimeError("Invalid Human Orn Shape {}".format(human_orns.shape))

        self.collision = False
        self.timeout = False
        self.goal_reached = False

        self.goal_pose = None
        self.her_goal = False
        self.iter_count = 0

        self.state_list = list()
        self.dynamic_human = True
        self.demonstration = False

        # assert(len(self.robot_poses) == len(self.robot_speeds) == len(self.human_poses) == len(self.human_yaw) == self.steps)

        # Adding demonstrated_poses
        self.demo_poses = None
        self.config_room = None

        self.scene_id = scene_id # used for saving the scene ids (for analysis purpose)
        self.label = None
        self.reward_weight = None
        self.boxes = None
        self.color = None
        self.flipped = False

        self.action_timestep = None
        self.visualization_timestep = None
        self.use_visualization_fps = False

    def flip_xy(self):
        self.flipped = True
        roll = 1
        if self.robot_poses is not None:
            self.robot_poses[:, :2] = np.roll(self.robot_poses[:, :2], roll, -1)
            self.robot_poses[:, 1] *= -1
        if self.human_poses_list is not None:
            for h, human_poses in enumerate(self.human_poses_list):
                self.human_poses_list[h][:, :2] = np.roll(human_poses[:, :2], roll, -1)
                self.human_poses_list[h][:, 1] *= -1
        if self.demo_poses is not None:
            self.demo_poses = np.roll(self.demo_poses, roll, -1)
            self.demo_poses[:, 1] *= -1
        if self.goal_pose is not None:
            self.goal_pose = np.roll(self.goal_pose, roll, -1)
            # self.goal_pose[0] = -1 * self.goal_pose[0]
            self.goal_pose[1] = -1 * self.goal_pose[1]

    def get_human_min_distance(self):
        """
        Calculate the minimum distance between two 2D trajectories.

        Parameters:
        traj1 (list of tuples/lists): Trajectory 1, a list of (x, y) coordinates.
        traj2 (list of tuples/lists): Trajectory 2, a list of (x, y) coordinates.

        Returns:
        float: The minimum distance between the two trajectories.
        """

        def euclidean_distance(point1, point2):
            """Calculate the Euclidean distance between two 2D points."""
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        min_dist = float('inf')
        for human_poses in self.human_poses_list:
            for p1 in human_poses:
                for p2 in self.robot_poses:
                    dist = euclidean_distance(p1, p2)
                    if dist < min_dist:
                        min_dist = dist

        return min_dist

    def add_boxes(self, boxes):
        self.boxes = boxes
    def add_config_room(self, config_room):
        self.config_room = config_room
        
    def add_single_human_yaw(self, orn_list):
        """
        See __init__ for options of the orn
        """
        # TODO Extend for multiple humans
        if(len(self.human_orn_list)==0):
            self.human_orn_list = [self.human_yaw[::]] * len(orn_list)

        for i, orn in enumerate(orn_list):
            if isinstance(orn, float):
                # Just a yaw input
                yaw = orn
            elif len(orn) == 3:
                # Euler Input
                yaw = orn[2]
            elif len(orn) == 4:
                # Quaternion Input
                yaw = p.getEulerFromQuaternion(orn)[2]
            else:
                raise RuntimeError("Invalid Human Orn Shape {}".format(orn.shape))

            # if len(self.human_yaw) == 0:
            #     self.human_yaw = np.array([yaw])
            # else:
            #     self.human_yaw = np.vstack((self.human_yaw, yaw))

            if len(self.human_orn_list[i]) == 0:
                self.human_orn_list[i] = np.array([yaw])
            else:
                self.human_orn_list[i] = np.vstack((self.human_orn_list[i], yaw))

    def add_goal_pose(self, goal_pose, her=False):
        self.goal_pose = np.array(goal_pose)
        if her:
            self.her_goal = True

    def add_single_human_pos(self, human_pos_list):
        # TODO Extend for multiple humans
        if(len(self.human_poses_list)==0):
            self.human_poses_list = [self.human_poses[::]] * len(human_pos_list)

        for i, human_pos in enumerate(human_pos_list):

            if len(self.human_poses_list[i]) == 0:
                self.human_poses_list[i] = np.array([human_pos[:2]])
            else:
                self.human_poses_list[i] = np.vstack((self.human_poses_list[i], human_pos[:2]))


    def add_single_state(self, state):
        if len(self.state_list) == 0:
            self.state_list = np.array([state])
        else:
            self.state_list = np.vstack((self.state_list, state))

    def add_robot_step(self, new_robot_pose, speed=None, human_in_fov=False, orientation=None):
        """
        (x,y,z) for new robot position
        """
        self.steps += 1
        # Robot Position
        # =============================================================
        new_robot_pose = new_robot_pose if self.full_robot_pose else new_robot_pose[:2]
        if self.robot_poses is None:
            self.robot_poses = np.array([new_robot_pose])
            if speed is not None:
                self.robot_speeds = np.array([speed])
            else:
                self.robot_speeds = np.array([1.0])
            return
        else:
            self.robot_poses = np.vstack((self.robot_poses, new_robot_pose))

        # Orientation
        # =============================================================
        if self.robot_orientations is None and orientation is not None:
            self.robot_orientations = np.array([orientation])
        else:
            self.robot_orientations = np.vstack((self.robot_orientations, orientation))
        

        speed = speed if speed is not None else 1.0
        self.robot_speeds = np.vstack((self.robot_speeds, speed))

        # Human in FOV
        # =============================================================
        if self.human_in_fov is None:
            self.human_in_fov = np.array([human_in_fov])
        else:
            # print(self.human_in_fov, np.array([human_in_fov]),)
            self.human_in_fov = np.concatenate((self.human_in_fov, np.array([human_in_fov])))
        

    def add_human_step(self, human_poses_list, human_orn_list, dymanic=True):
        self.show_human = True
        self.show_human_orn = dymanic
        self.dynamic_human = dymanic
        self.add_single_human_pos(human_poses_list)
        self.add_single_human_yaw(human_orn_list)

    def get_robot_yaws(self):
        robot_yaws = []
        for i in range(len(self)-1):
            pose_a = self.robot_poses[i]
            pose_b = self.robot_poses[i+1]
            y_diff = pose_b[1] - pose_a[1]
            x_diff = pose_b[0] - pose_a[0]
            yaw = np.atan2(y_diff, x_diff)
            robot_yaws.append(yaw)

        # Duplicate the last yaw
        if len(robot_yaws) > 0:
            robot_yaws.append(robot_yaws[-1])
        else:
            robot_yaws.append(0)

        return robot_yaws

    def get_human_arrow_diff(self):
        """
        Get (dx, dy) based on the current human position and orientation
        """
        # The (dx, dy) offset that the person is looking at
        look_points_diff = []
        for orn in self.human_yaw:
            dx = np.cos(orn)
            dy = np.sin(orn)
            look_points_diff.append(np.array([dx, dy]))

        return np.array(look_points_diff)

    def get_extent(self):
        # TODO this is ugly! We should get rid of that except statement, for the case no human data is available.
        max_x = np.max(self.robot_poses[:, 0])
        min_x = np.min(self.robot_poses[:, 0])

        max_y = np.max(self.robot_poses[:, 1])
        min_y = np.min(self.robot_poses[:, 1])

        if self.show_human:
            try:
                for human_poses in self.human_poses_list:
                    max_x = max(np.max(self.robot_poses[:, 0]), np.max(human_poses[:, 0]))
                    min_x = min(np.min(self.robot_poses[:, 0]), np.min(human_poses[:, 0]))

                    max_y = max(np.max(self.robot_poses[:, 1]), np.max(human_poses[:, 1]))
                    min_y = min(np.min(self.robot_poses[:, 1]), np.min(human_poses[:, 1]))
            except:
                pass

        if self.goal_pose is not None:
            max_x = max(max_x, self.goal_pose[0])
            min_x = min(min_x, self.goal_pose[0])

            max_y = max(max_y, self.goal_pose[1])

            min_y = min(min_y, self.goal_pose[1])

        return min_x, max_x, min_y, max_y

    def add_episode_info(self, info):
        if info["max_episode_steps_reached"]:
            self.timeout = True
        if info["bumped"]:
            self.collision = True
        if info["success"]:
            self.goal_reached = True

    # def __next__(self):
    #     self.iter_count += 1
    #     i = self.iter_count - 1
    #     if i >= len(self):
    #         raise StopIteration
    #     return self.robot_poses[i], self.robot_speeds[i], self.human_poses[i], self.human_yaw[i]
    #
    # def __iter__(self):
    #     return self

    def __iter__(self):
        return iter(zip(self.robot_poses, self.robot_speeds, self.human_poses, self.human_yaw))

    def __len__(self):
        return len(self.robot_poses)

    def save_to_txt(self, fname):
        np.savetxt(fname, self.robot_poses)


class TrajectoryPlotter:
    def __init__(self, path=None, min_robot_speed=0.0, max_robot_speed=0.5, env = None):
        # List of trajectories
        self.trajectory_list = list()
        # self.plot_label = list() # Same length as list of trajectories

        # Plots are saved by default, but can also be displayed
        self.show_plts = False
        self.save_plts = True

        # Plotting save path
        self.output_path = os.path.join(os.getcwd(), 'plots') if path is None else path

        # Pre-define a few colors for labels used
        # Reserve 'green' for goal position and 'red' for obstical
        self.label_color_dict = {
                        "demonstration" : (0, 0, 1.0),  # Blue
                        "untrained" : (1.0, 1.0, 0), # Yellow
                        "trained" : (1.0, 0.5, 0),
                        'Demo': 'darkorange', #'k',
                        'Robot': 'orangered'
        }


        # Otherwise, create a random color map and an iteger for assigning
        #  new, unique colors.

        # Misc plotting settings
        self.opacity = 0.6 # opacity
        self.dpi = 600
        self.dpi_video = 400

        # Plot callback funcs can be added, e.g. special plotting calls from specific environments
        # Must follow convention fig, ax = func(fig, ax)
        self.plot_callback_funcs = list()
        self.plot_trajectory_callback_funcs = list()
        # self.plot_trajectory_callback_funcs.append(self.human_orientation_arrow)

        self.min_robot_speed = min_robot_speed
        self.max_robot_speed = max_robot_speed
        self.vel_cmap = plt.get_cmap('cool')
        self.normalizer = plt.Normalize(self.min_robot_speed, self.max_robot_speed)

        self.collision_label = False
        self.timeout_label = False
        self.goal_label = False
        self.goal_her_label = False
        self.goal_reached_label = False
        self.start_label = False
        self.human_fov_label = False

        self.arrow_length = 0.1
        self.arrow_width = 0.2
        self.robot_alpha = 0.5

        self.linewidth_robot = 1
        self.linewidth_human_fov = 2.5

        self.color_human = "firebrick"
        self.colormap_humans = matplotlib.cm.get_cmap('gnuplot')
        self.color_goal = "mediumblue"
        self.color_goal_her = "lightblue"
        self.color_goal_reached = "darkgreen"
        self.color_start = "k"
        self.color_collision = "k"
        self.color_timeout = "darkorange"  # was "navy"
        self.color_human_fov = "darkred"

        self.marker_human = "o"
        self.marker_goal = "*"
        self.marker_goal_her = "*"
        self.marker_goal_reached = "*"
        self.marker_start = "."
        self.marker_collision = "x"
        self.marker_timeout = "x"

        self.video_fps = 5

        self.env = env # the environment variable from which the image properties can be extracted

    def reset(self):
        """
        Remove all saved trajectories
        """
        self.trajectory_list = list()
        # self.plot_label = list()

        self.collision_label = False
        self.timeout_label = False
        self.goal_label = False
        self.start_label = False

    def add_new_trajectory(self, new_trajectory : Trajectory, label=None, color=None):
        """
        Add a new trajectory to plot on the current figure
        """
        label = label if label is not None else "Run {}".format(len(self.trajectory_list))
        new_trajectory.label = label # storing label in the trajectory object
        self.trajectory_list.append(new_trajectory)
        # self.plot_label.append(label)
        if color:
            self.label_color_dict[label] = color

    def add_full_legend(self, ax, loc=0):
        self.goal_label = self.collision_label = \
            self.timeout_label = self.start_label = \
            self.goal_reached_label = self.goal_her_label = True

        custom_lines = [
            mpatches.Arrow(0, 0.5*self.arrow_width, self.arrow_length, 0, color='blue', width=self.arrow_width),
            mpatches.Arrow(0, 0.5*self.arrow_width, self.arrow_length, 0, color='r',  width=self.arrow_width),
        ]
        labels = [
            'Robot', 'Human',
        ]
        ax.legend(handles=custom_lines, labels=labels, loc=loc)

    def plot_state_video(self, env, fname="test_state"):
        from matplotlib.animation import PillowWriter
        fig, ax = plt.subplots()
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        if hasattr(env, "state_video_plotter"):
            plot_func = env.state_video_plotter
        else:
            raise RuntimeError("Env does not have state_video_plotter()")

        writer = PillowWriter(fps=5)
        legend_added = False
        c = None
        with writer.saving(fig, os.path.join(self.output_path, "{}.gif".format(fname)), dpi=self.dpi_video):
            for j, (trajectory) in tqdm(enumerate(self.trajectory_list), desc="State Animation"):
                plot_label = trajectory.label
                if trajectory.demonstration:
                    continue
                for i in range(len(trajectory)):
                    fig, ax, c = plot_func(fig, ax, trajectory.state_list[i], c)
                    if not legend_added:
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(plt.gca())
                        cax = divider.append_axes("right", "5%", pad="3%")
                        plt.colorbar(c, cax=cax, label='Depth [m] (VAE Decoded)')
                        # fig.colorbar(c)
                        legend_added = True
                    writer.grab_frame()

    def plot_video(
            self,
            fig=None,
            ax=None,
            rel_margin=0.1,
            igibson_scene=None,
            show_robot_arrows=False,
            remove_furniture=False,
            fname="test",
            aspect=(4,3),
            figsize=None,
            tight_layout=False,
            mp4=True,
            keep_trajectories=True,
            ros_map_path=None,
            ros_map_yaml_path=None,
    ):
        # TODO reuse imshow artist
        from matplotlib.animation import PillowWriter
        from matplotlib.animation import FFMpegWriter

        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)

        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        # Find extent of plot with all trajectories
        # ======================================================================
        x_min, x_max, y_min, y_max = self.calc_extent(rel_margin, aspect=aspect)

        if tight_layout:
            plt.tight_layout()

        # Spot the demo trajectory in the list and save for later
        demo_traj = demo_label = None
        for j, trajectory in enumerate(self.trajectory_list):
            if trajectory.demonstration:
                demo_traj = trajectory
                demo_label = trajectory.label

        self.L1 = None # for storing fov lines
        self.L2 = None # for storing fov lines

        writer = FFMpegWriter(fps=self.video_fps) if mp4 else PillowWriter(fps=self.video_fps)
        extension = ".mp4" if mp4 else ".gif"

        with writer.saving(fig, os.path.join(self.output_path, "{}{}".format(fname, extension)), dpi=self.dpi_video):
            for j, trajectory in tqdm(
                    enumerate(self.trajectory_list),
                    desc="Map Animation",
                    total=len(self.trajectory_list)
            ):
                
                plot_label = trajectory.label

                if trajectory.demonstration:
                    continue

                if not keep_trajectories:
                    ax.clear()

                # Add scene and goal
                # ======================================================================
                if keep_trajectories and j == 0:
                    if igibson_scene is not None:
                        self.add_scene(ax, igibson_scene, remove_furniture=remove_furniture)
                    if ros_map_path is not None:
                        self.add_scene_ros(ax, ros_map_path, ros_map_yaml_path)

                # self.add_full_legend(ax)
                self.add_fixed_goal(ax, trajectory.goal_pose)

                # Add demo trajectory
                # ======================================================================
                if demo_traj is not None:
                    c = self.get_color(demo_label)
                    self.add_robot_movement(ax, demo_traj, c)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                if trajectory.human_in_fov is not None:
                    human_in_fov = list(trajectory.human_in_fov)
                else:
                    human_in_fov = list()

                for i in range(len(trajectory.robot_poses)):
                    if i == 0:
                        # Add start
                        # ======================================================================
                        self.add_start_condition(ax, trajectory)

                    if i < len(trajectory)-1:
                        # Add robot pose
                        # ====================================================================================
                        dx = trajectory.robot_poses[i + 1][0] - trajectory.robot_poses[i][0]
                        dy = trajectory.robot_poses[i + 1][1] - trajectory.robot_poses[i][1]

                        if show_robot_arrows:
                            speed = self.normalizer(trajectory.robot_speeds[i])
                            c = self.vel_cmap(speed)
                            arrow = mpatches.Arrow(trajectory.robot_poses[i][0], trajectory.robot_poses[i][1], dx, dy, color=c,
                                                 width=self.arrow_width, alpha=self.robot_alpha)
                            ax.add_patch(arrow)

                        else:
                            c = trajectory.color
                            # c = self.get_color(plot_label)
                            ax.plot(trajectory.robot_poses[i:i+2, 0], trajectory.robot_poses[i:i+2, 1], color=c, alpha=self.robot_alpha,
                                    linewidth=self.linewidth_robot)

                        self.add_fov_line(dx, dy, trajectory.robot_poses[i], ax)

                    else:
                        # Add termination condition to plot
                        # ======================================================================
                        self.add_termination_condition(ax, trajectory)

                    # Add human pose
                    # ====================================================================================
                    if trajectory.show_human:

                        multiple_humans = len(trajectory.human_poses_list) > 1

                        if not trajectory.dynamic_human:
                            for human_pos in trajectory.human_poses_list: # for multiple humans
                                # ax.plot(human_pos[:, 0], human_pos[:, 1], self.marker_human, color=self.color_human)
                                circ = plt.Circle(xy=(human_pos[0, 0], human_pos[0, 1]), radius=0.2,
                                                  color=self.color_human)
                                ax.add_patch(circ)
                        else:

                            for idx in range(len(trajectory.human_poses_list)): # for multiple humans
                                human_yaw = trajectory.human_orn_list[idx][i]
                                human_pose = trajectory.human_poses_list[idx][i]

                                dx = np.cos(human_yaw) * self.arrow_length
                                dy = np.sin(human_yaw) * self.arrow_length

                                color = self.color_human if not multiple_humans else self.colormap_humans(
                                    idx / len(trajectory.human_poses_list))
                                alpha = 1 if multiple_humans else 0.5

                                arrow = mpatches.Arrow(human_pose[0], human_pose[1], dx, dy, color=color,
                                                    width=self.arrow_width, alpha=alpha)
                                ax.add_patch(arrow)

                        if 0 < i < len(human_in_fov):
                            self.add_human_in_fov_indication_iteratively(
                                trajectory.robot_poses[i-1, :2],
                                trajectory.robot_poses[i, :2],
                                human_in_fov=human_in_fov[i-1],
                                ax=ax,
                            )

                    writer.grab_frame()
        print("Created Map Animation {}".format("{}{}".format(fname, extension)))


    def plot(
            self,
            fig=None,
            ax=None,
            show_robot_vel=False,
            show_robot_arrows=False,
            igibson_scene=None,
            fixed_goal_pose=None,
            fname='normalized_velocity',
            show_hum_orn_arrows=False,
            rel_margin=0.3,
            x_shift=0.0,
            y_shift=0.0,
            legend=True,
            tight_layout=True,
            colormap=True,
            show_human_in_fov=True,
            remove_furniture=False,
            close_plot=False,
            axis_equal=True,
            aspect=(4,3),
            robot_color=None,
            ros_map_path=None,
            ros_map_yaml_path=None,
            ):
        """
        Plot all current trajectoreis on the same ax.  
        :param ax:  Current plot axis (for ex. subplots).  If None, new will be created.
        :param show_robot_vel:  Display color of the robot's trajectory as blue-red.  
                                Add a legend to the axis with a label.
        :param show_robot_arrows:  Display the robot movement as arrows
        :param igibson_scene:  Adds the occupancy map to the figure.
        :param fixed_goal_pose:  Set a fixed goal pose to display as a green dot on the map
        :param fname:  The file name to save as.
        :param show_hum_orn_arrows:  Display the human movement as arrows
        """
        # Preparations
        # ======================================================================
        if ax is None:
            fig, ax = plt.subplots(dpi=self.dpi)

        if igibson_scene is not None:
            self.add_scene(ax, igibson_scene, remove_furniture=remove_furniture)

        if ros_map_path is not None:
            self.add_scene_ros(ax, ros_map_path, ros_map_yaml_path)

        added_labels = []

        # Loop over all trajectories
        # ======================================================================
        for i, trajectory in tqdm(enumerate(self.trajectory_list), "Plot", total=len(self.trajectory_list), disable=True):

            # Custom box plot command for FLorian's scenes TODO
            # ======================================================================
            if i == 0 and trajectory.boxes:
                if len(trajectory.boxes) > 0:
                    for b, box in enumerate(trajectory.boxes):
                        rectangle = patches.Rectangle(
                            (box[0] - 0.5, box[1] - 0.5), 1, 1,
                            linewidth=2,
                            edgecolor='black',
                            facecolor='slategrey',
                            label="Obstacle" if b == 0 else None,
                            hatch='///'
                        )
                        ax.add_patch(rectangle)

            # Custom box plot command for RLHF Study scenes TODO
            # ======================================================================
            if hasattr(trajectory, "config_room"):
                if i == 0 and trajectory.config_room:
                    for b in trajectory.config_room["randomization"]["cubes_large"].keys():
                        dim = trajectory.config_room["cube_large_dim"]
                        pos = trajectory.config_room["randomization"]["cubes_large"][b]["pos"]
                        orn = trajectory.config_room["randomization"]["cubes_large"][b]["orn"]
                        rectangle = patches.Rectangle(
                            xy=(pos[0] - dim[0], pos[1] - dim[1]),
                            width=dim[0]*2,
                            height=dim[1]*2,
                            linewidth=2,
                            edgecolor='black',
                            facecolor='slategrey',
                            label="Obstacle" if b == 0 else None,
                            hatch='///',
                            rotation_point="center",
                            angle=np.rad2deg(p.getEulerFromQuaternion(orn)[2])
                        )
                        ax.add_patch(rectangle)

                    for b in trajectory.config_room["randomization"]["cubes_small"].keys():
                        dim = trajectory.config_room["cube_small_dim"]
                        pos = trajectory.config_room["randomization"]["cubes_small"][b]["pos"]
                        orn = trajectory.config_room["randomization"]["cubes_small"][b]["orn"]
                        rectangle = patches.Rectangle(
                            xy=(pos[0] - dim[0], pos[1] - dim[1]),
                            width=dim[0]*2,
                            height=dim[1]*2,
                            linewidth=2,
                            edgecolor='black',
                            facecolor='slategrey',
                            label="Obstacle" if b == 0 else None,
                            hatch='///',
                            rotation_point="center",
                            angle=np.rad2deg(p.getEulerFromQuaternion(orn)[2])
                        )
                        ax.add_patch(rectangle)

                    # TODO Add walls
                    width = trajectory.config_room["width"]
                    length = trajectory.config_room["length"]
                    wall_thickness = trajectory.config_room["wall_thickness"]

                    rectangle = patches.Rectangle(
                        xy=(-width/2, -length/2 - wall_thickness/2),
                        width=width,
                        height=wall_thickness,
                        linewidth=2,
                        edgecolor='black',
                        facecolor='slategrey',
                        label="Obstacle" if b == 0 else None,
                        hatch='///',
                        rotation_point="center",
                    )
                    ax.add_patch(rectangle)

                    rectangle = patches.Rectangle(
                        xy=(-width / 2, +length / 2 - wall_thickness / 2),
                        width=width,
                        height=wall_thickness,
                        linewidth=2,
                        edgecolor='black',
                        facecolor='slategrey',
                        label="Obstacle" if b == 0 else None,
                        hatch='///',
                        rotation_point="center",
                    )
                    ax.add_patch(rectangle)

                    rectangle = patches.Rectangle(
                        xy=(-width / 2 - wall_thickness / 2, -length / 2),
                        width=wall_thickness,
                        height=length,
                        linewidth=2,
                        edgecolor='black',
                        facecolor='slategrey',
                        label="Obstacle" if b == 0 else None,
                        hatch='///',
                        rotation_point="center",
                    )
                    ax.add_patch(rectangle)

                    rectangle = patches.Rectangle(
                        xy=(+width / 2 - wall_thickness / 2, -length / 2),
                        width=wall_thickness,
                        height=length,
                        linewidth=2,
                        edgecolor='black',
                        facecolor='slategrey',
                        label="Obstacle" if b == 0 else None,
                        hatch='///',
                        rotation_point="center",
                    )
                    ax.add_patch(rectangle)

            
            plot_label = trajectory.label
            # Avoid adding duplicate labels to the legend
            # ======================================================================
            if plot_label in added_labels:
                label = None
            else:
                added_labels.append(plot_label)
                label = plot_label

            # Plot human in FOV with coloring
            # ======================================================================
            if show_human_in_fov and not trajectory.demonstration:
                self.add_human_fov_indication(ax, trajectory)

            # Plot robot movement with/without velocity coloring
            # ======================================================================
            if show_robot_arrows and not trajectory.demonstration:
                if show_robot_vel:
                    c = None # Auto calculate per point
                else:
                    c = self.get_color(plot_label)
                # print("add_robot_movement_with_arrows, c", c)
                self.add_robot_movement_with_arrows(ax, trajectory, label=label, color=c)
            elif show_robot_vel and not trajectory.demonstration:
                # print("add_robot_movement_with_speed")
                self.add_robot_movement_with_speed(ax, trajectory, label=label)
            else:
                if trajectory.color is not None:
                    c = trajectory.color
                else:
                    c = self.get_color(plot_label)
                self.add_robot_movement(ax, trajectory, c, label=label)
                # print("add_robot_movement, c", c)

            # Add termination condition to plot
            # ======================================================================
            self.add_termination_condition(ax, trajectory)

            # Add reward_weight to plot
            # ======================================================================
            # self.add_trajectory_info(ax, trajectory)

            # Goal pose
            # ======================================================================
            if trajectory.her_goal:
                self.add_fixed_goal_her(ax, trajectory.goal_pose)
            else:
                self.add_fixed_goal(ax, trajectory.goal_pose)

            # Add start
            # ======================================================================
            self.add_start_condition(ax, trajectory)

            # Add human to plot
            # ======================================================================
            if not trajectory.show_human:
                continue
            human_label = None if "Human" in added_labels else "Human"
            added_labels.append("Human")
            # print("dynamic human", trajectory.dynamic_human)
            if not show_hum_orn_arrows or not trajectory.dynamic_human:
                # ax.plot(trajectory.human_poses[:, 0], trajectory.human_poses[:, 1], self.marker_human, label=human_label, color=self.color_human)
                for human_poses in trajectory.human_poses_list: # For multiple humans
                    # ax.plot(human_poses[:, 0], human_poses[:, 1], self.marker_human, label=human_label, color=self.color_human)
                    circ = plt.Circle(xy=(human_poses[0, 0], human_poses[0, 1]), radius=0.2, color=self.color_human, label=human_label)
                    ax.add_patch(circ)
            else:
                self.add_human_arrows(ax, trajectory, label=human_label)

        # Find extent of plot with all trajectories
        # ======================================================================
        x_min, x_max, y_min, y_max = self.calc_extent(rel_margin, aspect=aspect)
        ax.set_xlim(x_min + x_shift, x_max + x_shift)
        ax.set_ylim(y_min + y_shift, y_max + y_shift)

        # Add a colormap if we show the velocities
        # ======================================================================
        if show_robot_vel and colormap:
            self.add_vel_colormap(fig)

        if legend:
            ax.legend(
                loc=9,
                ncol=2,
                borderpad=0.3,
                labelspacing=0.3,
                handleheight=0.7,
                handlelength=1.1,
                columnspacing=0.5,
                markerscale=0.8
            )


        if tight_layout:
            plt.tight_layout()

        if axis_equal:
            ax.set_aspect("equal")

        # Show Figure
        # ======================================================================
        if self.show_plts:
            plt.show()

        # Save figure
        # ======================================================================
        if self.save_plts:
            os.makedirs(self.output_path, exist_ok=True)
            plot_path = os.path.join(self.output_path, fname + '.png')
            fig.savefig(plot_path)
            print("Created Map Plot {}".format(plot_path))

        if close_plot:
            plt.close(fig)
            try:
                matplotlib.pyplot.close()
                del ax
                del fig
            except:
                pass

        else:
            return fig, ax

    def calc_extent(self, rel_margin, aspect=(4, 3)):
        max_x_global = min_x_global = max_y_global = min_y_global = None
        for trajectory in self.trajectory_list:
            # Find extent of plot
            # ======================================================================
            min_x, max_x, min_y, max_y = trajectory.get_extent()
            # print("trajectory extent", max_x, min_x, max_y, min_y)
            min_x_global = min(min_x_global, min_x) if min_x_global is not None else min_x
            max_x_global = max(max_x_global, max_x) if max_x_global is not None else max_x
            min_y_global = min(min_y_global, min_y) if min_y_global is not None else min_y
            max_y_global = max(max_y_global, max_y) if max_y_global is not None else max_y

        # Adjust extent of plot
        # ======================================================================
        delta_x = abs(max_x_global - min_x_global)
        delta_y = abs(max_y_global - min_y_global)

        ratio = aspect[0] / aspect[1]
        if delta_x/delta_y >= ratio:
            delta_y = delta_x / ratio
        elif delta_x/delta_y < ratio:
            delta_x = delta_y * ratio

        center_x = (max_x_global - min_x_global) / 2 + min_x_global
        center_y = (max_y_global - min_y_global) / 2 + min_y_global

        margin_x = delta_x * rel_margin
        margin_y = delta_y * rel_margin

        x_min = center_x - delta_x / 2 - margin_x
        x_max = center_x + delta_x / 2 + margin_x

        y_min = center_y - delta_y / 2 - margin_y
        y_max = center_y + delta_y / 2 + margin_y

        return x_min, x_max, y_min, y_max


    def add_human_arrows(self, ax, traj, label=None, color="r",
                            arrow_width=0.02, arrow_length=0.1):

        for ped_id, (human_poses, human_yaw) in enumerate(zip(traj.human_poses_list, traj.human_orn_list)): # multiple pedestrians
            traj.human_yaw = human_yaw[::] # get_human_arrow_diff uses self.human_yaw, so here its set
            multiple_humans = len(traj.human_poses_list) > 1

            for i, (pos, orn_diff) in enumerate(zip(human_poses, traj.get_human_arrow_diff())):
                if i == 0:
                    ax.plot(pos[0], pos[1], self.marker_start, label=("Start" if not self.start_label else None),
                            color=self.color_start)
                    self.start_label = True

                color = self.color_human if not multiple_humans else self.colormap_humans(ped_id / len(traj.human_poses_list))
                use_label = label if i == 0 else None
                if multiple_humans and use_label is not None:
                    use_label += " {}".format(ped_id + 1)
                orn_diff *= arrow_length
                alpha = 1 if multiple_humans else 0.5
                arrow = mpatches.Arrow(pos[0], pos[1], orn_diff[0], orn_diff[1], color=color, label=use_label, width=arrow_width, alpha=alpha)
                ax.add_patch(arrow)

    def add_human_fov_indication(self, ax, trajectory):
        """
        Segments the robot trajectory in parts, in which the human can be seen and plots those.
        """
        if trajectory.human_in_fov is None:
            return

        human_in_fov = list(trajectory.human_in_fov)

        # SEGMENT THE TRAJECTORY TO WHERE THE HUMAN CAN BE SEEN
        # =====================================================
        segments = list()
        current_segment = list()
        for i in range(0, len(human_in_fov)):
            if human_in_fov[i]:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = list()
        segments.append(current_segment)

        # PLOT THE PARTS, WHERE HUMAN IN FOV
        # =====================================================
        for ids in segments:
            if ids:
                ids_extended = ids
                ids_extended.append(ids[-1]+1)
                poses = trajectory.robot_poses[ids_extended, :]
                label = "Human in FOV" if not self.human_fov_label else None
                ax.plot(
                    poses[:, 0], poses[:, 1],
                    color=self.color_human_fov,
                    alpha=self.opacity,
                    linewidth=self.linewidth_human_fov,
                    label=label
                )
                self.human_fov_label = True

    def add_fixed_goal(self, ax, fixed_goal, label="Goal"):
        if fixed_goal is None:
            return
        label = label if not self.goal_label else None
        ax.plot(fixed_goal[0], fixed_goal[1], self.marker_goal, label=label, color=self.color_goal)
        self.goal_label = True

    def add_fixed_goal_her(self, ax, fixed_goal, label="HER Goal"):
        if fixed_goal is None:
            return
        label = label if not self.goal_her_label else None
        ax.plot(fixed_goal[0], fixed_goal[1], self.marker_goal_her, label=label, color=self.color_goal_her)
        self.goal_her_label = True

    def add_trajectory_info(self, ax, trajectory):
        if trajectory.reward_weight is None:
            return
        x, y = trajectory.robot_poses[0, 0], trajectory.robot_poses[0, 1]
        s = "["
        for i in list(trajectory.reward_weight):
            s += "{}".format(int(i))
        s += "]"
        ax.text(x, y, s)

    def add_vel_colormap(self, fig, cax=None, cmap_label="Velocity [m/s]", fraction=1, shrink=0.5, ticks=None):
        x_dum = np.arange(self.min_robot_speed, self.max_robot_speed, 0.01)
        y_dum = np.zeros(len(x_dum))
        segments = self.build_segments(x_dum, y_dum)
        lc = LineCollection(segments, array=x_dum, cmap=self.vel_cmap, norm=self.normalizer)
        if cax is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(lc, cax=cax, label=cmap_label)
        else:
            fig.colorbar(lc, ax=cax, label=cmap_label, fraction=fraction, shrink=shrink, ticks=ticks)

    def add_start_condition(self, ax, trajectory):
        x, y = trajectory.robot_poses[:, 0], trajectory.robot_poses[:, 1]
        ax.plot(x[0], y[0], self.marker_start, label=("Start" if not self.start_label else None),
                color=self.color_start)
        self.start_label = True

    def add_termination_condition(self, ax, trajectory: Trajectory):
        x, y = trajectory.robot_poses[:, 0], trajectory.robot_poses[:, 1]
        if trajectory.collision:
            ax.plot(x[-1], y[-1], self.marker_collision, label=("Collision" if not self.collision_label else None),
                    color=self.color_collision)
            self.collision_label = True
        elif trajectory.timeout:
            ax.plot(x[-1], y[-1], self.marker_timeout, label=("Timeout" if not self.timeout_label else None),
                    color=self.color_timeout)
            self.timeout_label = True
        elif trajectory.goal_reached:
            ax.plot(x[-1], y[-1], self.marker_goal_reached, label=("Success" if not self.goal_reached_label else None),
                    color=self.color_goal_reached)
            self.goal_reached_label = True
        # else:
        #     ax.plot(x[-1], y[-1], '.', color="green", label=("Goal" if not self.goal_label else None))
        #     self.goal_label = True


    def add_robot_movement_with_arrows(self, ax, traj, color=None, label=None,
                                        arrow_width=0.1):
        for i in range(len(traj)-1):
            use_label = label if i == 0 else None
            dx = traj.robot_poses[i+1][0] - traj.robot_poses[i][0]
            dy = traj.robot_poses[i+1][1] - traj.robot_poses[i][1]
            if color is None:
                # print(traj.robot_speeds)
                speed = self.normalizer(traj.robot_speeds[i])
                c = self.vel_cmap(speed)
            else:
                c = color
            arrow = mpatches.Arrow(traj.robot_poses[i][0], traj.robot_poses[i][1], dx, dy, color=c, label=use_label, width=arrow_width)
            ax.add_patch(arrow)

    def add_robot_movement_with_fancy_arrows(self, ax, traj, color=None, label=None):
        for i in range(len(traj)-1):
            use_label = label if i == 0 else None
            if color is None:
                speed = self.normalizer(traj.robot_speeds[i])
                c = self.vel_cmap(speed)
            else:
                c = color
            arrow = mpatches.FancyArrowPatch(traj.robot_poses[i, :2], traj.robot_poses[i+1, :2], color=c, label=use_label)
            ax.add_patch(arrow)

    def add_robot_movement_with_speed(self, ax, traj, 
                                        linewidth=1,
                                        label=None,
                                    ):
        # Use LineCollection for colloring each section
        normalizer = plt.Normalize(self.min_robot_speed, self.max_robot_speed)
        # print("robot movement with speed", traj.robot_poses)
        segments = self.build_segments(traj.robot_poses[:, 0], traj.robot_poses[:, 1])
        if label is None:
            lc = LineCollection(segments, cmap=self.vel_cmap, norm=self.normalizer, alpha=self.opacity, linewidth=self.linewidth_robot)
        else:
            lc = LineCollection(segments, cmap=self.vel_cmap, norm=self.normalizer, alpha=self.opacity, label=label, linewidth=self.linewidth_robot)

        # Set the color of each line segment according to the robot speed
        lc.set_array(np.squeeze(traj.robot_speeds))
        lc.set_linewidth(self.linewidth_robot)

        # Add to plot
        ax.add_collection(lc)

    def add_robot_movement(self, ax, traj, c, label=None):
        if label is None:
            ax.plot(traj.robot_poses[:, 0], traj.robot_poses[:, 1], color=c, alpha=self.opacity, linewidth=self.linewidth_robot)
        else:
            ax.plot(traj.robot_poses[:, 0], traj.robot_poses[:, 1], color=c, alpha=self.opacity, label=label, linewidth=self.linewidth_robot)
           
    def get_color(self, label):
        if label is None:
            # Do not need to match it in the dictionary
            return self.generate_color()

        if label in self.label_color_dict.keys():
            return self.label_color_dict[label]

        new_color = self.generate_color()

        self.label_color_dict[label] = new_color
        return new_color

    def generate_color(self):
        iter_count = 0

        random_color = np.random.rand(3,)
        while self.check_color_in_use(random_color):
            random_color = np.random.rand(3,)
            iter_count += 1
            if iter_count >= 200: # Arbitrary number
                raise ValueError("All Colors in Use")

        return random_color

    def check_color_in_use(self, potential_color):
        for color in self.label_color_dict:
            if self.same_color(self.label_color_dict[color], potential_color):
                return True

        # Green and Red are also reserved
        check_red = self.same_color((1.0, 0, 0), potential_color)
        check_green = self.same_color((0, 0, 1.0), potential_color)
        return check_red or check_green

    def same_color(self, color1, color2, eps=0.1):
        if isinstance(color1, str) and not isinstance(color2, str):
            return False
        color1, color2 = np.array(color1[:2]), np.array(color2[:2])
        return (np.linalg.norm(color1 - color2) < eps)


    def build_segments(self, x, y):
        """
        This creates a segment between each adjacent point in a line.  
        Then each segment can be colored separately.  
        For example, line with the set of points:
                line:  [ (0,0), (1,1), (2,3) ]
            We would get two segments:
                segments: [
                            [(0,0), (1,1)],
                            [(1,1), (2,3)]
                          ]
                where each segment is represented by two points.
        """
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def check_for_flipped_trajectories(self):
        flipped = False
        try:
            for trajectory in self.trajectory_list:
                if trajectory.flipped:
                    flipped = True
                if flipped and not trajectory.flipped:
                    print("Warning, one of the trajecoties is not flipped!")
        except:
            pass
        return flipped


    def add_scene_ros(self, ax, map_path=None, map_yaml_path=None):
        '''
        This function is used to add map image to the plot, in which the trajectory will be
        plotted.

        Parameters:

            ax : matplotlib.axes._subplots.AxesSubplot()
                The maplotlib object in which the map will be plotted

            map_path : str()
                Path to the map image

            map_yaml_path : str()
                Path to the corresponding yaml file of map
        '''

        map, map_origin, map_extent = self.read_map_chat(map_path, map_yaml_path)  # reading map

        if self.check_for_flipped_trajectories():
            # extracting minimum and maximum extents of the map
            min_x = map_origin[0]
            min_y = map_origin[1]
            max_x = map_extent[0] + map_origin[0]
            max_y = map_extent[1] + map_origin[1]
            map = np.rollaxis(map, 1)

        else:
            min_x = map_origin[1]
            min_y = map_origin[0]
            max_x = map_extent[1] + map_origin[1]
            max_y = map_extent[0] + map_origin[0]

        print("min_x, max_x, min_y, max_y", min_x, max_x, min_y, max_y)
        print("map_extent", map_extent)

        # Plotting map
        # ax.imshow(map, cmap="Greys", extent=[min_x, max_x, min_y, max_y])
        ax.imshow((255 - map), cmap="Greys", extent=map_extent, vmin=-40, vmax=255)

    def zoom_points(self, points, reference_point, zoom_factor):
        zoomed_points = []  # Initialize a list to hold the zoomed points

        if not isinstance(points, (list, np.ndarray)):
            points = [points,]

        for point in points:
            # Step 1: Translate point relative to reference point
            translated_point = [point[0] - reference_point[0], point[1] - reference_point[1]]

            # Step 2: Scale the translated point
            scaled_point = [translated_point[0] * zoom_factor, translated_point[1] * zoom_factor]

            # Step 3: Translate back by adding the reference point coordinates
            zoomed_point = [scaled_point[0] + reference_point[0], scaled_point[1] + reference_point[1]]

            # Append the zoomed point to the list
            zoomed_points.append(zoomed_point)
        return zoomed_points

    def read_map(self, map_path, map_yaml_path):
        '''
        This function is used to read the map image and map yaml file. From the yaml file informations like
        resolution, map origin can be extracted.

        Parameters:

            map_path : str()
                Path to the map image

            map_yaml_path : str()
                Path to the corresponding yaml file of map

        Returns:

            map : np.ndarray()
                Returns the map from loaded from image as numpy array

            map_origin : np.ndarray()
                Returns the origin (x_o,y_o,z_o) of the map as numpy array
                eg: [-5.0, -5.0, 0]

            map_extent : np.ndarray()
                Returns the maximum length of the map in x and y directions
        '''
        map = np.array(Image.open(map_path))

        # Extracting data from yaml file
        import yaml
        with open(map_yaml_path) as f:
            map_yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

        map_resolution = map_yaml_data['resolution']  # Resolution is mapping from pixel value to real world distance
        map_origin = np.array(map_yaml_data["origin"]) * map_resolution
        map_extent = np.array(map.shape[:2]) * map_resolution
        print("map_resolution", map_resolution)
        print("map_origin", map_origin)
        print("map_extent", map_extent)
        return map, map_origin, map_extent

    def read_map_chat(self, map_path, map_yaml_path):
        map = np.array(Image.open(map_path))
        import yaml
        with open(map_yaml_path) as f:
            map_yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

        map_resolution = map_yaml_data['resolution']
        # Assuming the origin in the YAML file is [x, y, z]
        # and represents the real-world coordinates of the bottom-left corner.
        map_origin = np.array(map_yaml_data["origin"][:2])  # Only take x, y
        map_extent = np.array(map.shape[:2]) * map_resolution

        print("map_resolution", map_resolution)
        print("map_origin", map_origin)
        print("map_extent", map_extent)

        # Adjust map_extent to represent [left, right, bottom, top]
        map_extent = [map_origin[0], map_origin[0] + map_extent[1],
                      map_origin[1], map_origin[1] + map_extent[0]]
        return map, map_origin, map_extent

    def add_scene(self, ax, scene, scene_id=None, grey_tone_walls=0.2, grey_tone_obs=0.35, remove_furniture=False):
        """
        Add an igibson traversability map to the axes
        :param gray_tone_...:   Zero is for black, 1.0 is for white
        """
        # trav_map = Image.open(os.path.join(get_scene_path(scene.scene_id), "floor_trav_{}.png".format(f)))
        # trav_map = Image.open(os.path.join(get_scene_path(scene.scene_id), "layout", "floor_trav_no_door_0.png"))
        # trav_map = trav_map.convert("L")
        # trav_map = 255 - np.array(trav_map) # Invert colors

        # trav_map = scene.floor_map[0]
        trav_map_obs = 255 - np.array(Image.open(os.path.join(igibson.ig_dataset_path, "scenes", scene.scene_id, "layout", "floor_trav_no_door_0.png")))
        trav_map_no_obs = 255 - np.array(Image.open(os.path.join(igibson.ig_dataset_path, "scenes", scene.scene_id, "layout", "floor_trav_no_obj_0.png")))

        # Color obstacles as gray_tone
        trav_map_obs[trav_map_obs == 255] = (1-grey_tone_obs) * 255 if not remove_furniture else 0

        # Set the Walls to black
        trav_map_obs[trav_map_no_obs == 255] = (1-grey_tone_walls) * 255
        trav_map = trav_map_obs

        # trav_map = cv2.resize(trav_map, (scene.trav_map_size, scene.trav_map_size))
        #trav_map = 255 - trav_map # Invert colors

        #trav_map[trav_map == 0] = grey_tone * 255
        trav_map = np.flip(trav_map, axis=0)
        # print("Trav Map Shape:  ",  trav_map.shape)

        # Scale the x/y coordinates to meters in the world map
        
        # print("Map Trav Res:  ",  scene.trav_map_resolution)
        # print("Map Trav Siye:  ",  scene.trav_map_size)

        min_x, min_y = scene.map_to_world(np.array([0,0]))
        max_x, max_y = scene.map_to_world(np.array([scene.trav_map_size,scene.trav_map_size]))
        # print("Zero Position:   ",  min_x, min_y)
        # print("Max Position:   ",  max_x, max_y)
        # print("Trav map orig size:  ",  scene.trav_map_original_size)


        ax.imshow(trav_map, cmap="Greys", extent=[min_x, max_x, min_y, max_y])
        # ax.imshow(trav_map, cmap="Greys", extends)

    def save_trajectory_to_pkl(self, folder_path=None, base_file_name=None, append_termination=True):
        '''
        This function is used to store the trajectory objects in pickle file. Each trajectory object
        is stored in separate files and file naming follows incremental order.

        Parameters:

            folder_path : str()
                Folder path in which all the trajectories to be stored
                Default : /home/trajectory_files_<current time stamp>"

            base_file_name : str()
                Name of the trajectory file for incremental naming.
                The name also includes the termination condition like timeout/collision/goal_reached
                Eg: if base_file_name is 'demo', then file names will be 'demo_1_collision/timeout/goal_reached'
        '''
        
        if(folder_path == None):
            print("[Info] Setting default folder path and file name")
            print("[Info] Folder path : /trajectory_files_<current time stamp>")
            folder_path = "./trajectory_list__{}".format(get_date_time_str())
            folder_path = os.path.join(self.output_path, folder_path)
        os.makedirs(folder_path, exist_ok=True)

        if base_file_name == None:
            print("[Info] Setting default file name")
            print("[Info] File names: trajectory_1, trajectory_2, ..")
            base_file_name = "trajectory_"

        for i, trajectory in enumerate(self.trajectory_list):
            label = trajectory.label
            termination_condition = ""
            if append_termination:
                if trajectory.collision:
                    termination_condition = "_collision"
                elif trajectory.timeout:
                    termination_condition = "_timeout"
                elif trajectory.goal_reached:
                    termination_condition = "_success"

            file_name = base_file_name + str(i) + "_{}".format(label) + termination_condition + ".pkl"
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "wb") as file:
                pkl.dump(trajectory, file, pkl.HIGHEST_PROTOCOL)
        

    def load_trajectory_from_pkl(self, folder_path, file_name = None, discard_existing=True):
        '''
        Parameters:

                folder_path : str()
                    Folder path in which all trajectories stored

                file_name : str()
                    Name of the trajectory file. If the file name is provided only that particular
                    trajectory will be loaded

                discard_existing : bool()
                    If this variable is set "True" existing trajectory list will be deleted and stored 
                    trajectories will be loaded, else stored trajectories will be appended to existing list
                    of trajectories.
        '''
        


        if(discard_existing):
            self.trajectory_list = []
        
        if(file_name==None):
            trajectory_files = os.listdir(folder_path)
            trajectory_files.sort()
            for file_name in trajectory_files:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'rb') as file:
                    trajectory_obj = pkl.load(file)
                    self.trajectory_list.append(trajectory_obj)
        else:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as file:
                trajectory_obj = pkl.load(file)
                self.trajectory_list.append(trajectory_obj)

    def add_human_in_fov_indication_iteratively(self, robot_pose, robot_pose_last, human_in_fov, ax):
        if not human_in_fov:
            return
        poses = np.stack([robot_pose, robot_pose_last])
        label = "Human in FOV" if not self.human_fov_label else None
        ax.plot(
            poses[:, 0], poses[:, 1],
            color=self.color_human_fov,
            alpha=self.opacity,
            linewidth=self.linewidth_human_fov,
            label=label,
        )
        self.human_fov_label = True

    def add_fov_line(self, dx, dy, robot_pose, ax):
        '''
        This function is used to add the field of view of the camera to the plots. 

        Parameters:

            dx : float()
                Difference in x coordinate between current robot pose and next robot pose

            dy : float()
                Difference in y coordinate between current robot pose and next robot pose

            robot_pose : np.ndarray()
                The current robot pose (x, y) coordinates

            ax : matplotlib.axes._subplots.AxesSubplot
                The matplotlib object in which the trajectory will be plotted
        '''
        if "vertical_fov" not in self.env.config.keys():
            return
        
        if (self.L1 is not None and self.L2 is not None):
            try:
                ax.lines.remove(self.L1)
                ax.lines.remove(self.L2)
            except:
                pass
                print("Removing FOV Lines L1/L2 from ax did not work!")

        # Extracting camera and image properties
        vertical_fov = self.env.config['vertical_fov']
        img_width = self.env.config['image_width']
        img_height = self.env.config['image_height']

        # Calculating horizontal field of view from vertical field of view and image dimension
        horizontal_fov = ( vertical_fov*img_width ) / img_height
        length = 6
        robot_orientation = np.arctan2(dy, dx) # computing robot orientation

        # calculating points for line 1
        angle = np.deg2rad(np.rad2deg(robot_orientation) + (horizontal_fov/2))
        pt_1_x = robot_pose[0] + np.cos(angle) * length
        pt_1_y = robot_pose[1] + np.sin(angle) * length
        line_1 = np.array( [ [robot_pose[0], robot_pose[1]], [pt_1_x, pt_1_y]] ) 

        # calculating points for line 2
        angle = np.deg2rad(np.rad2deg(robot_orientation) - (horizontal_fov/2))
        pt_2_x = robot_pose[0] + np.cos(angle) * length
        pt_2_y = robot_pose[1] + np.sin(angle) * length
        line_2 = np.array( [[robot_pose[0], robot_pose[1]], [pt_2_x, pt_2_y]] )

        # Plotting lines
        ax.plot(
            line_1[:, 0],
            line_1[:, 1],
            color=self.color_human_fov,
            linestyle='dashed',
            alpha=self.opacity,
            linewidth=self.linewidth_robot
        )
        self.L1 = ax.lines[-1]  # storing the line object for removing later

        ax.plot(
            line_2[:, 0],
            line_2[:, 1],
            color=self.color_human_fov,
            linestyle='dashed',
            alpha=self.opacity,
            linewidth=self.linewidth_robot
        )
        self.L2 = ax.lines[-1]  # storing the line object for removing later

    def get_demo_free_trajectory_list(self):
        trajectory_list_no_demo = []
        for trajectory in self.trajectory_list:
            if trajectory.label not in ["Demo", "HER"]:
                trajectory_list_no_demo.append(trajectory)
        return trajectory_list_no_demo

if __name__ == "__main__":
    seconds = 2
    action_freq = 1./5.
    scene_id = "Rs_int"
    robot_start = (-1, -1)
    robot_goal = (1, 1)
    min_speed = 0.25
    max_speed = 1.0
    steps = int(seconds/action_freq)

    dummy_human_pos = (np.cos(np.linspace(0, seconds, num=steps)), 
                        np.sin(np.linspace(0, seconds, num=steps)),
                        np.zeros(steps))
    dummy_human_orn_eul = ( np.zeros(steps),
                        np.zeros(steps),
                        np.linspace(0, np.pi, num=steps))
    dummy_robot_pos = (np.linspace(robot_start[0], robot_goal[0], num=steps),
                        np.linspace(robot_start[1], robot_goal[1], num=steps),
                        np.zeros(steps),
        )
    dummy_robot_speeds = np.random.uniform(low=min_speed, high=max_speed, size=steps)

    # transpose
    dummy_human_pos = np.array(dummy_human_pos).transpose()
    dummy_human_orn_eul = np.array(dummy_human_orn_eul).transpose()
    dummy_human_orn = np.array([p.getQuaternionFromEuler(dummy_human_orn_eul[i]) for i in range(steps)])
    dummy_robot_pos = np.array(dummy_robot_pos).transpose()



    

    # Create iGibson Scene
    from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
    from igibson.simulator import Simulator
    from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    s = Simulator(
        mode="headless",
        rendering_settings=settings,
    )
    scene = InteractiveIndoorScene(
        scene_id,
        load_object_categories=[],  # To load only the building. Fast
        build_graph=True,
    )
    # scene.trav_map_resolution = scene.trav_map_default_resolution
    s.import_scene(scene)
    

    tp = TrajectoryPlotter()

    # Add trajectory
    t = Trajectory(dummy_robot_pos, dummy_robot_speeds, dummy_human_pos, dummy_human_orn)
    tp.add_new_trajectory(t, label="untrained")

    half_steps = int(steps/2)
    half_steps2 = steps - half_steps
    dummy_robot_pos2 = (np.append(np.linspace(robot_start[0], robot_goal[0], num=half_steps), np.linspace(robot_goal[0], robot_goal[0], num=half_steps2)),
                        np.append(np.linspace(robot_start[1], robot_start[1], num=half_steps), np.linspace(robot_start[1], robot_goal[1], num=half_steps2)),
                        np.zeros(steps),
        )
    dummy_robot_pos2 = np.array(dummy_robot_pos2).transpose()
    t2 = Trajectory(dummy_robot_pos2, dummy_robot_speeds, dummy_human_pos, dummy_human_orn)
    tp.add_new_trajectory(t2, label="trained")

    tp.plot(igibson_scene=scene, 
            show_robot_vel=True,
            show_robot_arrows=True,
            fixed_goal_pose=robot_goal,
            show_hum_orn_arrows=True,
            )





