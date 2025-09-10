from igibson.render.viewer import Viewer
from igibson.utils.constants import ViewerMode
import cv2
import numpy as np
import os
import random
import time
import logging
import sys
import pybullet as p
from igibson.utils.utils import rotate_vector_2d
from copy import deepcopy


log = logging.getLogger(__name__)

class Viewer_custom(Viewer):

    def __init__(self, simulator, renderer):
        super().__init__(simulator=simulator, renderer=renderer )
        self.pause_render = False

    def set_cam_pose_and_view(self, cam_pose, cam_view_direction):
        '''
        This function is used to change the camera pose and view direction on runtime

        Parameters:
            cam_pose : list()
                The cam_pose is a list of 3 elements representing position of camera in x, y, z direction

            cam_view_direction: list()
                The cam_view_direction is a list of 3 elements representing direction of view.
        '''
        self.initial_pos = cam_pose
        self.initial_view_direction = cam_view_direction
        self.reset_viewer()


    def update(self, ):

        """
        Update images of Viewer
        """

        self.out_frame = None
        # print("updating in custom viewer ....")
        camera_pose = np.array([self.px, self.py, self.pz])
        if self.renderer is not None:
            self.renderer.set_camera(camera_pose, camera_pose + self.view_direction, self.up)

        if self.renderer is not None and not self.pause_render:
            frame = cv2.cvtColor(np.concatenate(self.renderer.render(modes=("rgb")), axis=1), cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((300, 300, 3)).astype(np.uint8)

        # Text with the position and viewing direction of the camera of the external viewer
        # text_color = (0, 0, 0)
        # cv2.putText(
        #     frame,
        #     "px {:1.1f} py {:1.1f} pz {:1.1f}".format(self.px, self.py, self.pz),
        #     (10, 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     text_color,
        #     1,
        #     cv2.LINE_AA,
        # )
        # cv2.putText(
        #     frame,
        #     "[{:1.1f} {:1.1f} {:1.1f}]".format(*self.view_direction),
        #     (10, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     text_color,
        #     1,
        #     cv2.LINE_AA,
        # )
        # cv2.putText(
        #     frame,
        #     ["nav mode", "manip mode", "planning mode"][self.mode],
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     text_color,
        #     1,
        #     cv2.LINE_AA,
        # )
        # self.show_help_text(frame)
        if not self.pause_render:
            cv2.imshow("Viewer", frame)

        self.last_key = cv2.waitKey(1)

        if self.last_key != -1:
            # Update the last pressed key and record the time
            self.last_pressed_key = self.last_key
            self.time_last_pressed_key = time.time()

        move_vec = self.view_direction[:2]
        # step size is 0.1m
        step_size = 0.1
        move_vec = move_vec / np.linalg.norm(move_vec) * step_size

        # show help text
        if self.last_key == ord("h"):
            self.show_help += 1

        # move
        elif self.last_key in [ord("w"), ord("s"), ord("a"), ord("d")]:
            if self.last_key == ord("w"):
                yaw = 0.0
            elif self.last_key == ord("s"):
                yaw = np.pi
            elif self.last_key == ord("a"):
                yaw = -np.pi / 2.0
            elif self.last_key == ord("d"):
                yaw = np.pi / 2.0
            move_vec = rotate_vector_2d(move_vec, yaw)
            self.px += move_vec[0]
            self.py += move_vec[1]
            if self.mode == ViewerMode.MANIPULATION:
                self.move_constraint(self._mouse_ix, self._mouse_iy)

        elif self.last_key in [ord("t")]:
            self.pz += step_size

        elif self.last_key in [ord("g")]:
            self.pz -= step_size

        # turn left
        elif self.last_key == ord("q"):
            self.theta += np.pi / 36
            self.view_direction = np.array(
                [np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(self.phi), np.sin(self.phi)]
            )
            if self.mode == ViewerMode.MANIPULATION:
                self.move_constraint(self._mouse_ix, self._mouse_iy)

        # turn right
        elif self.last_key == ord("e"):
            self.theta -= np.pi / 36
            self.view_direction = np.array(
                [np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(self.phi), np.sin(self.phi)]
            )
            if self.mode == ViewerMode.MANIPULATION:
                self.move_constraint(self._mouse_ix, self._mouse_iy)

        # quit (Esc)
        elif self.last_key == 27:
            if self.video_folder != "":
                log.info(
                    "You recorded a video. To compile the frames into a mp4 go to the corresponding subfolder"
                    + " in /tmp and execute: "
                )
                log.info("ffmpeg -i %5d.png -y -c:a copy -c:v libx264 -crf 18 -preset veryslow -r 30 video.mp4")
                log.info("The last folder you collected images for a video was: " + self.video_folder)
            sys.exit()

        # Start/Stop recording. Stopping saves frames to files
        elif self.last_key == ord("r"):
            if self.recording:
                self.recording = False
                self.pause_recording = False
            else:
                log.info("Start recording*****************************")
                # Current time string to use to save the temporal urdfs
                timestr = time.strftime("%Y%m%d-%H%M%S")
                # Create the subfolder
                self.video_folder = os.path.join(
                    "/tmp", "{}_{}_{}".format(timestr, random.getrandbits(64), os.getpid())
                )
                os.makedirs(self.video_folder, exist_ok=True)
                self.recording = True
                self.frame_idx = 0

        # Pause/Resume recording
        elif self.last_key == ord("p"):
            if self.pause_recording:
                self.pause_recording = False
            else:
                self.pause_recording = True

        # Switch amoung navigation, manipulation, motion planning / execution modes
        elif self.last_key == ord("m"):
            self.left_down = False
            self.middle_down = False
            self.right_down = False
            if self.planner is not None:
                self.mode = (self.mode + 1) % len(ViewerMode)
            else:
                # Disable planning mode if planner not initialized (assume planning mode is the last available mode)
                assert ViewerMode.PLANNING == len(ViewerMode) - 1, "Planning mode is not the last available viewer mode"
                self.mode = (self.mode + 1) % (len(ViewerMode) - 1)

        elif self.last_key == ord("z"):
            self.initial_pos = [self.px, self.py, self.pz]
            self.initial_view_direction = self.view_direction

        elif self.last_key == ord("x"):
            self.reset_viewer()

        elif self.is_robosuite and self.last_key in {ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")}:
            idxx = int(chr(self.last_key))
            self.renderer._switch_camera(idxx)
            if not self.renderer._is_camera_active(idxx):
                cv2.destroyWindow(self.renderer._get_camera_name(idxx))

        if self.recording and not self.pause_recording:
            cv2.imwrite(
                os.path.join(self.video_folder, "{:05d}.png".format(self.frame_idx)), (frame * 255).astype(np.uint8)
            )
            self.frame_idx += 1

        self.out_frame=deepcopy(frame)

        # if self.renderer is not None:
        #     if self.is_robosuite:
        #         frames = self.renderer.render_active_cameras(modes=("rgb"))
        #         names = self.renderer._get_names_active_cameras()
        #         assert len(frames) == len(names)
        #         if len(frames) > 0:
        #             for (rgb, cam_name) in zip(frames, names):
        #                 frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #                 cv2.imshow(cam_name, frame)
        #     else:
        #         frames = self.renderer.render_robot_cameras(modes=("rgb"), cache=False)
        #         if len(frames) > 0:
        #             frame = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
        #             cv2.imshow("RobotView", frame)

        # self.out_frame_robot=deepcopy(frame)