from igibson.simulator import Simulator
from igibson.utils.constants import PYBULLET_BASE_LINK_INDEX, PyBulletSleepState, SimulatorMode
from igibson.render.viewer import ViewerSimple


from igibson_rlhf.scenes.viewer_custom import Viewer_custom


class SimulatorCustom(Simulator):
    def __init__(self, mode,
                 physics_timestep,
                 action_timestep,
                 image_width,
                 image_height,
                 vertical_fov,
                 rendering_settings,
                 use_pb_gui):
        super().__init__(
                mode=mode,
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=image_width,
                image_height=image_height,
                vertical_fov=vertical_fov,
                rendering_settings=rendering_settings,
                use_pb_gui=use_pb_gui,
            )


    def initialize_viewers(self):
        
        if self.mode == SimulatorMode.GUI_NON_INTERACTIVE:
            self.viewer = ViewerSimple(renderer=self.renderer)
        elif self.mode == SimulatorMode.GUI_INTERACTIVE:
            print("Initializing custom viewer .............................")
            self.viewer = Viewer_custom(simulator=self, renderer=self.renderer)
