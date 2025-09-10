import os
import pybullet as p
import igibson
from igibson.objects.object_base import BaseObject
import numpy as np

DEFAULT_RENDERING_PARAMS = {
    "use_pbr": True,
    "use_pbr_mapping": False,
    "shadow_caster": True,
}

class VRController(BaseObject):
    def __init__(self):
        super().__init__(class_id=400, rendering_params=DEFAULT_RENDERING_PARAMS)

    def _load(self, simulator):
        visual_filename = os.path.join(os.path.dirname(__file__), "assets", "vr_controller_vive_1_5", "vr_controller_vive_1_5.obj")
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=visual_filename, meshScale=1.0)

        body_id = p.createMultiBody(
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, 0],
            baseInertialFrameOrientation=p.getQuaternionFromEuler([np.pi, np.pi/2, -np.pi/2]),
            baseMass=0,
        )
        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)
        return [body_id]