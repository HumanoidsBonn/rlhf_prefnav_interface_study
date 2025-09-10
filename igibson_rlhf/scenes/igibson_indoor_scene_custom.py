import logging
import os
import cv2
import numpy as np
from PIL import Image
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene as IIS


log = logging.getLogger(__name__)


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]

class InteractiveIndoorSceneCustom(IIS):
    """
    Custom InteractiveIndoorScene where we can exchange and modify some functions to our needs.
    """

    def load_trav_map(self, maps_path):
        """
        Loads the traversability maps for all floors

        :param maps_path: String with the path to the folder containing the traversability maps
        """
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.floor_map = []
        self.floor_graph = []
        for floor in range(len(self.floor_heights)):
            if self.trav_map_type == "with_obj":
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor))))
            # ===========================================================
            # KEY DIFFERENCE TO SUPER() IS IMPLEMENTATION OF NO_DOOR MAP:
            # ===========================================================
            elif self.trav_map_type == "no_door":
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_door_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                self.trav_map_size = int(
                    self.trav_map_original_size * self.trav_map_default_resolution / self.trav_map_resolution
                )

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size))

            # We then erode the image. This is needed because the code that computes shortest path uses the global map
            # and a point robot
            if self.trav_map_erosion != 0:
                trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            # We search for the largest connected areas
            if self.build_graph:
                self.build_trav_graph(maps_path, floor, trav_map)

            self.floor_map.append(trav_map)
