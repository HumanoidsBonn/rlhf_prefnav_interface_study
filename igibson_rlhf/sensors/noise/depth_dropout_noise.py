import numpy as np
import cv2
from igibson.sensors.sensor_noise_base import BaseSensorNoise
from igibson_rlhf.sensors.rgbd_sensor import RGBD_sensor


class DepthDropoutNoise(BaseSensorNoise):
    """
    This class can be used to add dropout noise to the simulated depth images. In dropout noise,
    the depth information is completely lost and camera reads 0. The dropout noise occurs extremely
    in the object boundaries or on dark coloured surface. [1]

    Reference:
     [1] Hammond, P. D. (2019). Deep Synthetic Noise Generation for RGB-D Data Augmentation.
     https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=8516&context=etd
    """

    def __init__(self, env, modalities):
        super(DepthDropoutNoise, self).__init__(env)
        self.env = env
        self.modalities = modalities
        # the noise to be added will depend on the depth information at each pixel.
        self.distance_based = self.config.get("depth_dropout_noise_distance_based", True)
        # used to dilate object boundaries if set to True (see det_dilate())
        self.dilate = self.config.get("depth_dropout_noise_dilate", False)
        # Probability with which noise is added at minium depth
        self.min_prob = self.config.get("depth_dropout_noise_min_prob", 0.1)
        # Probability with which noise is added at maximum depth
        self.max_prob = self.config.get("depth_dropout_noise_max_prob", 0.97)
        self.RGBD_sensor = None

    def set_distance_based(self, dist_based):
        '''
        This function can be used to set boolean varible self.distance_based, which is used 
        to decide whether to add noise based on distance or not

        Parameters:

            dist_based : bool()
                If "True" is passed, the noise to be added to depth image will depend on depth 
                information at each pixel, else noise is indepdent of depth information
        '''
        self.distance_based = dist_based

    def set_dilate(self, dilate):
        '''
        This function is used to set the variable self.dilate, which decides where to dilate
        object boundaries while adding noise

        Parameters:

            dilate : bool() 
                If this varible is True, the object boundaries are dialated, else not dialated
        '''

        self.dilate = dilate

    def set_min_prob(self, min_prob):
        '''
        This function is used to assign probability value with which dropout noise is added 
        to object boundaries at minium depth of the depth image.

        Parameters:

            min_prob : float()
                Probability with which noise is added at minium depth
        
        '''
        if (min_prob > 1 or min_prob <= 0):
            print("DepthDropoutNoise : Invalid min probability, setting default value / min_prob ~ (0,1)")
        self.min_prob = min_prob

    def set_max_prob(self, max_prob):
        '''
        This function is used to assign probability value with which dropout noise is added 
        to object boundaries at maximum depth of the depth image.

        Parameters:

            max_prob : float()
                Probability with which noise is added at maximum depth
        '''
        if (max_prob > 1 or max_prob <= 0):
            print("DepthDropoutNoise : Invalid max probability, setting default value / max_prob ~ (0,1)")
            self.max_prob = 0.97
        self.max_prob = max_prob

    def add_noise(self, depth, rgb_img=None):
        '''
        This function is used to add dropout noise to the depth image. If self.distance based is True,
        the noise will be synthesized based on depth information of each pixel. In this function "rgb" 
        image is used to identify the boundaries of the object and then dropout noise is added to those
        boundaries

        Steps to compute dropout noise:

            1.  Find edges using canny edge detection (Can be modified with custom methods)
                    1.1 convert rgb image into gray scale [0,255]
                    1.2 Add gaussian noise to smoothen the image
                    1.3 use canny edge detection to detect edges
                    1.4 Dilate edges if self.dilate is set true
            2.  Compute minimum and maximum depth values of the given depth image
            3.  Extract coordinates of the edge pixel 
            4.  If distance based method
                    4.1 Extract depth at edges
                    4.2 Compute change in probability per meter
                    4.3 compute probability vector which contains probability values 
                        for depth values at edges
                    4.4 Add dropout noise with probability computed in 4.4

            5. If distance independent method
                    5.1 Add drop noise to all edge pixel
        
        Parameters:

            depth : np.ndarray()
                Depth image

        Returns:

            depth : np.ndarray()
                Depth image with dropout noise

        '''
        if rgb_img is None and self.RGBD_sensor is None:
            self.init_rgbd_sensor()
            rgbd_obs = self.RGBD_sensor.get_obs(self.env)
            rgb_img = self.RGBD_sensor.get_rgb(rgbd_obs)

        # Step 1
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) * 255  # step 1.1
        gray_img = np.uint8(gray_img)  # step 1.1
        blurred_img = cv2.GaussianBlur(gray_img, (11, 11), 0)  # step 1.2
        edged_img = cv2.Canny(blurred_img, 0, 100, 3)  # step 1.3

        if self.dilate:  # step 1.4
            edged_img = cv2.dilate(edged_img, (1, 1), iterations=2)  # dilated edges

        # Step 2
        max_depth = np.max(depth)
        min_depth = np.min(depth)

        # step 3
        x, y = np.where(edged_img == 255)  # extracing x-coordinate, y-coordinate of pixel with value 255
        pts = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # stacking x, y to form array of shape (nx2)

        # step 4
        if self.distance_based:  # checking whether to add distance based noise

            depth_at_pts = depth[pts[:, 0], pts[:, 1]]  # Step 4.1
            prob_per_meter = (abs(self.max_prob - self.min_prob)) / (max_depth - min_depth)  # Step 4.2
            prob_vector = (depth_at_pts * prob_per_meter) + self.min_prob  # Step 4.3

            for i in range(0, len(prob_vector)):  # step 4.4
                depth[pts[i][0], pts[i][1]] *= np.random.choice([0, 1], p=[prob_vector[i][0], 1 - prob_vector[i][0]])

        # step 5
        else:

            depth[pts[:, 0], pts[:, 1]] = 0  # step 5.1

        return depth

    def init_rgbd_sensor(self):
        if self.RGBD_sensor is not None:
            return
        self.RGBD_sensor = RGBD_sensor(self.env, self.modalities)
