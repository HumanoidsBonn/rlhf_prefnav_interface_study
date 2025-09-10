import numpy as np
from igibson.sensors.sensor_noise_base import BaseSensorNoise

class DepthGaussianNoise(BaseSensorNoise):
    '''
    This class can be used to add depth noise to depth image. Depth noise can be defined as 
    deviation from the true value. Usually depth noise occurs severly at extremly near or far
    distances. [1]

    Reference:
     [1] Hammond, P. D. (2019). Deep Synthetic Noise Generation for RGB-D Data Augmentation.
     https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=8516&context=etd
    '''
    def __init__(self, env):
        super(DepthGaussianNoise, self).__init__(env)

        self.distance_based = self.config.get("depth_gaussian_noise_distance_based", True) # the noise to be added will depend on the depth information at each pixel.
                                   # If its set False, the noise to be added is independent of depth information.
        self.min_dev = self.config.get("depth_gaussian_noise_min_dev", 0.01) # = 1%, allowed percentage deviation from actual value for minimum depth in the image
        self.max_dev = self.config.get("depth_gaussian_noise_max_dev", 0.05) # = 5%, allowed percentage deviation from actual value for maximum depth in the image
        self.const_std = self.config.get("depth_gaussian_noise_const_std", 0.01) # This is standard deviation used when self.distance_based=False (see description of set_const_std())
        
    def set_min_dev(self, minimum_deviation):
        '''
        This function is used to assign values to allowable percentage deviation from true value 
        at minimum depth.
        
        Parameters:

            minimum_deviation : float()
                Allowable percentage deviation from true value at minimum depth and it 
                should be between [0,1]
        
        Eg:
            If minimum depth = 0.1 and min_dev = 0.01, then allowed standard deviation =0.1 *0.01 =0.001
        '''
        self.min_dev = minimum_deviation

    def set_max_dev(self, maximum_deviation):
        '''
        This function is used to assign values to allowable percentage deviation from true value 
        at maximum depth.
        
        Parameters:

            maximum_deviation : float()
                Allowable percentage deviation from true value at maximum depth and it 
                should be between [0,1]
        
        Eg:
            If maximum depth = 0.9 and min_dev = 0.05, then allowed standard deviation =0.9 *0.05 =0.045
        '''
        self.max_dev = maximum_deviation

    def set_const_std(self, std):
        '''
        This function is used to set the standard deviation of gaussian distributed noise and this 
        standard deviation is used only when self.distance_based = False

        Parameters:

            std : float()
                Standard deviation of the gaussian distributed noise
        '''
        self.const_std =std

    def set_distance_based(self, dist_based):
        '''
        This function can be used to set boolean varible self.distance_based, which is used 
        to decide whether to add noise based on distance or not

        Parameters:

            dist_based : bool()
                If "True" is passed, the noise to be added to depth image will depend on depth 
                information at each pixel, else noise is indepdent of depth information
        '''
        self.distance_based=dist_based

    def add_noise(self, depth):
        '''
        This function is used to add noise to the depth image. If self.distance based is True,
        the noise will be synthesized based on depth information of each pixel.

        Steps to compute "Distance based additive noise":
            1. compute minium and maximum depth value from depth image
            2. compute allowable deviation per metre show in eqn [1]
            3. Multiply allowable deviation per metre with actual depth and add min_dev
               of pixel to get deviation allowed for the depth of that pixel
            4. Synthesize noise using normal distribution of zero mean and computed standard deviation
               from step 3.

        If distance based is False, normally distributed noise with constant standard deviation is
        added to the depth image

        Parameters:

            depth : np.ndarray()
                Depth image

        Returns:

            depth : np.ndarray()
                Depth image with gaussian noise
        '''

        if(self.distance_based): # checking whether to add distance based noise

            # step 1
            min_depth = np.min(depth) # minimum depth value in given depth image
            max_depth = np.max(depth) # maximum depth value in given depth image

            # step 2
            deviation_per_metre = (abs(self.max_dev - self.min_dev))/(max_depth-min_depth) # eqn [1]
            
            #step 3
            std_dev = (depth*deviation_per_metre)+self.min_dev
            
            #step 4
            depth = depth + np.random.normal(0, std_dev)

        else:
            depth=depth+np.random.normal(0, 0.05, size=depth.shape) # adding depth independent noise
            
        return depth