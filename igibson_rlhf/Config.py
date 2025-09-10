"""
Confugration of the arg-parser settings.
Control program flow / save path / iGibson scene / ...
"""
import os
import argparse
import h5py
import yaml
# from gibson2.utils import set_random_seed, make_folder, get_timestamp
import gibson2.utils as g2u


def add_dict_to_h5py(f, dic):
    for key in dic.keys():
        if dic[key]:
            if isinstance(dic[key], dict):
                d = f.create_group(key)
                add_dict_to_h5py(d, dic[key])
            else:
                # f.attrs[key] = dic[key]
                f.create_dataset(key, data=dic[key])

def add_h5py_meta_data(h5py_file, config):
    with h5py.File(h5py_file, 'a') as f:

        metadata = f.create_group("metadata")

        metadata_config = metadata.create_group("config")
        metadata_yaml = metadata.create_group("yaml")

        # Add the program config data
        for key in config._default_values.keys():
            # metadata_config.attrs[key] = config[key]
            metadata_config.create_dataset(key, data=config[key])

        # Add iGibson configuration data
        add_dict_to_h5py(metadata_yaml, config.yaml_data)



class Config(dict):
    """
    Program Configuration Class.
    """
    _optional_actors = ["random", "astar", "brrt", "rl_policy"]

    # Default Value Only.  New items added MUST have a default value
    _default_values = {
        "gui": False,
        "save_path": "./data/",
        "runs": 3,
        "actor": "rl_policy",
        "seed":  42,
        "yaml": "__igibson_configs__/22_09_05__single_dynamic_pedestrian_nav.yaml",
        # "yaml": "__igibson_configs__/debug.yaml",
        # "yaml": os.path.join(configs_path, "turtlebot_static_nav.yaml")
        "action_timestep": 1/5.,
        "physics_timestep": 1/120.,
        "pb": False,
        "rl_yaml": "__experiment_configs__/22_08_25__final.yml",
        "model_path": "",
    }

    # Help for the argparser
    _help = {
        "gui": "Set flag to display GUI",
        "runs": "Number of episodes to run",
        "save_path": "Where the trajectories from each episode are saved",
        "actor":  f"Actor to use.  Current options are:   {_optional_actors}",
        "seed": "Set the seed of randomness",
        "yaml":  "iGibson yaml config file",
        "action_timestep":  "iGibson time between calling 'step'.  Must be a multiple of physics_timestep",
        "physics_timestep":  "iGibson time between the physics simulation.  Must be a multiple of action_timestep",
        "pb":  "Launch Pybullet GUI",
        "rl_yaml": "yaml file specific for rl training and model configs",
        "model_path": "The model path for the rl_policy actor to load from"
    }

    def __init__(self):
        """
        Create an Config object with default values
        Info on different initializers:
            https://stackoverflow.com/questions/5738470/whats-an-example-use-case-for-a-python-classmethod
        """
        for key in self._default_values.keys():
            self[key] = self._default_values[key]

        # Config must be initialized by calling:
        #   self.setup()
        self.initialized = False

    @classmethod
    def cmd_line_load(cls):
        """
        Create a config object from the command line.
        """
        instance = cls()
        parser = argparse.ArgumentParser(description='Generate training data from iGibson Simulator.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # Add arguments/help, anything in the config
        for key in instance.keys():
            if key in instance._help.keys():
                help_str = instance._help[key]
            else:
                help_str = ""

            if type(instance._default_values[key]) == bool:
                # Add as just a flag if the default type is boolean
                parser.add_argument("--" + str(key), action='store_true', help=help_str)
            else:
                parser.add_argument("--" + str(key), type=type(instance._default_values[key]), 
                                    default=instance._default_values[key], help=help_str)

        # Get the arguments from the command line
        args = vars(parser.parse_args())

        # Post-processing on arguments


        # Assign the command line arguments to the configuration
        for key in args.keys():
            if args[key] is not None:
                instance[key] = args[key]

        # Run setup on the instance
        instance.setup()

        return instance

    def setup(self):
        """
        Setup things like creating the save directory.
        Allow access to elements in the config.
        """
        self.initialized = True

        # Seed random generator
        if self["seed"] is not None:
            g2u.set_random_seed(self["seed"])

        # Create the hdf5 save file
        if len(self["save_path"]) > 0:
            self["save_path"] = os.path.expanduser(self["save_path"])
            g2u.make_folder(self["save_path"])
            # Create the save file
            filename = "dataset_" + g2u.get_timestamp() + ".hdf5"
            self["save_file"] = os.path.join(self["save_path"], filename)
            # self["save_file"] = h5py.File(os.path.join(self["save_path"], filename), 'a')
        else:
            self["save_path"] = None
            self["save_file"] = None


        # Load the iGibson yaml file.  Store in self.yaml_data
        self["yaml"] = os.path.expanduser(self["yaml"])
        self.yaml_data = yaml.load(open(self["yaml"], "r"), Loader=yaml.FullLoader)

        if "action_freq" in self.yaml_data:
            self["action_timestep"] = 1/self.yaml_data["action_freq"]
            print("Config: action_timestep updated to {} s!".format(self["action_timestep"]))

        if "action_freq_robot" in self.yaml_data:
            self["action_timestep_robot"] = 1/self.yaml_data["action_freq_robot"]
            print("Config: action_timestep_robot updated to {} s!".format(self["action_timestep_robot"]))

        if "physics_freq" in self.yaml_data:
            self["physics_timestep"] = 1/self.yaml_data["physics_freq"]
            print("Config: physics_timestep updated to {} s!".format(self["physics_timestep"]))


        # add_h5py_meta_data(self["save_file"], self)


    def __setitem__(self, key, item):
        """
        Ensure that the item has a key in the dictionary

        Interesting list of functions part of this class (since it inherits from dict):
            https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
        """
        # if key not in self._default_values.keys():
        #     raise KeyError("No Default Value for Config option:  ", key)

        # Note: self[key] = item, or infinite recursion
        super().__setitem__(key, item)

    def __getitem__(self, key):
        """
        Access items from the config dictionary
        """
        if not self.initialized:
            raise RuntimeError("Configuration Not Initialized.  Run Config.setup()")
        return super().__getitem__(key)


if __name__ == "__main__":
    # print("Loading config from command line")
    c = Config.cmd_line_load()
    print("Config Settings:  ")
    print(c)
