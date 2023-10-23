import sys
from configparser import ConfigParser
import os


class RuntimeContext(object):
    """ runtime enviroment
    """

    def __init__(self):
        """ initialization
        """
        # configuration initialization
        config_parser = ConfigParser()
        config_file = self.get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()
        
        file_section = sections[0]
        self.data_file_r = config_parser.get(file_section, "data_file_r")
        self.data_file_s = config_parser.get(file_section, "data_file_s")
        self.data_file_r_local = config_parser.get(file_section, "data_file_r_local")
        self.data_file_s_local = config_parser.get(file_section, "data_file_s_local")
        self.save_dir = config_parser.get(file_section, "save_dir")
        self.output_dir = config_parser.get(file_section, "output_dir")

        base_section = sections[1]
        self.knob = config_parser.get(base_section, "knob")
        self.jar_path = config_parser.get(base_section, "jar_path")
        self.java_path = config_parser.get(base_section, "java_path")
        self.k = config_parser.get(base_section, "k")


    def get_config_file_name(self):
        """ get the configuration file name according to the command line parameters
        """
        argv = sys.argv
        config_type = "dev"  # default configuration type
        if None != argv and len(argv) > 1:
            config_type = argv[1]
        config_file = 'config/' + config_type + ".cfg"
        # logger.("get_config_file_name() return : " + config_file)
        return config_file
