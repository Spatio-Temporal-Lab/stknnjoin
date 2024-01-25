import sys
from configparser import ConfigParser
import os


def get_config_file_name():
    """ get the configuration file name according to the command line parameters
    """
    argv = sys.argv
    config_type = "dev"  # default configuration type
    if not argv and len(argv) > 1:
        config_type = argv[1]
    config_file = 'config/' + config_type + ".cfg"
    return config_file


class RuntimeContext(object):
    def __init__(self):
        """ initialization
        """
        # configuration initialization
        self.logger = None
        config_parser = ConfigParser()
        config_file = get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()

        file_section = sections[0]
        self.data_file_r = config_parser.get(file_section, "data_file_r")
        self.data_file_s = config_parser.get(file_section, "data_file_s")
        self.data_file_r_local = config_parser.get(file_section, "data_file_r_local")
        self.data_file_s_local = config_parser.get(file_section, "data_file_s_local")
        self.save_dir = config_parser.get(file_section, "save_dir")
        self.output_dir = config_parser.get(file_section, "output_dir")
        self.data_type = config_parser.get(file_section, "data_type", fallback=self.output_dir.split('/')[1])
        self.nodes = config_parser.get(file_section, "nodes", fallback=5)

        base_section = sections[1]
        self.knob = config_parser.get(base_section, "knob")
        self.jar_path = config_parser.get(base_section, "jar_path")
        self.java_path = config_parser.get(base_section, "java_path")
        self.k = config_parser.get(base_section, "k")
