from makocolor import MakoColor
from optparse import OptionParser
from camera import Cameras
from makomono import *
import configparser

# ToDo Add to Misc
def bound_to_list(list_str):
    li = list(list_str.split(","))
    li_int = []
    for l in li:
        li_int.append(int(l))
    return li_int


parser = OptionParser()
parser.add_option("-c", "--cam", type=str,
                  default="Camera1",
                  help="Cam number used in ini files")
parser.add_option("-g", "--timeout", type=int,
                  default=5,
                  help="Timeout to show image")
(options, args) = parser.parse_args()

TIMEOUT = options.timeout

# Read ini information
config = configparser.ConfigParser()
config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/InstrumentConfig.ini')
INSTRUMENT_TYPE = config[options.cam]["InstrumentType"][1:-1]
ADDRESS = config[options.cam]["Address"][1:-1]
config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/VisionMtto.ini')
green_lb = bound_to_list(config[options.cam]['green_lb'][1:-1])
green_ub = bound_to_list(config[options.cam]['green_ub'][1:-1])
red_lb = bound_to_list(config[options.cam]['red_lb'][1:-1])
red_ub = bound_to_list(config[options.cam]['red_ub'][1:-1])

if INSTRUMENT_TYPE == "MakoMono":
    cam = MakoMono(address=ADDRESS, timeout=TIMEOUT)
elif INSTRUMENT_TYPE == "MakoColor":
    cam = MakoColor(address=ADDRESS, timeout=TIMEOUT)
else:
    cam = Cameras(address=ADDRESS, timeout=TIMEOUT)

cam.open(address=ADDRESS)
cam.led_color_counter(timeout=TIMEOUT,green_lb=green_lb, green_ub=green_ub, red_lb=red_lb, red_ub=red_ub)
