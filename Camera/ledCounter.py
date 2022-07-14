from optparse import OptionParser
from camera import *
from makomono import *
import configparser

# Utilities # ToDo Add to Misc
def roi_to_str(list_of_roi):
    roi = ""
    for POINT in list_of_roi:
        roi = roi + "," + POINT[1:-1]
    roi = roi[1:]
    return roi


# Execution parameters
parser = OptionParser()
parser.add_option("-c", "--cam", type=str,
                  default="",
                  help="Cam number used in ini files")

parser.add_option("-p", "--cicode", type=str,
                  default="",
                  help="IP Address from Ethernet Camera")

parser.add_option("-a", "--applyroi", type=str,
                  default='False',
                  help="Apply ROI flag")

parser.add_option("-l", "--lednumber", type=int,
                  default=0,
                  help="LED target number to count")

parser.add_option("-g", "--timeout", type=int,
                  default=5,
                  help="Timeout to show image")

parser.add_option("-v", "--scale", type=float,
                  default=0.5,
                  help="Scale to resize image to show")

parser.add_option("-s", "--fixturesize", type=str,
                  default="",
                  help="Fixture size")

(options, args) = parser.parse_args()

# Read ini information
config = configparser.ConfigParser()
config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/InstrumentConfig.ini')
INSTRUMENT_TYPE = config[options.cam]["InstrumentType"][1:-1]
ADDRESS = config[options.cam]["Address"]
config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/VisionMtto.ini')
POINTS_OF_ROI = [(config[options.cam]['ROI.Left']), (config[options.cam]['ROI.Top']), (config[options.cam]['ROI.Right']),
                 (config[options.cam]['ROI.Bottom'])]
try:
    EXPOSURE = config[options.cicode]["Exposure"][1:-1]
    THRESHOLD = config[options.cicode]["Threshold"][1:-1]
except:
    EXPOSURE = config[options.cam]["Exposure"][1:-1]
    THRESHOLD = config[options.cam]["Threshold"][1:-1]
    
# Roi to str
POINTS_OF_ROI = roi_to_str(POINTS_OF_ROI)

# Fixture size & number of leds from argument
FIXTURE_SIZE = options.fixturesize
NUMBER_OF_LEDS = options.lednumber

if options.applyroi == 'True':
    APPLY_ROI = True
else:
    APPLY_ROI = False

TIMEOUT = options.timeout
SCALE = options.scale

# Select class
cam = None

if INSTRUMENT_TYPE == "MakoMono":
    cam = MakoMono()
elif INSTRUMENT_TYPE == "MakoColor":
    cam = MakoColor()
else:
    cam = Cameras()

# Query test for led count

cam.open(ADDRESS)
THRESHOLD = int(THRESHOLD)
cam.set_data(exposure=EXPOSURE,
             threshold=THRESHOLD,
             fixture_size=FIXTURE_SIZE,
             led_number=NUMBER_OF_LEDS,
             apply_ROI=APPLY_ROI,
             ROI=POINTS_OF_ROI)

cam.get_image()
cam.apply_ROI()
cam.apply_filters()
cam.count_leds()
leds, result = cam.get_data()
print(leds)
cam.show_image(scale=SCALE, timeout=TIMEOUT)
cam.close()
