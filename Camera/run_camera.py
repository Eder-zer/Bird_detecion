"""
Child class from tde_runner
specific runner for camera methods (Parent-child)
For running methods self.set_queue("command") must be called outside this module after
self.run() method was called as a thread
eder.zermeno@acuitybrands.com
"""
from makocolor import MakoColor
from camera import Cameras
from makomono import *
import configparser
import optparse
import sys
import os
# ToDo last edits
from write_txt import LabviewIpc
lv_ipc = LabviewIpc()
#ToDo fix import
from Tde_runner import *


# ToDo add to MISC library
def bound_to_list(list_str):
    li = list(list_str.split(","))
    li_int = []
    for l in li:
        li_int.append(int(l))
    return li_int

def roi_to_str(list_of_roi):
    roi = ""
    for POINT in list_of_roi:
        roi = roi + "," + POINT[1:-1]
    roi = roi[1:]
    return roi

class runner_cam(tde_runner):

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.camera = self.user_parameters()
        # Read ini information
        config = configparser.ConfigParser()
        config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/InstrumentConfig.ini')
        self.inst_type = config[self.camera]["InstrumentType"][1:-1]
        self.address = config[self.camera]["Address"][1:-1]

        # Run class as per InstrumentConfig.ini[InstrumentType]
        if self.inst_type == "MakoMono":
            self.inst = MakoMono(address=self.address)
        elif self.inst_type == "MakoColor":
            self.inst = MakoColor(address=self.address)
        else:
            self.inst = Cameras(address=self.address)

    def user_parameters(self):
        try:
            parser = optparse.OptionParser(description='Synapse parameters')
            group = optparse.OptionGroup(parser, "Options")
            group.add_option('--camera', '-c', type="str",
                             default="Camera1",
                             help="Camera as it appears at InstrumentConfig.ini",
                             metavar="camera")
            parser.add_option_group(group)
            options, arguments = parser.parse_args()
            camera = options.camera
        except:
            camera = "Camera1"
        return camera

    def run(self):
        while self.is_running:
            # If code is running on main get input from cmd, else get it from the main threading
            if __name__ == "__main__":
                user_input = sys.stdin.readline().rstrip()
                print(f"User Input: {user_input}")
                lv_ipc.write_text(txt=f"userinput: {user_input}")
            else:
                user_input = self.get_message()
            # Continue with the execution
            user_input_filtered = user_input.upper()
            # Close console
            if user_input_filtered is None or user_input_filtered == '' or user_input_filtered == 'EXIT':
                # If message was an EXIT break the while
                break
            else:
                arguments = user_input_filtered.split('_')

                if arguments[0] == 'OPEN':
                    # Open camera instance
                    self.inst.open(self.address)
                    if __name__ == "__main__":
                        print("CAMERA_OPEN")
                    else:
                        self.set_response("CAMERA_OPEN")
                        lv_ipc.write_text(txt=f"Camera_open")

                if arguments[0] == 'LED-ENGINES':
                    try:
                        area_offset = int(arguments[1])
                    except:
                        area_offset = 68000
                    try:
                        perimeter_offset = int(arguments[2])
                    except:
                        perimeter_offset = 6300
                    try:
                        exposure = int(arguments[3])
                    except:
                        exposure = 9998
                    try:
                        timeout = int(arguments[4])
                    except:
                        timeout = 10
                    # Open camera instance
                    led_engines = self.inst.led_engines(area_offset=area_offset,
                                                        perimeter_offset=perimeter_offset,
                                                        exposure=exposure,
                                                        timeout=timeout)
                    if __name__ == "__main__":
                        print(f"CAMERA_LED-ENGINES_{led_engines}")
                        lv_ipc.write_text(txt=f"CAMERA_LED-ENGINES_{led_engines}")
                    else:
                        self.set_response(f"CAMERA_LED-ENGINES_{led_engines}")

                if arguments[0] == 'LED-COLOR':
                    # Arguments _Timeout _30
                    # Set up method before call
                    config = configparser.ConfigParser()
                    config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/VisionMtto.ini')
                    green_lb = bound_to_list(config[self.camera]['green_lb'][1:-1])
                    green_ub = bound_to_list(config[self.camera]['green_ub'][1:-1])
                    red_lb = bound_to_list(config[self.camera]['red_lb'][1:-1])
                    red_ub = bound_to_list(config[self.camera]['red_ub'][1:-1])
                    try:
                        timeout = int(arguments[1])
                    except:
                        timeout = 15
                    # Call camera method
                    result_green, result_red = self.inst.led_color_counter(timeout=timeout, green_lb=green_lb,
                                                green_ub=green_ub, red_lb=red_lb, red_ub=red_ub)
                    if __name__ == "__main__":
                        print(f"CAMERA_LED-COLOR_G_{result_green}_R_{result_red}")
                        lv_ipc.write_text(txt=f"CAMERA_LED-COLOR_G_{result_green}")
                        lv_ipc.write_text(txt=f"CAMERA_LED-COLOR_R_{result_red}")
                    else:
                        self.set_response(f"CAMERA_LED-COLOR_G_{result_green}_R_{result_red}")

                if arguments[0] == 'LED-COUNTER':

                    # Set up method before call
                    try:
                        timeout = int(arguments[1])
                    except:
                        timeout = 5
                    try:
                        leds = int(arguments[2])
                    except:
                        leds = 10
                    try:
                        cicode = arguments[3]
                    except:
                        cicode = ""
                        pass
                    try:
                        lumens = arguments[4]
                    except:
                        lumens = ""
                        pass
                    try:
                        size = arguments[5]
                    except:
                        size = ""
                        pass
                    config = configparser.ConfigParser()
                    config.read('C:/LTS/app-tde-luminaires/Deploy/StationConfig/VisionMtto.ini')
                    points_of_roi = [(config[self.camera]['ROI.Left']), (config[self.camera]['ROI.Top']),
                                     (config[self.camera]['ROI.Right']),
                                     (config[self.camera]['ROI.Bottom'])]
                    points_of_roi = roi_to_str(points_of_roi)

                    LPM = False
                    exposure = 0
                    threshold = 0

                    try:
                        section = 'lumens={}, size={}, {}'.format(str(lumens), str(size), str(self.camera))
                        exposure = int(config[section]["Exposure"][1:-1])
                        threshold = int(config[section]["Threshold"][1:-1])
                        LPM = True
                    except:
                        LMP = False
                    if not LPM:
                        try:
                            exposure = int(config[cicode]["Exposure"][1:-1])
                            threshold = int(config[cicode]["Threshold"][1:-1])
                        except:
                            exposure = int(config[self.camera]["Exposure"][1:-1])
                            threshold = int(config[self.camera]["Threshold"][1:-1])
                    # Call camera method
                    result_leds = self.inst.led_counter(exposure=exposure, threshold=threshold, points_of_roi=points_of_roi,
                                          leds=leds, timeout=timeout)
                    if __name__ == "__main__":
                        print(f"CAMERA_LED-COUNTER_{result_leds}")
                        lv_ipc.write_text(txt=f"CAMERA_LED-COUNTER_{result_leds}")
                    else:
                        self.set_response(f"CAMERA_LED-COUNTER_{result_leds}")

                elif arguments[0] == 'CLOSE':
                    self.inst.close()
                    if __name__ == "__main__":
                        print(f"CAMERA_CLOSE")
                        lv_ipc.write_text(txt=f"CameraClose")
                    else:
                        self.set_response(f"CAMERA_CLOSE")

        self.inst.close()
        sys.exit(1)

if __name__ == "__main__":
    obj = runner_cam()
    obj.run()
