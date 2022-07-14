# Child class
import numpy as np
from camera import Cameras
import ipaddress
import cv2 as cv
import time
from harvesters.core import Harvester
from filters import run as runFilter
from evaluations import objectCount


class GenieColor(Cameras):
    # Init class
    def __init__(self, cam=None, cams=None, splitted_ROI="", frame_resized=None, vimba=None, result=False, leds=-1,
                 address="", frame=None, filtered_image=None, threshold=0, cropped_image=None, frame_opencv=None,
                 applyROI=False, led_number=None, ROI=None,
                 green_lb=np.array([50, 50, 20]), green_ub=np.array([80, 255, 255]),
                 red_lb=np.array([1, 50, 10]), red_ub=np.array([20, 255, 255]), timeout=0,):
        super().__init__(address, frame, filtered_image, threshold, cropped_image, frame_opencv, applyROI, led_number,
                         ROI, green_lb, green_ub, red_lb, red_ub, timeout)
        self.set_cam(cam)
        self.set_cams(cams)
        self.set_splitted_roi(splitted_ROI)
        self.set_frame_resized(frame_resized)
        self.set_vimba(vimba)
        self.set_result(result)
        self.set_leds(leds)
        # Added
        self.harvester = Harvester()
        self.cti_path = 'C:\\LTS\\app-tde-luminaires\\SourceCode\\python_environment\\Instruments\\Camera\\cti\\mvGenTLProducer.cti'
        self.width = 2448
        self.height = 2048
        self.pixel_format = 'BayerRG8'

    # Getters and Setters
    def set_cti_path(self, cti_path):
        self.cti_path = cti_path
        return

    def get_cti_path(self):
        return self.cti_path

    def set_cam(self, cam):
        self.cam = cam
        return

    def get_cam(self):
        return self.cam

    def set_cams(self, cams):
        self.cams = cams
        return

    def get_cams(self):
        return self.cams

    def set_splitted_roi(self, splitted_roi):
        self.splitted_roi = splitted_roi
        return

    def get_splitted_roi(self):
        return self.splitted_roi

    def set_frame_resized(self, frame_resized):
        self.frame_resized = frame_resized
        return

    def get_frame_resized(self):
        return self.frame_resized

    def set_vimba(self, vimba):
        self.vimba = vimba
        return

    def get_vimba(self):
        return self.vimba

    def set_result(self, result):
        self.result = result
        return

    def get_result(self):
        return self.result

    def set_leds(self, leds):
        self.leds = leds
        return

    def get_leds(self):
        return self.leds

    # Override parent methods
    def open(self, address):
        self.harvester.add_file(self.cti_path)
        self.harvester.update()
        self.cam = self.harvester.create({'user_defined_name': f'{address}'})
        self.cam.remote_device.node_map.Width.value = self.width
        self.cam.remote_device.node_map.Height.value = self.height
        self.cam.remote_device.node_map.PixelFormat.value = self.pixel_format
        self.cam.start()
        if self.cam is None:
            raise Exception('Address not valid.')

    def set_data(self, exposure, threshold, ROI, led_number, apply_ROI, fixture_size):
        self.ROI = ROI
        self.splited_ROI = ROI.split(",")
        self.splited_ROI = [int(n) for n in self.splited_ROI]
        self.cam.remote_device.node_map.ExposureTime.value = exposure
        self.exposure = exposure
        #exposure_feat = cam.get_feature_by_name('ExposureTimeAbs')
        #exposure_feat.set(val=exposure)
        self.threshold = threshold
        self.led_number = led_number
        self.applyROI = apply_ROI

    def get_image(self):
        buffer = self.cam.fetch()
        component = buffer.payload.components[0]
        _1d = component.data
        _2d = component.data.reshape(
             component.height, component.width
        )
        print(buffer)
        assert isinstance(_2d, np.ndarray)
        self.frame = cv.cvtColor(_2d, cv.COLOR_BayerRG2RGB)
        self.frame_opencv = self.frame
        buffer.queue()

    def get_continuous_image(self):

        elapsed = 0
        start = round(time.time())

        def frame_handler(cam, frame):
            # print('Frame acquired: {}'.format(frame), flush=True)
            self.cam.queue_frame(frame)
            self.frame_queue.put(frame)

        with self.vimba:
            with self.cam as cam:
                cam.start_streaming(cam, frame_handler)
                while elapsed < self.timeout:
                    elapsed = round(time.time()) - start
                cam.stop_streaming()
        return

    def to_opencv(self, frame):
        frame_BayerGR8 = frame.as_numpy_ndarray()
        frame_opencv = cv.cvtColor(frame_BayerGR8, cv.COLOR_BayerGR2RGB)
        return frame_opencv

    def apply_ROI(self):
        if self.applyROI is True:
            self.cropped_image = runFilter(self.frame_opencv, "crop", f"{self.ROI}", False, timeout=0)
        else:
            self.cropped_image = self.frame_opencv

    def apply_filters(self):
        self.filtered_image = runFilter(self.cropped_image, "threshold", f"{self.threshold},255,binary", False,
                                        timeout=0)

    def count_leds(self):
        leds = objectCount(self.filtered_image, "RETR_EXTERNAL", "CHAIN_APPROX_NONE")
        self.leds = leds
        if leds == self.led_number:
            self.result = True
            self.leds = leds

    def get_data(self):
        return self.leds, self.result

    def show_image(self, frame, scale, timeout):
        Width = int(frame.shape[1] * scale)  # Resize Width to 50%
        Height = int(frame.shape[0] * scale)  # Resize Height to 50%
        frame_resized = cv.resize(frame, (Width, Height))  # Resize image
        if timeout >= 0:
            cv.imshow("frame", frame_resized)
            cv.waitKey(timeout * 1000)
            cv.destroyAllWindows()
        else:
            pass

    def close(self):
        self.cam.stop()
        self.cam.destroy()
        self.harvester.reset()
