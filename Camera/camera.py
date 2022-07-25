"""
* - Garza, Mauricio [mauricio.garza@acuitybrands.com]
* - Sasso, David [david.sasso@task.com.mx]
* - Zermeño, Eder [eder.zermeno@acuitybrands.com]
June,2022
"""
# Parent class
import cv2 as cv
import ipaddress
from filters import run as runFilter
from evaluations import objectCount
import time
import numpy as np
import threading, queue
import copy


class Cameras:
    # Init class
    def __init__(self, address="", frame=None, filtered_image=None, threshold=0, cropped_image=None, frame_opencv=None,
                 applyROI=False, led_number=None, ROI=None,
                 green_lb=np.array([0, 0, 0]), green_ub=np.array([0, 0, 0]),
                 red_lb=np.array([0, 0, 0]), red_ub=np.array([0, 0, 0]), timeout=0):
        self.address = address
        self.frame = frame
        self.filtered_image = filtered_image
        self.threshold = threshold
        self.cropped_image = cropped_image
        self.frame_opencv = frame_opencv
        self.applyROI = applyROI
        self.led_number = led_number
        self.ROI = ROI
        self.green_lb = green_lb
        self.green_ub = green_ub
        self.red_lb = red_lb
        self.red_ub = red_ub
        self.green_mask = None
        self.red_mask = None
        self.color_frame = None
        self.frame_queue = queue.Queue()
        self.timeout = timeout
        self.green_count = 0
        self.red_count = 0
        self.area = []
        self.perimeter = []

    # Getters and setters - Properties
    def set_green_ub(self, green_ub):
        self.green_ub = green_ub
        return

    def get_green_ub(self):
        return self.green_ub

    def set_green_lb(self, green_lb):
        self.green_lb = green_lb
        return

    def get_green_lb(self):
        return self.green_lb

    def set_red_ub(self, red_ub):
        self.red_ub = red_ub
        return

    def get_red_ub(self):
        return self.red_ub

    def set_red_lb(self, red_lb):
        self.red_lb = red_lb
        return

    def get_red_lb(self):
        return self.red_lb

    def get_address(self):
        return self.address

    def set_address(self, address):
        self.address = address
        return

    def get_frame(self):
        return self.frame

    def set_frame(self, frame):
        self.frame = frame
        return

    def get_filtered_image(self):
        return self.filtered_image

    def set_filtered_image(self, filtered_image):
        self.filtered_image = filtered_image
        return

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
        return

    def get_cropped_image(self):
        return self.cropped_image

    def set_cropped_image(self, cropped_image):
        self.cropped_image = cropped_image
        return

    def get_frame_opencv(self):
        return self.frame_opencv

    def set_frame_opencv(self, frame_opencv):
        self.frame_opencv = frame_opencv
        return

    def get_apply_roi(self):
        return self.applyROI

    def set_apply_roi(self, apply_roi):
        self.applyROI = apply_roi
        return

    def get_led_number(self):
        return self.led_number

    def set_led_number(self, led_number):
        self.led_number = led_number
        return

    def get_roi(self):
        return self.ROI

    def set_roi(self, roi):
        self.ROI = roi
        return

    # Methods
    def open(self, address):
        pass

    def set_data(self, exposure, threshold, ROI, led_number, apply_ROI, fixture_size):
        pass

    def get_image(self):
        pass

    def apply_ROI(self):
        if self.get_roi():
            self.cropped_image = runFilter(self.frame_opencv, "crop", f"{self.ROI}", False, timeout=0)
        else:
            self.cropped_image = self.frame_opencv

    def apply_filters(self):
        self.filtered_image = runFilter(self.cropped_image, "threshold", f"{self.threshold},255,binary", False,
                                        timeout=0)

    def count_leds(self):
        leds = objectCount(self.filtered_image, "RETR_EXTERNAL", "CHAIN_APPROX_NONE")
        if leds != self.led_number:
            print("Test Failed")
        elif leds == self.led_number:
            print("Test Successful")
        return leds

    def example_frame(self):
        self.get_image()
        self.show_image(self.frame,2,10)
        return

    def get_data(self):
        leds, result = 0, False
        return leds, result

    def get_continuous_image(self):
        # Child classes enqueues frames async
        return

    def show_image(self, frame, scale, timeout):
        if scale < 1 and scale > 0:
            Width = int(frame.shape[1] * scale)
            Height = int(frame.shape[0] * scale)
            frame_resized = cv.resize(frame, (Width, Height))
        else:
            raise ValueError('Scale parameter not in range.')
        if timeout >= 0:
            cv.imshow("frame", frame_resized)
            cv.waitKey(timeout)
            cv.destroyAllWindows()
        else:
            pass

    def set_exposure(self, exposure):
        return

    def set_gamma(self, gamma):
        return

    def apply_offset(self, show_result: bool):
        return

    def fill_and_count(self, frame):
        return

    def close(self):
        pass

    # Class utilities / Static Methods

    def get_frame_rgb(self):
        to_show = copy.deepcopy(self.frame)
        to_show = to_show.as_numpy_ndarray()
        to_show = cv.cvtColor(to_show, cv.COLOR_BayerGR2RGB)
        return to_show

    def set_frame_from_img(self, img_path):
        self.set_frame(cv.imread(img_path))
        return

    def frame_to_color(self):
        copied_frame = copy.deepcopy(self.frame)
        self.color_frame = cv.cvtColor(copied_frame, cv.COLOR_BayerGR2RGB)
        self.color_frame = cv.cvtColor(self.color_frame, cv.COLOR_BGR2HSV)
        # cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        return

    def crop_template(self, template):
        try:
            self.frame = self.frame.as_numpy_ndarray()
            copied_frame = copy.deepcopy(self.frame)
            copied_frame = cv.cvtColor(copied_frame, cv.COLOR_BayerGR2GRAY)
            template_img = cv.imread(template, cv.IMREAD_GRAYSCALE)
            ret, thresh_template = cv.threshold(template_img, 10, 255, cv.THRESH_BINARY)
            ret, thresh_copied = cv.threshold(copied_frame, 10, 255, cv.THRESH_BINARY)
            w, h = thresh_template.shape
            res = cv.matchTemplate(thresh_copied, thresh_template, cv.TM_CCOEFF_NORMED)
            _, _, _, maxLoc = cv.minMaxLoc(res)
            cv.rectangle(self.frame, maxLoc, (maxLoc[0]+h, maxLoc[1]+w), (0, 255, 255), 2)
            # self.show_image(self.frame,0.5,5)
            crop_img = self.frame[maxLoc[1]:maxLoc[1]+w, maxLoc[0]:maxLoc[0]+h, :]
            # self.show_image(crop_img,0.5,1)
            self.frame = crop_img
            return
        except:
            print("no template not found")
            self.frame = np.zeros(1, 1, 3)

    def generate_masks(self):
        try:
            copy_green = self.color_frame.copy()
            copy_red = self.color_frame.copy()
            # self.show_image(copy_red,0.5,1)
            copied_frame = copy.deepcopy(self.frame)
            green_mask = cv.inRange(copy_green, self.green_lb, self.green_ub)
            result = cv.bitwise_and(copied_frame, copied_frame, mask=green_mask)
            self.green_mask = result
            red_mask = cv.inRange(copy_red, self.red_lb, self.red_ub)
            result = cv.bitwise_and(copied_frame, copied_frame, mask=red_mask)
            self.red_mask = result
            # Get led color
            valid_color = 1000
            UV_on = 10000
            print(f"green: {np.sum(self.green_mask)} red: {np.sum(self.red_mask)}")
            if np.sum(self.green_mask) > valid_color:
                if np.sum(self.green_mask) > np.sum(self.red_mask):
                    if np.sum(self.green_mask) < UV_on:
                        self.green_count = self.green_count + 1
            if np.sum(self.red_mask) > valid_color:
                if np.sum(self.red_mask) > np.sum(self.green_mask):
                    if np.sum(self.red_mask) < UV_on:
                        self.red_count = self.red_count + 1
            return
        except Exception:
            return

    def consume_frames(self):
        start = round(time.time())
        elapsed = 0
        counter = 0
        while elapsed < self.timeout:
            # Get frame from camera queue
            self.frame = self.frame_queue.get()
            # Chop the image
            self.crop_template('led.png')
            # Change bayerGR8(camera) to HSV for picture analysis
            self.frame_to_color()
            # Get green/red data
            self.generate_masks()
            elapsed = round(time.time()) - start
            counter = counter + 1
        self.frame_queue.task_done()
        return

    def led_color_counter(self, timeout, green_lb, green_ub, red_lb, red_ub):
        try:
            self.green_count = 0
            self.red_count = 0
            self.timeout = timeout
            # Green limits
            self.set_green_lb(np.array(green_lb))
            self.set_green_ub(np.array(green_ub))
            # Red limits
            self.set_red_lb(np.array(red_lb))
            self.set_red_ub(np.array(red_ub))
            # Start picture analysis thread
            threading.Thread(target=self.consume_frames, daemon=True).start()
            # Call camera for frame enqueue
            self.get_continuous_image()
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            return self.green_count, self.red_count
        except:
            return 0, 0

    def led_counter(self, exposure, threshold, points_of_roi, leds, timeout):
        try:
            self.set_data(exposure=exposure, threshold=threshold, ROI=points_of_roi, led_number=leds,
                          apply_ROI=False, fixture_size="")
            self.get_image()
            self.apply_ROI()
            self.apply_filters()
            self.count_leds()
            result_leds, result = self.get_data()
            self.show_image(frame=self.frame, scale=0.5, timeout=timeout)
            return result_leds
        except:
            return 0

    def led_engines(self, area_offset, perimeter_offset, exposure, timeout):
        try:
            self.timeout = timeout
            self.set_exposure(exposure=exposure)
            self.set_gamma(gamma=4)
            # Start picture analysis thread
            threading.Thread(target=self.consume_engines, daemon=True).start()
            # Call camera for frame enqueue
            self.get_continuous_image()
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            self.set_gamma(gamma=1)
            area_max = max(self.area)
            perimeter_max = max(self.perimeter)
            result = area_max > area_offset and perimeter_max > perimeter_offset
            return result
        except:
            return False
            #lines = self.apply_offset(show_result=show_result)

    def consume_engines(self):
        start = round(time.time())
        self.area = []
        self.perimeter = []
        elapsed = 0
        counter = 0
        while elapsed < self.timeout:
            # Get frame from camera queue
            self.frame = self.frame_queue.get()
            self.frame = self.frame.as_opencv_image()
            new_area, new_perimeter = self.fill_and_count()
            self.area.append(new_area)
            self.perimeter.append(new_perimeter)
            elapsed = round(time.time()) - start
            counter = counter + 1
        area_max = max(self.area)
        perimeter_max = max(self.perimeter)
        area_min = min(self.area)
        perimeter_min = min(self.perimeter)
        print(f"areaM:{area_max} perimetroM:{perimeter_max}")
        print(f"aream:{area_min} perimetrom:{perimeter_min}")
        self.frame_queue.task_done()
        return


    def configure_async_stream(self):
        '''
        Auxiliary function to start asynchronous adquisition.

        This function shall NOT be called by its own.
        Instead call the start_async_stream() function.
        '''
        self.recording = True

        def frame_handler(cam, frame):
            return

    def start_async_stream(self):
        '''
        Starts asynchronous adquisition in a new thread.
        This function puts the camera in a streaming mode,
        to get an image from this streaming see get_async_frame() function.
        '''
        threading.Thread(target=self.configure_async_stream, daemon=True).start()

    def is_streaming(self):
        '''
        Returns the status of the camera in a streaming mode.
        '''
        with self.vimba as vimba:
            with self.cam as cam:
                is_streaming = cam.is_streaming()
        return is_streaming

    def to_numpy(self, frame):
        '''
        Converts the vimba frame into a numpy array object.
        '''
        numpy_frame = frame.as_numpy_ndarray()
        return numpy_frame

    def resize_frame(self, frame, scale):
        '''
        Resize the given frame using the scale parameter.
        '''
        assert (0 < scale and scale <= 1), 'Invalid scale parameter, it should have a value between 0 and 1.'
        Width = int(frame.shape[1] * scale)  # Resize Width to <scale>%
        Height = int(frame.shape[0] * scale)  # Resize Height to <scale>%
        resized_frame = cv.resize(frame, (Width, Height))  # Resize image
        return resized_frame

    def get_async_frame(self):
        '''
        Returns a frame from the current streaming.
        If camera is not in stream mode returns None.
        '''
        try:
            frame = self.frame_queue.get()
        except:
            frame = None
        return frame

    def stop_async_stream(self):
        '''
        Stops the streaming from the camera.
        This function also stop the adquisition thread using self.recording attribute,
        and clears the queue object where the frame sare beign stored.
        '''
        self.recording = False
        # with self.vimba as vimba:
        #    with self.cam as cam:
        #        cam.stop_streaming()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        return

