# Child class
from camera import Cameras
import ipaddress
import cv2 as cv
from vimba import *
from filters import run as runFilter
from evaluations import objectCount
import numpy as np
import copy
import time


class MakoMono(Cameras):
    # Init class
    def __init__(self, cam=None, cams=None, splitted_ROI="", frame_resized=None, vimba=None, result=False, leds=-1,
                 address="", frame=None, filtered_image=None, threshold=0, cropped_image=None, frame_opencv=None, applyROI=False, led_number=None, ROI=None,
                 green_lb=np.array([50,20,20]), green_ub=np.array([100,255,255]),
                 red_lb=np.array([0,50,50]), red_ub=np.array([10,255,255]), timeout=0):
        super().__init__(address, frame, filtered_image, threshold, cropped_image, frame_opencv, applyROI, led_number, ROI, green_lb, green_ub, red_lb, red_ub, timeout)
        self.set_cam(cam)
        self.set_cams(cams)
        self.set_splitted_roi(splitted_ROI)
        self.set_frame_resized(frame_resized)
        self.set_vimba(vimba)
        self.set_result(result)
        self.set_leds(leds)
        # Connect - chk if open
        try:
            with Vimba.get_instance() as vimba:
                self.set_vimba(vimba)
        except IndexError:
            raise IndexError("Camera Not Found")
        except VimbaCameraError:
            raise VimbaCameraError("Access is Denied")


    # Getters and Setters

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

    def set_exposure(self, exposure):
        with self.vimba:
            with self.cam as cam:
                exposure_feat = cam.get_feature_by_name('ExposureTimeAbs')
                exposure_feat.set(val=exposure)
                return

    def set_gamma(self, gamma):
        with self.vimba:
            with self.cam as cam:
                gamma_feat = cam.get_feature_by_name('Gamma')
                gamma_feat.set(val=gamma)
                return

    # Override parent methods
    def open(self, address):
        with self.vimba as vimba:
            cams = vimba.get_all_cameras()
            for cam in cams:
                with cam:
                    ip_feature = cam.get_feature_by_name('GevCurrentIPAddress')
                    ip_int = ip_feature.get()
                    ip_str = str(ipaddress.IPv4Address(ip_int))
                    if ip_str == address:
                        # print(f'Connection Success with {ip_str} address')
                        self.cam = cam
                        break
                    else:
                        # print(f"Address: {ip_str} not valid")
                        pass
            if self.cam is None:
                 raise Exception('Address not valid.')

    def set_data(self, exposure, threshold, ROI, led_number, apply_ROI, fixture_size):
        with self.vimba:
            with self.cam as cam:
                self.ROI = ROI
                self.splitted_ROI = ROI.split(",")
                self.splitted_ROI = [int(n) for n in self.splitted_ROI]
                exposure_feat = cam.get_feature_by_name('ExposureTimeAbs')
                exposure_feat.set(val=exposure)
        self.threshold = threshold
        self.led_number = led_number
        self.applyROI = apply_ROI

    def get_image(self):
        with self.vimba:
            with self.cam as cam:
                self.frame = cam.get_frame()
                self.frame_opencv = self.frame.as_opencv_image()  # Visualize image with OpenCV

    def get_continuous_image(self):

        elapsed = 0
        start = round(time.time())

        def frame_handler(cam, frame):
            # print('Frame acquired: {}'.format(frame), flush=True)
            self.cam.queue_frame(frame)
            self.frame_queue.put(frame)

        with self.vimba:
            with self.cam as cam:
                cam.start_streaming(frame_handler)
                while elapsed < self.timeout:
                    elapsed = round(time.time()) - start
                cam.stop_streaming()
        return

    def to_opencv(self, frame):
        frame_opencv = frame.as_opencv_image()
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
        try:
            Width = int(self.filtered_image.shape[1] * scale)  # Resize Width to 50%
            Height = int(self.filtered_image.shape[0] * scale)  # Resize Height to 50%
            self.frame_resized = cv.resize(self.filtered_image, (Width, Height))  # Resize image
            if timeout >= 0:
                cv.imshow("frame", self.frame_resized)
                cv.waitKey(timeout * 1000)
                cv.destroyAllWindows()
            else:
                pass
        except:
            Width = int(frame.shape[1] * scale)  # Resize Width to 50%
            Height = int(frame.shape[0] * scale)  # Resize Height to 50%
            self.frame_resized = cv.resize(frame, (Width, Height))  # Resize image
            if timeout >= 0:
                cv.imshow("frame", self.frame_resized)
                cv.waitKey(timeout * 1000)
                cv.destroyAllWindows()
            else:
                pass

    def apply_offset(self, show_result: bool):

        def apply_gaussian_blur():
            img = copy.deepcopy(self.frame_opencv)
            kernel_size = 5
            blur_gray = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
            return blur_gray

        def apply_canny(img):
            low_threshold = 50
            high_threshold = 150
            edges = cv.Canny(img, low_threshold, high_threshold)
            return edges

        def get_lines(img):

            def show_lines(img, line_image):
                lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
                self.show_image(frame=lines_edges, scale=0.2, timeout=15)
                return

            rho = 1
            theta = np.pi / 180
            threshold = 15
            min_line_length = 1500
            max_line_gap = 1500
            line_image = np.copy(img) * 0
            lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            try:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                if show_result:
                    show_lines(img, line_image)
                return lines
            except:
                return []

        self.show_image(self.frame_opencv, 0.5, timeout=5)
        image = apply_canny(apply_gaussian_blur())
        lines_at_image = get_lines(image)
        return len(lines_at_image)

    def fill_and_count(self):
        try:
            img = copy.deepcopy(self.frame)
            hh, ww = img.shape[:2]
            thresh = cv.threshold(img, 254, 255, cv.THRESH_BINARY)[1]
            # self.show_image(frame=thresh, timeout=5, scale=0.5)
            # Get largest contour
            contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(big_contour)
            perimeter = cv.arcLength(big_contour, True)
            # draw white filled
            result = np.zeros_like(img)
            cv.drawContours(result, [big_contour], 0, (255, 255, 255), 1)
            # save
            rot_fig = cv.minAreaRect(big_contour)
            (center), (width, height), angle = rot_fig
            im_bgr = cv.vconcat([self.frame, result])
            # self.show_image(im_bgr, scale=0.1, timeout=1)
        except:
            area = 0
            perimeter = 0
        return area, perimeter


    def close(self):
        del (self.vimba)
