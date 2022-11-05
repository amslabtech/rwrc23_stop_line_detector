#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage

import cv2
import itertools
import math
import numpy as np
import random

class StopLineDetector:
    _hz: float
    _trans_upper_left: float
    _trans_upper_right: float
    _line_grad_th: float
    _line_length_th: int
    _close_line_th: int
    _cand_hsv_lo: list
    _cand_hsv_hi: list
    _rect_w_lo: int
    _rect_h_lo: int
    _rect_h_hi: int
    _whiteness_th: float
    _pub: rospy.Publisher
    _sub: rospy.Subscriber
    _input_image: np.ndarray
    _compressed_image: CompressedImage

    def __init__(self):
        rospy.init_node("stop_line_detector", anonymous=True)

        self._hz = rospy.get_param("~hz", 15.0)
        self._trans_upper_left = rospy.get_param("~trans_upper_left", 1/3)
        self._trans_upper_right = rospy.get_param("~trans_upper_right", 2/3)
        self._line_grad_th = rospy.get_param("~line_grad_th", 2/3)
        self._line_length_th = rospy.get_param("~line_length_th", 10)
        self._close_line_th = rospy.get_param("~close_line_th", 7)
        self._cand_hsv_lo = [int(n) for n in rospy.get_param("~cand_hsv_lo", [0,0,0])]
        self._cand_hsv_hi = [int(n) for n in rospy.get_param("~cand_hsv_hi", [180,255,255])]

        self._rect_w_lo = rospy.get_param("~rect_w_lo", 150)
        self._rect_h_lo = rospy.get_param("~rect_h_lo", 30)
        self._rect_h_hi = rospy.get_param("~rect_h_hi", 60)
        self._whiteness_th = rospy.get_param("~whiteness_th", 3.0)

        self._pub = rospy.Publisher("/stop_line_image/compressed", CompressedImage, queue_size=1, tcp_nodelay=True)
        # self._sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self._compressed_image_callback, queue_size=1, tcp_nodelay=True)
        self._sub = rospy.Subscriber("/realsense/color/image_raw/compressed", CompressedImage, self._compressed_image_callback, queue_size=1, tcp_nodelay=True)
        self._input_image = np.empty(0)
        self._compressed_image = CompressedImage()
        self._compressed_image.format = "jpeg"

    def _compressed_image_callback(self, data: CompressedImage):
        self._input_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)
        self._compressed_image.header = data.header
        self._run()

    def _run(self):
        trans_img = self._image_trans(self._input_image)
        lines = self._detect_lines(trans_img)
        connected_lines = self._connect_close_lines(lines)
        result_img = self._detect_whiteline(trans_img, connected_lines)

        self._compressed_image.data = cv2.imencode(".jpg", result_img)[1].squeeze().tolist()
        self._pub.publish(self._compressed_image)

    def _image_trans(self, img):
        p1 = np.array([math.floor(img.shape[1] * self._trans_upper_left), 0])  # param
        p2 = np.array([math.floor(img.shape[1] * self._trans_upper_right), 0])  # param
        p3 = np.array([0, img.shape[0]-1])
        p4 = np.array([img.shape[1]-1, img.shape[0]-1])
        dst_width = math.floor(np.linalg.norm(p2 - p1) * 1.0)
        dst_height = math.floor(np.linalg.norm(p3 - p1))
        trans_src = np.float32([p1, p2, p3, p4])
        trans_dst = np.float32([[0, 0],[dst_width, 0],[0, dst_height],[dst_width, dst_height]])

        trans_mat = cv2.getPerspectiveTransform(trans_src, trans_dst)
        trans_img = cv2.warpPerspective(img, trans_mat, (dst_width, dst_height))

        return trans_img

    def _get_line_coordinate(self, line):
        return int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])

    def _calc_endpoints_distance(self, src, dst):
        dists = (
            math.hypot(src[0] - dst[0], src[1] - dst[1]),
            math.hypot(src[0] - dst[2], src[1] - dst[3]),
            math.hypot(src[2] - dst[0], src[3] - dst[1]),
            math.hypot(src[2] - dst[2], src[3] - dst[3]),
        )

        return min(dists)

    def _search_close_lines(self, src_lines, dst_lines):
        close_lines = []
        for dst in dst_lines:
            if dst in src_lines:
                continue
            dists = []
            for src in src_lines:
                dists.append(self._calc_endpoints_distance(src, dst))
            if min(dists) < self._close_line_th:  # param
                close_lines.append(dst)

        return close_lines

    def _connect_lines(self, lines):
        endpoints = []
        for line in lines:
            endpoints.extend([(line[0], line[1]), (line[2], line[3])])
        start = min(endpoints, key=lambda a: a[0])
        end = max(endpoints, key=lambda a: a[0])

        return start[0], start[1], end[0], end[1]

    def _detect_lines(self, img):
        ##preprpceeding
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl_img = clahe.apply(gray_img)
        prep_img = cv2.GaussianBlur(cl_img, (3,3), 0)

        ##detect
        detector = cv2.createLineSegmentDetector()
        lines, _, _, _ = detector.detect(prep_img)
        lines = lines.tolist()

        ##visualize
        # detected_img = trans_img.copy()
        #
        # for line in lines:
        #     x1, y1, x2, y2 = get_line_coordinate(line)
        #     color = [random.randint(0, 255) for _ in range(3)]
        #     detected_img = cv2.line(detected_img, (x1, y1), (x2, y2), color, 2)

        ##filter
        filtered_lines = []

        for line in lines:
            x1, y1, x2, y2 = self._get_line_coordinate(line)
            line_grad = (y2 - y1) / (x2 - x1 + 1e-10)
            line_length = math.hypot(x2 - x1, y2 - y1)
            if abs(line_grad) < self._line_grad_th and line_length > self._line_length_th:  # param
                filtered_lines.append((x1, y1, x2, y2))
                # color = [random.randint(0, 255) for _ in range(3)]
                # detected_img = cv2.line(detected_img, (x1, y1), (x2, y2), color, 2)

        return filtered_lines

    def _connect_close_lines(self, input_lines):
        ##connect
        lines = None
        close_lines = []
        grouped_lines = []
        connected_lines = []

        for line in input_lines:
            if line in grouped_lines:
                continue
            while True:
                lines = [line] if lines is None else lines
                close_lines = self._search_close_lines(lines, input_lines)
                if len(close_lines):
                    lines.extend(close_lines)
                grouped_lines.extend(lines)
                if len(close_lines) == 0:
                    connected_lines.append(self._connect_lines(lines))
                    lines = None
                    break

        # connected_img = trans_img.copy()
        # for x1, y1, x2, y2 in connected_lines:
        #     color = [random.randint(0, 255) for _ in range(3)]
        #     connected_img = cv2.line(connected_img, (x1, y1), (x2, y2), color, 2)

        return connected_lines

    def _detect_whiteline(self, img, lines):
        ##handpick
        candidate_imgs = []
        candidate_areas = []
        mean_brightnesses = []

        for i, line_a in enumerate(lines):
            for j, line_b in enumerate(lines):
                if j <= i:
                    continue

                candidate_area = (
                    min(line_a[0], line_a[2], line_b[0], line_b[2]),
                    max(line_a[0], line_a[2], line_b[0], line_b[2]),
                    min(line_a[1], line_a[3], line_b[1], line_b[3]),
                    max(line_a[1], line_a[3], line_b[1], line_b[3]))
                candidate_img = img.copy()[candidate_area[2]:candidate_area[3], candidate_area[0]:candidate_area[1]]
                if candidate_img.shape[0] < 1 or candidate_img.shape[1] < 1:
                    continue

                candidate_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2HSV)
                candidate_img = cv2.inRange(candidate_img, tuple(self._cand_hsv_lo), tuple(self._cand_hsv_hi))  # param
                # candidate_img = cv2.inRange(candidate_img, (150, 150, 150), (255, 255, 255))  # RGB
                candidate_imgs.append(candidate_img)
                candidate_areas.append(candidate_area)
                mean_brightness = candidate_img.mean()
                mean_brightnesses.append(mean_brightness)

        mean_all_brightness = np.array(mean_brightnesses).mean() + 1e-6

        ##debug
        # for br in (mean_brightnesses):
        #     print(f"mean_all_brightness: {mean_all_brightness} \n")
        #     print(f"brightness: {br} , whiteness: {br/mean_all_brightness}\n\n")
        #     cv2.imshow('cand_img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        ##result
        result_img = img.copy()
        whiteness_max = 0.0
        for img, area, br in zip(candidate_imgs, candidate_areas, mean_brightnesses):
            whiteness = br / mean_all_brightness
            # print(whiteness)
            if self._rect_h_lo < img.shape[0] < self._rect_h_hi and self._rect_w_lo < img.shape[1] and whiteness > self._whiteness_th:  # param
                result_img = cv2.rectangle(result_img, (area[0], area[2]), (area[1], area[3]), (0, 255, 0), 2)

        return result_img

    def __call__(self):
        # duration = int(1.0 / self._hz * 1e6)
        # rospy.Timer(rospy.Duration(nsecs=duration), self._compressed_image_callback)
        rospy.spin()


def main():
    StopLineDetector()()

if __name__ == "__main__":
    main()

    ##output
    # print(f"\nwhiteness_max : {whiteness_max}\n")
    # cv2.imshow('result', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
