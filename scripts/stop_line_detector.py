#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

import cv2
import itertools
import math
import numpy as np
import random

##pip install "ocrd-fork-pylsd == 0.0.3" (python3)
from pylsd.lsd import lsd

class StopLineDetector:
    _hz: float
    _trans_upper_left: list
    _trans_upper_right: list
    _stop_level: int
    _line_grad_th: float
    _line_length_th: int
    _close_line_th: int
    _hsv_lo: list
    _hsv_hi: list
    _split_num: int
    _resize_num: int
    _aspect_th: float
    _whiteness_th: float
    _texture_th: float
    _smoothness_th: float
    _pub_image: rospy.Publisher
    _pub_stop_line_flag: rospy.Publisher
    _sub_flag: rospy.Subscriber
    _sub_img: rospy.Subscriber
    _detect_flag: bool
    _stop_area: int
    _input_image: np.ndarray
    _compressed_image: CompressedImage
    _visualize: bool

    def __init__(self):
        rospy.init_node("stop_line_detector", anonymous=True)

        self._hz = rospy.get_param("~hz", 15.0)
        self._trans_upper_left = [int(n) for n in rospy.get_param("~trans_upper_left", [0,0])]
        self._trans_upper_right = [int(n) for n in rospy.get_param("~trans_upper_right", [680,0])]
        self._stop_level = rospy.get_param("~stop_level", 0)
        self._line_grad_th = rospy.get_param("~line_grad_th", 2/3)
        self._line_length_th = rospy.get_param("~line_length_th", 10)
        self._close_line_th = rospy.get_param("~close_line_th", 7)
        self._hsv_lo = [int(n) for n in rospy.get_param("~hsv_lo", [0,0,0])]
        self._hsv_hi = [int(n) for n in rospy.get_param("~hsv_hi", [180,255,255])]
        self._split_num = rospy.get_param("~split_num", 10)
        self._resize_num = rospy.get_param("~resize_num", 30)
        self._aspect_th = rospy.get_param("~aspect_th", 1.0)
        self._whiteness_th = rospy.get_param("~whiteness_th", 0.5)
        self._texture_th = rospy.get_param("~texture_th", 10)
        self._smoothness_th = rospy.get_param("~smoothness_th", 0.5)
        self._visualize = rospy.get_param("~visualize", False)

        self._pub_image = rospy.Publisher("/stop_line_image/compressed", CompressedImage, queue_size=1, tcp_nodelay=True)
        self._pub_stop_line_flag = rospy.Publisher("/stop_line_flag", Bool, queue_size=1, tcp_nodelay=True)
        self._sub_flag = rospy.Subscriber("/detect_line", Bool, self._detect_area_callback, queue_size=1, tcp_nodelay=True)
        self._sub_img = rospy.Subscriber("/realsense/color/image_raw/compressed", CompressedImage, self._compressed_image_callback, queue_size=1, tcp_nodelay=True)

        self._detect_flag = False
        self._stop_area = 0
        self._input_image = np.empty(0)
        self._compressed_image = CompressedImage()
        self._compressed_image.format = "jpeg"

    def _detect_area_callback(self, data: Bool):
        self._detect_flag = data.data

    def _compressed_image_callback(self, data: CompressedImage):
        self._input_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)
        self._compressed_image.header = data.header

    def _run(self, _) -> None:
        if self._input_image.shape[0] == 0:
            return
        if self._detect_flag == False:
            return

        self._stop_line_flag = False
        trans_img = self._image_trans(self._input_image)
        lines = self._detect_lines(trans_img)
        connected_lines = self._connect_close_lines(lines)

        result_img = self._detect_whiteline(trans_img, connected_lines)

        if self._visualize:
            self._compressed_image.data = cv2.imencode(".jpg", result_img)[1].squeeze().tolist()
            self._pub_image.publish(self._compressed_image)

        self._pub_stop_line_flag.publish(self._stop_line_flag)

    def _image_trans(self, img):
        p1 = np.array(self._trans_upper_left)  # param
        p2 = np.array(self._trans_upper_right)  # param
        p3 = np.array([0, img.shape[0]-1])
        p4 = np.array([img.shape[1]-1, img.shape[0]-1])
        dst_width = math.floor(np.linalg.norm(p2 - p1) * 1.0)
        dst_height = math.floor(np.linalg.norm(p3 - p1))
        trans_src = np.float32([p1, p2, p3, p4])
        trans_dst = np.float32([[0, 0],[dst_width, 0],[0, dst_height],[dst_width, dst_height]])

        trans_mat = cv2.getPerspectiveTransform(trans_src, trans_dst)
        trans_img = cv2.warpPerspective(img, trans_mat, (dst_width, dst_height))

        trans_target_level = (p1[1] + p2[1]) / 2.0
        self._stop_area = math.floor(dst_height * (self._stop_level - trans_target_level) / ((img.shape[0]-1) - trans_target_level))

        return trans_img

    def _get_line_coordinate(self, line):
        return int(line[0]), int(line[1]), int(line[2]), int(line[3]) #Pylsd

    def _calc_endpoints_distance(self, src, dst):
        dists = (
            math.hypot(src[0] - dst[0], src[1] - dst[1]),
            math.hypot(src[0] - dst[2], src[1] - dst[3]),
            math.hypot(src[2] - dst[0], src[3] - dst[1]),
            math.hypot(src[2] - dst[2], src[3] - dst[3]),)

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

        lines = lsd(prep_img) #Pylsd
        lines = lines.tolist() if lines is not None else []
        ##filter
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = self._get_line_coordinate(line)
            line_grad = (y2 - y1) / (x2 - x1 + 1e-10)
            line_length = math.hypot(x2 - x1, y2 - y1)
            if line_length > self._line_length_th and line_grad < self._line_grad_th:  # param
                filtered_lines.append((x1, y1, x2, y2))

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

        return connected_lines

    def _scale_box(self, img, resize):
        h, w = img.shape[:2]
        aspect = w / h

        nw = resize
        nh = round(nw / aspect)

        dst = cv2.resize(img, dsize=(nw, nh))

        return dst

    def _calc_luminance_var(self, img):
        mat = np.array(img)
        flat_mat = mat.flatten()
        grid = math.floor(img.shape[1] / min(self._split_num,img.shape[1])) #param
        var = []

        for v in range(img.shape[0]-1):
          for u in range(0,img.shape[1]-1,grid):
            end = min(u+grid, img.shape[1]-1)
            if (u != end):
              var.append(np.var(mat[v][u:end]))

        return var

    def _crop_rect(self, img, rect):
        result = img.copy()
        center, size, angle = rect
        center = tuple(map(int, center))  # float -> int
        size = tuple(map(int, size))  # float -> int
        h, w = img.shape[:2]  # 画像の高さ、幅

        # 画像を回転する。
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))

        # 切り抜く。
        cropped = cv2.getRectSubPix(rotated, size, center)
        if cropped is not None:
            result = cropped
            if size[1] > size[0]:
                result = result.transpose(1,0,2)[:,::-1]

        return result

    def _calc_rectangularity(self, contour, rect_size):
        contour_area = cv2.contourArea(contour)
        rect_area = rect_size[0] * rect_size[1] + 1e-10

        return contour_area / rect_area

    def _detect_whiteline(self, img, lines):
        ##handpick
        candidate_imgs = []
        candidate_areas = []
        mean_brightnesses = []
        textures_median = []
        smoothness = []
        result_img = img.copy()

        for i, line_a in enumerate(lines):
            for j, line_b in enumerate(lines):
                if j <= i:
                    continue

                contour = np.array(
                        [
                            [max(line_a[0], 0), max(line_a[1], 0)],
                            [min(line_a[2], img.shape[1]-1), min(line_a[3], img.shape[0]-1)],
                            [min(line_b[2], img.shape[1]-1), min(line_b[3], img.shape[0]-1)],
                            [max(line_b[0], 0), max(line_b[1], 0)]
                        ]
                        )
                rect = cv2.minAreaRect(contour)
                center, size, angle = rect
                rect_points = np.array(cv2.boxPoints(rect), dtype='int64')
                aspect = max(size[:]) / min(size[:])

                if self._calc_rectangularity(contour, size) > 0.7 and self._aspect_th <= aspect <= 2*self._resize_num: #param
                    ###debug
                    # cv2.imshow('img', self._crop_rect(img.copy(), rect))
                    # key = cv2.waitKey(5)

                    candidate_img = self._crop_rect(img.copy(), rect)
                    candidate_img = self._scale_box(candidate_img, self._resize_num) #param
                    gray_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
                    flat_img = gray_img.flatten()

                    textures = self._calc_luminance_var(gray_img)
                    textures_median.append(np.median(textures))
                    smoothness.append(sum([n<self._texture_th for n in textures]) / (len(textures)+1e-10))
                    candidate_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2HSV)  #HSV
                    candidate_img = cv2.inRange(candidate_img, tuple(self._hsv_lo), tuple(self._hsv_hi))# param
                    candidate_imgs.append(candidate_img)
                    candidate_areas.append(rect_points)
                    mean_brightness = candidate_img.mean()
                    mean_brightnesses.append(mean_brightness)
        if mean_brightnesses:
            mean_all_brightness = np.array(mean_brightnesses).mean() + 1e-6

        ###result
        for img, area, br, tex, smooth in zip(candidate_imgs, candidate_areas, mean_brightnesses, textures_median, smoothness):
            whiteness = br / mean_all_brightness
            corner_level = max(area[:,1])
            if self._whiteness_th < whiteness and self._smoothness_th < smooth:  # param
                if corner_level > self._stop_area:
                    self._stop_line_flag = True

                if self._visualize:
                    print("###########################################")
                    print(f"shape: {img.shape}")
                    print(f"whiteness: {whiteness}")
                    print(f"textures_median: {tex}")
                    print(f"smoothness: {smooth}\n")

                    bgr = (0,255,0)
                    if corner_level > self._stop_area:
                        bgr = (0,0,255)
                    cv2.polylines(result_img, [area], isClosed=True, color=bgr, thickness=2)

        return result_img

    def __call__(self):
        duration = int(1.0 / self._hz * 1e9)
        rospy.Timer(rospy.Duration(nsecs=duration), self._run)
        rospy.spin()
        cv2.destroyWindow('img')

def main():
    StopLineDetector()()

if __name__ == "__main__":
    main()
