#!/usr/bin/env python3

import itertools
import math
import random

import cv2
import numpy as np
import rospy

##pip install "ocrd-fork-pylsd == 0.0.3" (python3)
from pylsd.lsd import lsd
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse


class StopLineDetector:
    _hz: float
    _trans_upper_left: list
    _trans_upper_right: list
    _stop_level: int
    _line_length_th: int
    _close_line_th: int
    _hsv_lo: list
    _hsv_hi: list
    _split_num: int
    _resize_num: int
    _rectangularity_th: float
    _rect_angle_th: float
    _aspect_lo: float
    _aspect_hi: float
    _hight_lo: float
    _hight_hi: float
    _whiteness_th: float
    _texture_th: float
    _smoothness_th: float
    _pub_image: rospy.Publisher
    _sub_image: rospy.Subscriber
    _boot_flag: bool
    _stop_flag: bool
    _detect_flag: bool
    _stop_area_flag: bool
    _detection_count: int
    _detection_count_th: int
    _stop_area: int
    _input_image: np.ndarray
    _pub_image_msg: CompressedImage
    _visualize: bool

    def __init__(self):
        rospy.init_node("stop_line_detector", anonymous=True)

        self._hz = rospy.get_param("~hz", 10.0)
        self._trans_upper_left = [
            int(n) for n in rospy.get_param("~trans_upper_left", [0, 0])
        ]
        self._trans_upper_right = [
            int(n) for n in rospy.get_param("~trans_upper_right", [680, 0])
        ]
        self._stop_level = rospy.get_param("~stop_level", 0)
        self._line_length_th = rospy.get_param("~line_length_th", 10)
        self._close_line_th = rospy.get_param("~close_line_th", 7)
        self._parallelism_th = rospy.get_param("~parallelism_th", 0.80)
        self._hsv_lo = [int(n) for n in rospy.get_param("~hsv_lo", [0, 0, 0])]
        self._hsv_hi = [
            int(n) for n in rospy.get_param("~hsv_hi", [180, 255, 255])
        ]
        self._split_num = rospy.get_param("~split_num", 10)
        self._resize_num = rospy.get_param("~resize_num", 30)
        self._rectangularity_th = rospy.get_param("~rectangularity_th", 0.6)
        self._rect_angle_th = rospy.get_param("~rect_angle_th", 0.6)
        self._aspect_lo = rospy.get_param("~aspect_lo", 2.0)
        self._aspect_hi = rospy.get_param("~aspect_hi", 20.0)
        self._height_lo = rospy.get_param("~height_lo", 5.0)
        self._height_hi = rospy.get_param("~height_hi", 20.0)
        self._whiteness_th = rospy.get_param("~whiteness_th", 0.5)
        self._texture_th = rospy.get_param("~texture_th", 10)
        self._smoothness_th = rospy.get_param("~smoothness_th", 0.5)
        self._detection_count_th = rospy.get_param("~detection_count_th", 5)
        self._visualize = rospy.get_param("~visualize", False)

        self._pub_image = rospy.Publisher(
            "~stop_line_image/compressed",
            CompressedImage,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._sub_image = rospy.Subscriber(
            "/camera/image_raw/compressed",
            CompressedImage,
            self._compressed_image_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._request_server = rospy.Service(
            "~request", SetBool, self._request_callback
        )
        self._task_stop_client = rospy.ServiceProxy("/task/stop", SetBool)

        self._boot_flag = False
        self._detection_count = 0
        self._stop_area = 0
        self._input_image = np.empty(0)
        self._pub_image_msg = CompressedImage()
        self._pub_image_msg.format = "jpeg"

    def _compressed_image_callback(self, data: CompressedImage):
        self._input_image = cv2.imdecode(
            np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR
        )
        self._pub_image_msg.header = data.header

    def _request_callback(self, req: SetBool):
        self._boot_flag = req.data
        res: SetBoolResponse = SetBoolResponse(success=True)
        if self._boot_flag:
            res.message = "Stop line detection started."
        else:
            res.message = "Stop line detection stopped."
        return res

    def _run(self, _) -> None:
        if self._input_image.shape[0] == 0:
            return
        if self._boot_flag == False:
            if self._visualize:
                self._pub_image_msg.data = (
                    cv2.imencode(".jpg", self._input_image)[1]
                    .squeeze()
                    .tolist()
                )
                self._pub_image.publish(self._pub_image_msg)
            return

        self._stop_flag = False
        self._detect_flag = False
        self._stop_area_flag = False

        trans_img = self._image_trans(self._input_image)
        detected_lines = self._detect_lines(trans_img)
        result_img = self._detect_whiteline(trans_img, detected_lines)

        if self._detect_flag:
            self._detection_count += 1
        else:
            self._detection_count = 0

        if self._visualize:
            self._pub_image_msg.data = (
                cv2.imencode(".jpg", result_img)[1].squeeze().tolist()
            )
            self._pub_image.publish(self._pub_image_msg)

        if (
            self._stop_area_flag
            and self._detection_count >= self._detection_count_th
        ):
            self._stop_flag = True

        if self._stop_flag:
            resp: SetBoolResponse = self._task_stop_client(True)
            rospy.logwarn(resp.message)
            self._boot_flag = False

    def _image_trans(self, img):
        p1 = np.array(self._trans_upper_left)  # param
        p2 = np.array(self._trans_upper_right)  # param
        p3 = np.array([0, img.shape[0] - 1])
        p4 = np.array([img.shape[1] - 1, img.shape[0] - 1])
        dst_width = math.floor(np.linalg.norm(p2 - p1) * 1.0)
        dst_height = math.floor(np.linalg.norm(p3 - p1))
        trans_src = np.float32([p1, p2, p3, p4])
        trans_dst = np.float32(
            [[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]]
        )

        trans_mat = cv2.getPerspectiveTransform(trans_src, trans_dst)
        trans_img = cv2.warpPerspective(
            img, trans_mat, (dst_width, dst_height)
        )

        trans_target_level = (p1[1] + p2[1]) / 2.0
        self._stop_area = math.floor(
            dst_height
            * (self._stop_level - trans_target_level)
            / ((img.shape[0] - 1) - trans_target_level)
        )

        return trans_img

    def _get_line_coordinate(self, line):
        return int(line[0]), int(line[1]), int(line[2]), int(line[3])  # Pylsd

    def _calc_endpoints_distance(self, src, dst):
        dists = (
            math.hypot(src[0] - dst[0], src[1] - dst[1]),
            math.hypot(src[0] - dst[2], src[1] - dst[3]),
            math.hypot(src[2] - dst[0], src[3] - dst[1]),
            math.hypot(src[2] - dst[2], src[3] - dst[3]),
        )

        return min(dists)

    def _calc_parallelism(self, src, dst):
        vec1 = [src[2] - src[0], src[3] - src[1]]
        vec2 = [dst[2] - dst[0], dst[3] - dst[1]]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return abs(np.dot(vec1, vec2) / (norm1 * norm2))

    def _search_close_lines(self, src_lines, dst_lines):
        close_lines = []
        for dst in dst_lines:
            if dst in src_lines:
                continue
            dists = []
            for src in src_lines:
                dists.append(self._calc_endpoints_distance(src, dst))
            if min(dists) < self._close_line_th:  # param
                parallelism = self._calc_parallelism(src, dst)
                if parallelism > self._parallelism_th:
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl_img = clahe.apply(gray_img)
        prep_img = cv2.GaussianBlur(cl_img, (5, 5), 0)

        lines = lsd(prep_img, scale=0.6)  # Pylsd
        lines = lines.tolist() if lines is not None else []
        connected_lines = self._connect_close_lines(lines)

        ##filter
        filtered_lines = []
        for line in connected_lines:
            x1, y1, x2, y2 = self._get_line_coordinate(line)
            line_length = math.hypot(x2 - x1, y2 - y1)
            if line_length > self._line_length_th:  # param
                filtered_lines.append((x1, y1, x2, y2))

        ############### debug ###############
        # lines_img = img.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = self._get_line_coordinate(line)
        #     color = [random.randint(0, 255) for i in range(3)]
        #     detected_img = cv2.line(lines_img, (x1, y1), (x2, y2), color, 5)
        # cv2.imshow('lines', lines_img)
        # key = cv2.waitKey(5)

        # filtered_lines_img = img.copy()
        # for line in filtered_lines:
        #     x1, y1, x2, y2 = self._get_line_coordinate(line)
        #     color = [random.randint(0, 255) for i in range(3)]
        #     detected_img = cv2.line(filtered_lines_img, (x1, y1), (x2, y2), color, 5)
        # cv2.imshow('filtered_lines', filtered_lines_img)
        # key = cv2.waitKey(5)

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
        nh = max(round(nw / aspect), 1)

        dst = cv2.resize(img, dsize=(nw, nh))

        return dst

    def _calc_luminance_var(self, img):
        mat = np.array(img)
        flat_mat = mat.flatten()
        grid = math.floor(
            img.shape[1] / min(self._split_num, img.shape[1])
        )  # param
        var = []

        for v in range(img.shape[0] - 1):
            for u in range(0, img.shape[1] - 1, grid):
                end = min(u + grid, img.shape[1] - 1)
                if u != end:
                    var.append(np.var(mat[v][u:end]))

        return var

    def _crop_rect(self, img, rect):
        result = img.copy()
        center, size, angle = rect
        center = tuple(map(int, center))
        size = tuple(map(int, size))
        h, w = img.shape[:2]

        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))

        cropped = cv2.getRectSubPix(rotated, size, center)
        if cropped is not None:
            result = cropped
            if size[1] > size[0]:
                result = result.transpose(1, 0, 2)[:, ::-1]

        return result

    def _calc_rect_angle(self, area):
        vec1 = area[0] - area[1]
        vec2 = area[1] - area[2]

        if np.linalg.norm(vec1) < np.linalg.norm(vec2):
            long_side = vec2
        else:
            long_side = vec1

        angle = np.rad2deg(-np.arctan(long_side[1] / (long_side[0] + 1e-10)))

        return np.abs(angle)

    def _calc_rectangularity(self, contour, rect_size):
        contour_area = cv2.contourArea(contour)
        rect_area = rect_size[0] * rect_size[1] + 1e-10

        return contour_area / rect_area

    def _detect_whiteline(self, img, lines):
        ##handpick
        candidate_imgs = []
        candidate_areas = []
        textures_median = []
        smoothness = []
        white_ratio = []
        result_img = img.copy()

        for i, line_a in enumerate(lines):
            for j, line_b in enumerate(lines):
                if j <= i:
                    continue

                contour = np.array(
                    [
                        [max(line_a[0], 0), max(line_a[1], 0)],
                        [
                            min(line_a[2], img.shape[1] - 1),
                            min(line_a[3], img.shape[0] - 1),
                        ],
                        [
                            min(line_b[2], img.shape[1] - 1),
                            min(line_b[3], img.shape[0] - 1),
                        ],
                        [max(line_b[0], 0), max(line_b[1], 0)],
                    ]
                )
                rect = cv2.minAreaRect(contour)
                center, size, _ = rect
                rect_points = np.array(cv2.boxPoints(rect), dtype="int64")
                angle = self._calc_rect_angle(rect_points)
                aspect = max(size[0], size[1]) / (
                    min(size[0], size[1]) + 1e-10
                )

                ########## Debug ###########
                # rospy.logwarn("----------------------------------------------------")
                # rospy.logwarn(f"rectangularity: {self._calc_rectangularity(contour, size)}")
                # rospy.logwarn(f"angle: {angle}")
                # rospy.logwarn(f"aspect: {aspect}")

                if (
                    self._rectangularity_th
                    < self._calc_rectangularity(contour, size)
                    and angle <= self._rect_angle_th
                    and self._aspect_lo <= aspect <= self._aspect_hi
                    and self._height_lo
                    <= min(size[0], size[1])
                    <= self._height_hi
                ):

                    candidate_img = self._crop_rect(img.copy(), rect)
                    candidate_img = self._scale_box(
                        candidate_img, self._resize_num
                    )  # param
                    gray_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
                    flat_img = gray_img.flatten()

                    textures = self._calc_luminance_var(gray_img)
                    textures_median.append(np.median(textures))
                    smoothness.append(
                        sum([tex < self._texture_th for tex in textures])
                        / (len(textures) + 1e-10)
                    )
                    candidate_img = cv2.cvtColor(
                        candidate_img, cv2.COLOR_BGR2HSV
                    )  # HSV
                    candidate_img = cv2.inRange(
                        candidate_img, tuple(self._hsv_lo), tuple(self._hsv_hi)
                    )  # param
                    whole_area = candidate_img.size
                    white_ratio.append(
                        cv2.countNonZero(candidate_img) / (whole_area + 1e-10)
                    )

                    ############### Debug ###############
                    # rospy.logwarn("===========================================")
                    # rospy.logwarn(f"white_ratio: {white_ratio}")
                    # rospy.logwarn(f"smooth: {smoothness}")

                    candidate_imgs.append(candidate_img)
                    candidate_areas.append(rect_points)

        ###result
        for img, area, white, tex, smooth in zip(
            candidate_imgs,
            candidate_areas,
            white_ratio,
            textures_median,
            smoothness,
        ):
            corner_level = max(area[:, 1])

            # print("############### DEBUG ###############")
            # print(f"shape: {img.shape}")
            # print(f"whiteness: {white}")
            # print(f"textures_median: {tex}")
            # print(f"smoothness: {smooth}")
            # print(f"hight: {min([cv2.norm(area[0]-area[1]),cv2.norm(area[0]-area[2]),cv2.norm(area[0]-area[3])])}\n")
            # debug_img = result_img.copy()
            # cv2.polylines(debug_img, [area], isClosed=True, color=[255,0,128], thickness=2)
            # cv2.imshow('debug', debug_img)
            # key = cv2.waitKey(5)

            if (
                self._whiteness_th < white and self._smoothness_th < smooth
            ):  # param
                self._detect_flag = True
                if corner_level > self._stop_area:
                    self._stop_area_flag = True

                if self._visualize:
                    if corner_level > self._stop_area:
                        bgr = (0, 0, 255)
                        rospy.logerr(
                            "########## DETECTED IN STOP AREA ##########"
                        )
                        rospy.logerr(
                            f"detection count: {self._detection_count}"
                        )
                        rospy.logerr(f"shape: {img.shape}")
                        rospy.logerr(f"whiteness: {white}")
                        rospy.logerr(f"textures_median: {tex}")
                        rospy.logerr(f"smoothness: {smooth}\n")
                        rospy.logerr(
                            f"hight: {min([cv2.norm(area[0]-area[1]),cv2.norm(area[0]-area[2]),cv2.norm(area[0]-area[3])])}\n"
                        )
                    else:
                        bgr = (0, 255, 0)
                        log_rate = 0.5
                        rospy.logwarn_throttle(
                            log_rate, "##### DETECTED #####"
                        )
                        rospy.logwarn_throttle(
                            log_rate,
                            f"detection count: {self._detection_count}",
                        )
                        rospy.logwarn_throttle(log_rate, f"shape: {img.shape}")
                        rospy.logwarn_throttle(log_rate, f"whiteness: {white}")
                        rospy.logwarn_throttle(
                            log_rate, f"textures_median: {tex}"
                        )
                        rospy.logwarn_throttle(
                            log_rate, f"smoothness: {smooth}\n"
                        )
                        rospy.logwarn_throttle(
                            log_rate,
                            f"hight: {min([cv2.norm(area[0]-area[1]),cv2.norm(area[0]-area[2]),cv2.norm(area[0]-area[3])])}\n",
                        )

                    cv2.polylines(
                        result_img,
                        [area],
                        isClosed=True,
                        color=bgr,
                        thickness=2,
                    )

        return result_img

    def __call__(self):
        duration = int(1.0 / self._hz * 1e9)
        rospy.Timer(rospy.Duration(nsecs=duration), self._run)
        rospy.spin()
        cv2.destroyWindow("img")


def main():
    StopLineDetector()()


if __name__ == "__main__":
    main()
