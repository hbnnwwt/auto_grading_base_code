import cv2
import numpy as np


class ImagePreprocessor:
    """答题卡图像预处理模块：方向矫正、去噪、增强、二值化。"""

    MAX_DIMENSION = 8000

    def __init__(self, denoise_method='median', denoise_strength=2,
                 enhance_method='clahe', binarize_method='adaptive',
                 target_size=None):
        self.denoise_method = denoise_method
        self.denoise_strength = denoise_strength
        self.enhance_method = enhance_method
        self.binarize_method = binarize_method
        self.target_size = target_size
        self.last_correction = 0.0
        self.quality_warning = ""

    def load(self, path):
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        h, w = image.shape[:2]
        if max(h, w) > self.MAX_DIMENSION:
            import warnings
            warnings.warn(
                f"图像尺寸 {w}x{h} 超过 {self.MAX_DIMENSION} 像素，"
                f"内存占用约 {h * w * 3 / 1024 / 1024:.0f}MB")
        return image

    def resize(self, image):
        if self.target_size is None:
            return image
        h, w = image.shape[:2]
        tw, th = self.target_size
        if w != tw or h != th:
            image = cv2.resize(image, (tw, th))
        return image

    def _detect_with_contour(self, binary):
        """detect_orientation 的内部实现，同时返回最大轮廓和 minAreaRect。"""
        inv = 255 - binary
        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < binary.size * 0.03:
            return 0.0, None, None

        rect = cv2.minAreaRect(largest)
        (_, (w, h), angle) = rect

        tilt = angle + 90.0

        if w < h:
            bh, bw = binary.shape
            top = np.sum(binary[:bh // 5, :] == 0) / max(bh // 5 * bw, 1)
            bot = np.sum(binary[4 * bh // 5:, :] == 0) / max(
                (bh - 4 * bh // 5) * bw, 1)
            # P30: top 极低时（内容极少），密度比较不可靠，
            # 仅当 bot 有明显内容且远大于 top 时才判定为倒置
            if bot > top * 1.2 and bot > 0.005:
                return 180.0 - tilt, largest, rect
            return -tilt, largest, rect
        else:
            rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
            rh, rw = rotated.shape
            top = np.sum(rotated[:rh // 5, :] == 0) / max(rh // 5 * rw, 1)
            bot = np.sum(rotated[4 * rh // 5:, :] == 0) / max(
                (rh - 4 * rh // 5) * rw, 1)
            if bot > top * 1.2 and bot > 0.005:
                return -90.0 - tilt, largest, rect
            return 90.0 - tilt, largest, rect

    def detect_orientation(self, binary):
        """检测图像需要旋转的角度（支持任意角度，不限于90°倍数）。

        使用轮廓最小外接矩形（minAreaRect）：
        1. w vs h 判断竖向/横向（答题卡是竖向的）
        2. angle 偏离 -90° 的部分就是精确倾斜角
        3. 上下内容密度区分 0° 与 180° 翻转

        启发式问题：
        - 答题卡的外边框在二值图上是什么形状？如何用几何方法描述它的"倾斜"？
        - 如果学生把答题卡旋转了 180 度（上下颠倒），图像的上下两部分在
          像素分布上会有什么不同？如何利用这一特征区分 0° 和 180°？
        """
        raise NotImplementedError("请实现方向检测")

    def draw_detection_viz(self, binary):
        """在二值图上绘制轮廓检测结果的可视化，返回 BGR 图像。"""
        angle, contour, rect = self._detect_with_contour(binary)
        viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if contour is not None:
            cv2.drawContours(viz, [contour], -1, (0, 200, 0), 2)

        if rect is not None:
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(viz, [box], -1, (0, 100, 255), 2)
            # 标注检测角度
            cx = int(rect[0][0])
            cy = int(rect[0][1])
            label = f"{angle:+.1f} deg"
            cv2.putText(viz, label, (cx - 60, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

        return viz

    def correct_orientation(self, image, binary=None):
        """矫正图像方向（支持任意角度旋转）。

        启发式问题：
        - detect_orientation 返回的是"检测到的倾斜角"，矫正旋转时应该向哪个方向旋转？
        - warpAffine 的旋转矩阵如何构建？旋转中心应该选在哪里？
        """
        raise NotImplementedError("请实现方向矫正")

    def denoise(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.denoise_method == 'gaussian':
            k = self.denoise_strength * 2 + 1
            return cv2.GaussianBlur(gray, (k, k), 0)
        elif self.denoise_method == 'median':
            return cv2.medianBlur(gray, self.denoise_strength * 2 + 1)
        elif self.denoise_method == 'bilateral':
            return cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def enhance(self, gray):
        if self.enhance_method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        elif self.enhance_method == 'histeq':
            return cv2.equalizeHist(gray)
        elif self.enhance_method == 'gamma':
            inv = 255.0 / gray.max() if gray.max() > 0 else 1
            norm = (gray * inv).astype(np.uint8)
            return (np.power(norm / 255.0, 0.8) * 255).astype(np.uint8)
        return gray

    def binarize(self, gray):
        if self.binarize_method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.binarize_method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 开运算去除二值化后残留的孤立小噪声点（3x3 核，只影响 1-2px 孤立点）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    @staticmethod
    def extract_vertical_edges(gray, min_height_ratio=0.05, min_area=20):
        """从增强后的灰度图中提取垂直线（用于版面分析），过滤噪声。

        流程：高斯去噪 → Sobel X → 阈值化 → 连通区域高度过滤 → 闭运算加粗

        Args:
            gray: 增强后的灰度图
            min_height_ratio: 最小高度占图像高度的比例，低于此值的连通区域视为噪声
            min_area: 最小连通区域面积

        Returns:
            np.ndarray: 二值图，白色为垂直线

        启发式问题：
        - 垂直线在图像中有什么特征？Sobel 算子的 X 方向梯度能捕捉什么？
        - 边框线很长，文字笔画很短。连通区域分析中，什么几何特征可以区分两者？
        - 提取到的垂直线可能有断裂，什么形态学操作可以连接断开的线段？
        """
        raise NotImplementedError("请实现垂直线提取")

    @staticmethod
    def extract_horizontal_edges(gray, min_width_ratio=0.05, min_area=20):
        """从增强后的灰度图中提取水平线（用于版面分析），过滤噪声。

        流程：高斯去噪 → Sobel Y → 阈值化 → 连通区域宽度过滤 → 闭运算加粗

        Args:
            gray: 增强后的灰度图
            min_width_ratio: 最小宽度占图像宽度的比例，低于此值的连通区域视为噪声
            min_area: 最小连通区域面积

        Returns:
            np.ndarray: 二值图，白色为水平线

        启发式问题：
        - 水平线和垂直线的提取有什么对称关系？哪些参数需要调整？
        """
        raise NotImplementedError("请实现水平线提取")

    def process(self, image):
        """完整预处理流水线：加载 → 方向检测 → 矫正 → 去噪 → 增强 → 二值化。

        启发式问题：
        - 图像处理的典型顺序是什么？为什么去噪要在增强之前？
        - 方向检测和二值化，哪个步骤应该先做？如果先二值化，
          检测结果会更稳定还是更不稳定？
        - 如何检测图像是否过曝或过暗？二值化后的黑白像素比例能给出什么信息？
        """
        raise NotImplementedError("请实现预处理流水线")

    @property
    def before_correction(self):
        """矫正前的原图。"""
        return self._before

    @property
    def detection_viz(self):
        """角度检测可视化（二值图 + 轮廓 + 角度标注）。"""
        return self._detection_viz
