import os
import sys
import cv2
import numpy as np
from nibabel import Nifti1Image, save
from skimage import util, exposure
from skimage.segmentation import random_walker
from pydicom import read_file
import traceback
from copy import deepcopy
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt
from display import *
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
from mayavi import mlab

class MyClass(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.showImage.wheelEvent = self.on_wheel
        self.result_image.wheelEvent = self.on_wheel
        self.count = 1  # 用于保存文件，防止覆盖
        self.old_hook = sys.excepthook
        sys.excepthook = self.catch_exceptions
        self.is_dicom = []
        self.current = 0  # 用来翻页图片
        self.file_list = []  # 存储图片的路径
        self.original_image_tuple = ()  # 存储已经转化为RGB格式的图片,用元组类型目的是无法改动
        self.processed_image_list = []  # 默认为1张图片
        self.transit_list = []  # 存储中间值
        self.mask_list = []  # 存储掩膜

    def catch_exceptions(self, ty, value, trace_back):
        """
        https://blog.csdn.net/venture5/article/details/121422886
        :param ty: 异常的类型
        :param value: 异常的对象
        :param traceback: 异常的traceback
        """
        traceback_format = traceback.format_exception(ty, value, trace_back)
        traceback_string = "".join(traceback_format)
        self.error_label.setText(traceback_string)
        self.old_hook(ty, value, trace_back)

    def on_wheel(self, event):
        """用于使鼠标在图片上进行滚轮可以翻页"""
        if event.angleDelta().y() > 0:
            self.show_prev()
        else:
            self.show_next()

    def show_next(self):
        self.current += 1
        if self.current >= len(self.file_list):
            self.current = 0
        self.groupBox_3.setTitle(f'原图像{self.current + 1}/{len(self.file_list)}')
        self.show_image()
        if self.processed_image_list[self.current] is not None:
            self.show_processed_image()
        else:
            self.result_image.clear()
        if self.is_dicom[self.current]:
            self.read_dicom_info(self.file_list[self.current])
        else:
            self.clear_dicom_info()

    def show_prev(self):
        self.current -= 1
        if self.current < 0:
            self.current = len(self.file_list) - 1
        self.groupBox_3.setTitle(f'原图像{self.current + 1}/{len(self.file_list)}')
        self.show_image()
        if self.processed_image_list[self.current] is not None:
            self.show_processed_image()
        else:
            self.result_image.clear()
        if self.is_dicom[self.current]:
            self.read_dicom_info(self.file_list[self.current])
        else:
            self.clear_dicom_info()

    def show_image(self):
        img = self.original_image_tuple[self.current]
        _image = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
        jpg_out = jpg_out.scaled(self.showImage.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.showImage.setPixmap(jpg_out)  # 设置图片显示

    def show_processed_image(self):
        img = self.processed_image_list[self.current]
        _image = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
        jpg_out = jpg_out.scaled(self.showImage.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def clear(self):
        """清除图片操作"""
        self.result_image.clear()
        self.processed_image_list[self.current] = None

    def load_images(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
        try:
            for file in os.listdir(folder):
                if file.endswith(".dcm"):
                    self.file_list.append(os.path.join(folder, file))
            for i in self.file_list:
                dcm = read_file(i)
                img_arr = dcm.pixel_array
                # 获取像素点个数
                lens = img_arr.shape[0] * img_arr.shape[1]
                # 获取像素点的最大值和最小值
                arr_temp = np.reshape(img_arr, (lens,))
                max_val = max(arr_temp)
                min_val = min(arr_temp)
                # 图像归一化
                img_arr = (img_arr - min_val) / (max_val - min_val)
                img_arr = img_arr * 255
                img_arr = img_arr.astype(np.uint8)  # python类型转换
                img_o = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                self.original_image_tuple = self.original_image_tuple + (img_o,)
            self.show_image()
            self.read_dicom_info(self.file_list[self.current])
            self.processed_image_list = [None] * len(self.file_list)
            self.is_dicom = [True] * len(self.file_list)
            self.transit_list = list(self.original_image_tuple)
            self.groupBox_3.setTitle(f'原图像1/{len(self.file_list)}')
        except FileNotFoundError:
            pass

    def clear_dicom_info(self):
        self.patient_id_label.clear()
        self.patient_birth_date_label.clear()
        self.patient_sex_label.clear()
        self.patient_age_label.clear()
        self.study_date_label.clear()
        self.study_time_label.clear()
        self.study_id_label.clear()
        self.study_modality_label.clear()
        self.study_discription_label.clear()
        self.series_date_label.clear()
        self.series_time_label.clear()
        self.series_discription_label.clear()
        self.institution_name_label.clear()
        self.manufacturer_label.clear()

    def read_dicom_info(self, dcm_file):
        """如果dicom文件没有标注数据，则会报错，因此需要所有都用try except"""
        dcm = read_file(dcm_file)
        try:
            self.patient_id_label.setText(str(dcm.PatientID))
        except AttributeError:
            pass
        try:
            self.patient_birth_date_label.setText(str(dcm.PatientBirthDate))
        except AttributeError:
            pass
        try:
            self.patient_sex_label.setText(str(dcm.PatientSex))
        except AttributeError:
            pass
        try:
            self.patient_age_label.setText(str(dcm.PatientAge))
        except AttributeError:
            pass
        try:
            self.study_date_label.setText(str(dcm.StudyDate))
        except AttributeError:
            pass
        try:
            self.study_time_label.setText(str(dcm.StudyTime))
        except AttributeError:
            pass
        try:
            self.study_id_label.setText(str(dcm.StudyID))
        except AttributeError:
            pass
        try:
            self.study_modality_label.setText(str(dcm[0x0008, 0x0060].value))
        except AttributeError:
            pass
        try:
            self.study_discription_label.setText(str(dcm[0x0008, 0x01030].value))
        except AttributeError:
            pass
        try:
            self.series_date_label.setText(str(dcm.SeriesDate))
        except AttributeError:
            pass
        try:
            self.series_time_label.setText(str(dcm.SeriesTime))
        except AttributeError:
            pass
        try:
            self.series_discription_label.setText(str(dcm[0x0008, 0x103E].value))
        except AttributeError:
            pass
        try:
            self.institution_name_label.setText(str(dcm.InstitutionName))
        except AttributeError:
            pass
        try:
            self.manufacturer_label.setText(str(dcm.Manufacturer))
        except AttributeError:
            pass
        return dcm.pixel_array

    def open_image(self):  # 打开文件
        global fname

        # 定义文件读取函数，解决中文路径读取错误的问题
        def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            return cv_img

        fname, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*;;*.png;;All Files(*)")
        if fname:
            self.file_list.append(fname)
        try:
            if fname.endswith('dcm'):
                self.current = len(self.original_image_tuple) + 1
                # 获取像素矩阵
                img_arr = self.read_dicom_info(fname)
                # 获取像素点个数
                lens = img_arr.shape[0] * img_arr.shape[1]
                # 获取像素点的最大值和最小值
                arr_temp = np.reshape(img_arr, (lens,))
                max_val = max(arr_temp)
                min_val = min(arr_temp)
                # 图像归一化
                img_arr = (img_arr - min_val) / (max_val - min_val)
                img_arr = img_arr * 255
                img_arr = img_arr.astype(np.uint8)  # python类型转换
                img_o = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # numpy 转 image类
                self.original_image_tuple = self.original_image_tuple + (img_o,)
                _image = QtGui.QImage(img_o[:], img_o.shape[1], img_o.shape[0], img_o.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
                jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
                jpg_out = jpg_out.scaled(self.showImage.size(), aspectRatioMode=Qt.KeepAspectRatio)
                self.showImage.setPixmap(jpg_out)  # 设置图片显示
                self.processed_image_list.append(None)
                self.is_dicom.append(True)
            else:
                self.current = len(self.original_image_tuple) + 1
                img = cv_imread(fname)  # opencv读取图片
                img_o = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
                self.original_image_tuple = self.original_image_tuple + (img_o,)
                _image = QtGui.QImage(img_o[:], img_o.shape[1], img_o.shape[0], img_o.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
                jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
                jpg_out = jpg_out.scaled(self.showImage.size(), aspectRatioMode=Qt.KeepAspectRatio)
                self.showImage.setPixmap(jpg_out)  # 设置图片显示
                self.processed_image_list.append(None)
                self.is_dicom.append(False)
            self.transit_list = list(self.original_image_tuple)
            self.groupBox_3.setTitle(f'原图像{self.current}/{len(self.file_list)}')
        except FileNotFoundError:
            pass

    def save_file(self):
        cv2.imwrite(f'result{self.count}.png', self.processed_image_list[self.current])
        self.count += 1

    def canny(self):
        self.error_label.setText("请按下回车完成边缘检测")

        def CannyThreshold(lowThreshold):
            detected_edges = cv2.Canny(gray,
                                       lowThreshold,
                                       lowThreshold * ratio,
                                       apertureSize=kernel_size)
            dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
            cv2.imshow('canny', dst)
            im = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            self.processed_image_list[self.current] = im
            _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
            jpg_out = QtGui.QPixmap(_image)
            jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
            self.result_image.setPixmap(jpg_out)

        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 3
        kernel_size = 3

        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('canny', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('canny', 200, 200)

        cv2.createTrackbar('Min threshold', 'canny', lowThreshold, max_lowThreshold, CannyThreshold)

        CannyThreshold(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    """滤波处理"""

    def median_filter(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kSize = int(self.median_filter_lineEdit.text())
        img2 = cv2.medianBlur(img, kSize)
        im = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im
        _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def bilateral_filter(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ksize = int(self.bilateral_filter_ksize.text())
        sigmacolor = int(self.bilateral_filter_sigmacolor.text())
        sigmaspace = int(self.bilateral_filter_sigmaspace.text())
        img2 = cv2.bilateralFilter(img, ksize, sigmacolor, sigmaspace)
        im = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im
        _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def gaussian_filter(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k1 = int(self.gaussian_filter_width.text())
        k2 = int(self.gaussian_filter_height.text())
        ksize = (k1, k2)
        sigmaX = int(self.gaussian_filter_sigmaX.text())
        sigmaY = self.gaussian_filter_sigmaY.text()
        if len(sigmaY) == 0 or int(sigmaY) == 0:
            img2 = cv2.GaussianBlur(img, ksize, sigmaX)
        else:
            img2 = cv2.GaussianBlur(img, ksize, sigmaX, int(sigmaY))
        im = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im
        _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    """评估函数"""

    def psnr(self):
        im1 = self.processed_image_list[self.current]
        im2 = self.original_image_tuple[self.current]

        def cal_psnr(im1, im2):
            """https://zhuanlan.zhihu.com/p/150865007"""
            mse = (np.abs(im1 - im2) ** 2).mean()
            psnr = 10 * np.log10(255 * 255 / mse)
            return psnr

        value = cal_psnr(im1, im2)
        self.psnr_label.setText(str(value))

    def ssim(self):
        im1 = self.processed_image_list[self.current]
        im2 = self.original_image_tuple[self.current]

        def cal_ssim(im1, im2):
            # 计算ssim值
            # assert len(im1.shape) == 2 and len(im2.shape) == 2
            # assert im1.shape == im2.shape
            mu1 = im1.mean()
            mu2 = im2.mean()
            sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
            sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
            sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
            k1, k2, L = 0.01, 0.03, 255
            C1 = (k1 * L) ** 2
            C2 = (k2 * L) ** 2
            C3 = C2 / 2
            l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
            c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
            s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
            ssim = l12 * c12 * s12
            return ssim

        value = cal_ssim(im1, im2)
        self.ssim_label.setText(str(value))

    def gaussian_noise(self):
        """加噪从原图上加噪，加噪后的待处理图片保存在transit列表中，方便滤波和计算"""
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(self.gaussian_noise_mean.text())
        var = float(self.gaussian_noise_var.text())
        noise_img = util.random_noise(img, mode='gaussian', mean=mean, var=var)
        noise_img = noise_img * 255
        noise_img = noise_img.astype(np.uint8)
        im = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im
        _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def pepper_and_salt_noise(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        amount = float(self.pepper_and_salt_amount.text())
        noise_img = util.random_noise(img, mode='s&p', amount=amount)
        noise_img = noise_img * 255
        noise_img = noise_img.astype(np.uint8)
        im = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im
        _image = QtGui.QImage(im[:], im.shape[1], im.shape[0], im.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def draw(self):
        self.error_label.setText("长按勾画感兴趣区域，按回车完成勾画")
        # 定义变量存储所有点的坐标
        self.pts = []

        def draw_point(event, x, y, flags, param):
            if event == 0 and flags == 1:  # 发现鼠标移动且左键按下
                self.pts.append((x, y))
                cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
                if len(self.pts) >= 2:
                    cv2.line(img_copy, self.pts[-2], self.pts[-1], (0, 0, 255), 2)

        # 创建窗口并绑定鼠标事件
        cv2.namedWindow('label (press Esc to quit and press Enter to finish)', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('label (press Esc to quit and press Enter to finish)', 200, 200)
        cv2.setMouseCallback('label (press Esc to quit and press Enter to finish)', draw_point)

        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img_copy = deepcopy(img)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        while True:
            cv2.imshow('label (press Esc to quit and press Enter to finish)', img_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc键的键值为27
                self.pts = []
                break
            elif key == 13:  # 回车键的键值为13
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                pts = np.array([self.pts], dtype=np.int32)
                cv2.fillPoly(mask, pts, (1, 1, 1))
                img_copy = cv2.bitwise_and(img_copy, img_copy, mask=mask)
                self.pts = []
                break

        cv2.destroyAllWindows()

        num = self.find_number_in_list(self.mask_list, self.current)
        if num == -1:  # 说明这个图还没有勾画过
            self.mask_list.append((mask, self.current))  # 加入列表的是元组，附带一个编号，用于判断是否已经勾画过
        else:  # 这张图已经被勾画过，重画了一遍，覆盖之前的mask
            self.mask_list[num] = (mask, self.current)
        im = self.overlap(mask)
        # im = np.where(mask == 1, 255, mask)
        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im2
        _image = QtGui.QImage(im2[:], im2.shape[1], im2.shape[0], im2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def find_number_in_list(self, lst, number):
        for i, tup in enumerate(lst):
            if tup[1] == number:
                return i
        return -1

    def save_as_GT(self):
        self.GT_img = self.processed_image_list[self.current]
        self.GT_detect_label.setText(' ')

    def upload_GT(self):
        def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            return cv_img

        fname, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*;;*.png;;All Files(*)")
        self.GT_img = cv_imread(fname)  # opencv读取图片
        # self.GT_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.GT_detect_label.setText(' ')

    def dice(self):
        def dice_coef(y_true, y_pred, smooth=1):
            y_true = np.where(y_true > 0, 1, 0)
            y_pred = np.where(y_pred > 0, 1, 0)
            intersection = np.logical_and(y_true, y_pred)
            return 2. * intersection.sum() / (y_true.sum() + y_pred.sum())

        self.dice_label.setText(str(dice_coef(self.GT_img, self.processed_image_list[self.current])))

    def three_dimensional_overlap(self):
        # 三维重建
        # from mayavi import mlab
        lst1 = [tup[0] for tup in self.mask_list]
        lst1[0] = cv2.cvtColor(lst1[0], cv2.COLOR_GRAY2BGR)
        h, w, _ = lst1[0].shape
        self.data = np.zeros((h, w, len(lst1)))
        for i, img in enumerate(lst1):
            img = np.where(img == 1, 255, img)
            if img.ndim == 2:
                self.data[:, :, i] = img
            elif img.ndim == 3:
                self.data[:, :, i] = img[:, :, 0]
            else:
                raise ValueError("Image must be 2-dimensional or 3-dimensional")
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        X, Y, Z = np.mgrid[:h, :w, :len(lst1)]
        s = self.data.flatten() / 255.0
        # 创建实心点阵
        mlab.points3d(X.flatten(), Y.flatten(), Z.flatten(), s, color=(1, 0, 0), mode='cube', scale_factor=1)
        mlab.show()


    # def three_dimensional_overlap(self):
    #     lst1 = [tup[0] for tup in self.mask_list]
    #     lst1[0] = cv2.cvtColor(lst1[0], cv2.COLOR_GRAY2BGR)
    #     h, w, _ = lst1[0].shape
    #     self.data = np.zeros((h, w, len(lst1)))
    #     for i, img in enumerate(lst1):
    #         img = np.where(img == 1, 255, img)
    #         if img.ndim == 2:
    #             self.data[:, :, i] = img
    #         elif img.ndim == 3:
    #             self.data[:, :, i] = img[:, :, 0]
    #         else:
    #             raise ValueError("Image must be 2-dimensional or 3-dimensional")
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     X, Y, Z = np.meshgrid(np.arange(h), np.arange(w), np.arange(len(lst1)))
    #     s = self.data.flatten() / 255.0
    #     # 创建实心点阵
    #     ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), s=s, c='r', marker='o')
    #     plt.savefig('myplot.png')
    #     img = cv2.imread('myplot.png')
    #     _image = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
    #                           QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
    #     jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
    #     jpg_out = jpg_out.scaled(self.showImage.size(), aspectRatioMode=Qt.KeepAspectRatio)
    #     self.result_image.setPixmap(jpg_out)

    def save_nii(self):
        self.error_label.setText(".nii文件保存成功")
        data = self.data.astype(np.int16)
        img = Nifti1Image(data, np.eye(4))
        save(img, '3d_mask.nii')

    def grab_cut(self):
        self.error_label.setText("长按左键画出矩形框，按下回车完成勾画")
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.namedWindow('ROI selector', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('ROI selector', 200, 200)
        rect = cv2.selectROI(img)

        x, y, w, h = rect
        mask[y:y + h, x:x + w] = cv2.GC_PR_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50, cv2.GC_INIT_WITH_MASK)
        cv2.destroyAllWindows()
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        bg_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 1, 0).astype('uint8')

        fg = cv2.bitwise_and(img, img, mask=mask)
        bg = cv2.bitwise_and(img, img, mask=bg_mask)
        cv2.namedWindow('Background', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('Background', 200, 200)
        cv2.imshow('Background', bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        num = self.find_number_in_list(self.mask_list, self.current)
        if num == -1:  # 说明这个图还没有勾画过
            self.mask_list.append((mask, self.current))  # 加入列表的是元组，附带一个编号，用于判断是否已经勾画过
        else:  # 这张图已经被勾画过，重画了一遍，覆盖之前的mask
            self.mask_list[num] = (mask, self.current)

        im = self.overlap(mask)
        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im2
        _image = QtGui.QImage(im2[:], im2.shape[1], im2.shape[0], im2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def live_wire(self):
        pass

    def random_walk(self):
        self.error_label.setText("左键添加前景点，右键添加背景点，按下回车完成勾画")

        # 定义一个回调函数，用于处理鼠标事件
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 鼠标左键按下时，选取前景种子点
                cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)
                markers[y, x] = 1
                print(f"Frontground marker selected at ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                # 鼠标右键按下时，选取背景种子点
                cv2.circle(image_copy, (x, y), 2, (255, 0, 0), -1)
                markers[y, x] = 2
                print(f"Background marker selected at ({x}, {y})")

        # 读入一张灰度图像
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_copy = deepcopy(img)

        # 创建前景和背景种子点
        markers = np.zeros(img.shape, dtype=np.int32)

        # 显示图像，并设置鼠标事件回调函数
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('image', 200, 200)
        cv2.setMouseCallback("image", mouse_callback)

        while True:
            cv2.imshow("image", image_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                labels = random_walker(img, markers)
                mask = (labels == 1)
                bg_mask = np.uint8(1 - mask)
                bg_mask = cv2.resize(bg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                bg = cv2.bitwise_and(img, img, mask=bg_mask)
                cv2.destroyAllWindows()
                cv2.namedWindow("Background", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.moveWindow('Background', 200, 200)
                cv2.imshow('Background', bg)
                mask = mask.astype('uint8')
                cv2.waitKey(0)
                break
            if key == 27:
                break
        cv2.destroyAllWindows()
        num = self.find_number_in_list(self.mask_list, self.current)
        if num == -1:  # 说明这个图还没有勾画过
            self.mask_list.append((mask, self.current))  # 加入列表的是元组，附带一个编号，用于判断是否已经勾画过
        else:  # 这张图已经被勾画过，重画了一遍，覆盖之前的mask
            self.mask_list[num] = (mask, self.current)

        im = self.overlap(mask)
        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.processed_image_list[self.current] = im2
        _image = QtGui.QImage(im2[:], im2.shape[1], im2.shape[0], im2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def adaptive_histogram_equalization(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 创建CLAHE对象并进行自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_img = clahe.apply(img)
        eq_img_rgb = cv2.cvtColor(eq_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB format
        self.processed_image_list[self.current] = eq_img_rgb
        _image = QtGui.QImage(eq_img_rgb.data, eq_img_rgb.shape[1], eq_img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def histogram_equalization(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eq_img = cv2.equalizeHist(img)
        eq_img_rgb = cv2.cvtColor(eq_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB format
        self.processed_image_list[self.current] = eq_img_rgb
        _image = QtGui.QImage(eq_img_rgb.data, eq_img_rgb.shape[1], eq_img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def sharp(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 创建卷积核
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        # 对图像进行锐化
        img_sharp = cv2.filter2D(img, -1, kernel)
        img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB format
        self.processed_image_list[self.current] = img_sharp
        _image = QtGui.QImage(img_sharp.data, img_sharp.shape[1], img_sharp.shape[0], QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def stretch(self):
        if self.processed_image_list[self.current] is not None:
            img = self.processed_image_list[self.current]
        else:
            img = self.transit_list[self.current]
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # skimage中的对比度拉伸
        p2, p98 = np.percentile(img, (2, 98))  # 获取2%和98%的像素值
        img_stretched = exposure.rescale_intensity(img, in_range=(p2, p98))
        img_stretched = cv2.cvtColor(img_stretched, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB format
        self.processed_image_list[self.current] = img_stretched
        _image = QtGui.QImage(img_stretched.data, img_stretched.shape[1], img_stretched.shape[0],
                              QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(_image)
        jpg_out = jpg_out.scaled(self.result_image.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.result_image.setPixmap(jpg_out)

    def overlap(self,mask):
        img = self.original_image_tuple[self.current]
        # 将mask转换成图像数据
        mask = mask.astype(np.uint8)
        # 创建一个三通道的全黑掩膜
        overlay = np.zeros_like(img, dtype=np.uint8)

        # 将分割结果mask复制到掩膜的所有通道中
        overlay[:, :, 0] = mask

        # 将掩膜中mask值为255的像素设置成半透明红色
        overlay[np.where(mask == 1)] = (0, 0, 255)

        # 将原始图像与掩膜叠加
        result = cv2.addWeighted(img, 1, overlay, 0.5, 0)
        return result

    def help(self):
        # 定义要打开的PDF文件路径
        pdf_path = "MediLab设计和开发文档.pdf"

        # 获取当前脚本文件的绝对路径
        script_path = os.path.abspath(__file__)

        # 获取当前脚本文件所在的目录
        script_dir = os.path.dirname(script_path)

        # 使用系统默认的PDF阅读器打开文件
        os.startfile(os.path.join(script_dir, pdf_path))


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyClass()
    myWin.show()
    sys.exit(app.exec_())
