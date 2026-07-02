
import math
import os

import cv2
import numpy as np
import scipy
from scipy import ndimage, signal

class FingerprintImageEnhancer:

    def __init__(
        self,
        ridge_segment_blksze=16,
        ridge_segment_thresh=0.1,
        gradient_sigma=1,
        block_sigma=7,
        orient_smooth_sigma=7,
        ridge_freq_blksze=38,
        ridge_freq_windsze=5,
        min_wave_length=5,
        max_wave_length=15,
        relative_scale_factor_x=0.65,
        relative_scale_factor_y=0.65,
        angle_inc=3.0,
        ridge_filter_thresh=-3,
    ):

        self.ridge_segment_blksze = ridge_segment_blksze
        self.ridge_segment_thresh = ridge_segment_thresh
        self.gradient_sigma = gradient_sigma
        self.block_sigma = block_sigma
        self.orient_smooth_sigma = orient_smooth_sigma
        self.ridge_freq_blksze = ridge_freq_blksze
        self.ridge_freq_windsze = ridge_freq_windsze
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.relative_scale_factor_x = relative_scale_factor_x
        self.relative_scale_factor_y = relative_scale_factor_y
        self.angle_inc = angle_inc
        self.ridge_filter_thresh = ridge_filter_thresh

        self._mask = []
        self._normim = []
        self._orientim = []
        self._mean_freq = []
        self._median_freq = []
        self._freq = []
        self._freqim = []
        self._binim = []

    def __normalise(self, img: np.ndarray) -> np.ndarray:

        if np.std(img) == 0:
            raise ValueError("Image standard deviation is 0. Please review image again")
        normed = (img - np.mean(img)) / (np.std(img))
        return normed

    def __ridge_segment(self, img: np.ndarray):

        rows, cols = img.shape
        normalized_im = self.__normalise(img)

        new_rows = int(self.ridge_segment_blksze * np.ceil((float(rows)) / (float(self.ridge_segment_blksze))))
        new_cols = int(self.ridge_segment_blksze * np.ceil((float(cols)) / (float(self.ridge_segment_blksze))))

        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))
        padded_img[0:rows][:, 0:cols] = normalized_im
        for i in range(0, new_rows, self.ridge_segment_blksze):
            for j in range(0, new_cols, self.ridge_segment_blksze):
                block = padded_img[i : i + self.ridge_segment_blksze][:, j : j + self.ridge_segment_blksze]

                stddevim[i : i + self.ridge_segment_blksze][:, j : j + self.ridge_segment_blksze] = np.std(block) * np.ones(
                    block.shape
                )

        stddevim = stddevim[0:rows][:, 0:cols]
        self._mask = stddevim > self.ridge_segment_thresh
        mean_val = np.mean(normalized_im[self._mask])
        std_val = np.std(normalized_im[self._mask])
        self._normim = (normalized_im - mean_val) / (std_val)

    def __ridge_orient(self) -> None:

        sze = np.fix(6 * self.gradient_sigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1

        gauss = cv2.getGaussianKernel(int(sze), self.gradient_sigma)
        filter_gauss = gauss * gauss.T

        filter_grad_y, filter_grad_x = np.gradient(filter_gauss)

        gradient_x = signal.convolve2d(self._normim, filter_grad_x, mode="same")
        gradient_y = signal.convolve2d(self._normim, filter_grad_y, mode="same")

        grad_x2 = np.power(gradient_x, 2)
        grad_y2 = np.power(gradient_y, 2)
        grad_xy = gradient_x * gradient_y

        sze = np.fix(6 * self.block_sigma)

        gauss = cv2.getGaussianKernel(int(sze), self.block_sigma)
        filter_gauss = gauss * gauss.T

        grad_x2 = ndimage.convolve(grad_x2, filter_gauss)
        grad_y2 = ndimage.convolve(grad_y2, filter_gauss)
        grad_xy = 2 * ndimage.convolve(grad_xy, filter_gauss)

        denom = np.sqrt(np.power(grad_xy, 2) + np.power((grad_x2 - grad_y2), 2)) + np.finfo(float).eps

        sin_2_theta = grad_xy / denom
        cos_2_theta = (grad_x2 - grad_y2) / denom

        if self.orient_smooth_sigma:
            sze = np.fix(6 * self.orient_smooth_sigma)
            if np.remainder(sze, 2) == 0:
                sze = sze + 1
            gauss = cv2.getGaussianKernel(int(sze), self.orient_smooth_sigma)
            filter_gauss = gauss * gauss.T
            cos_2_theta = ndimage.convolve(cos_2_theta, filter_gauss)
            sin_2_theta = ndimage.convolve(sin_2_theta, filter_gauss)

        self._orientim = np.pi / 2 + np.arctan2(sin_2_theta, cos_2_theta) / 2

    def __ridge_freq(self):

        rows, cols = self._normim.shape
        freq = np.zeros((rows, cols))

        for i in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
            for j in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):
                blkim = self._normim[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze]
                blkor = self._orientim[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze]

                freq[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze] = self.__frequest(blkim, blkor)

        self._freq = freq * self._mask
        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]

        self._mean_freq = np.mean(non_zero_elems_in_freq)
        self._median_freq = np.median(non_zero_elems_in_freq)

        self._freq = self._mean_freq * self._mask

    def __frequest(self, blkim: np.ndarray, blkor: np.ndarray) -> np.ndarray:

        rows, _ = np.shape(blkim)

        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = math.atan2(sinorient, cosorient) / 2

        rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode="nearest")

        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset : offset + cropsze][:, offset : offset + cropsze]

        proj = np.sum(rotim, axis=0)
        dilation = scipy.ndimage.grey_dilation(proj, self.ridge_freq_windsze, structure=np.ones(self.ridge_freq_windsze))

        temp = np.abs(dilation - proj)

        peak_thresh = 2

        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)

        _, cols_maxind = np.shape(maxind)

        if cols_maxind < 2:
            return np.zeros(blkim.shape)
        no_of_peaks = cols_maxind
        wave_length = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (no_of_peaks - 1)
        if self.min_wave_length <= wave_length <= self.max_wave_length:
            return 1 / np.double(wave_length) * np.ones(blkim.shape)
        return np.zeros(blkim.shape)

    def __ridge_filter(self):

        norm_im = np.double(self._normim)
        rows, cols = norm_im.shape
        newim = np.zeros((rows, cols))

        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]


        non_zero_elems_in_freq = freq_1d[0][ind]
        non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

        unfreq = np.unique(non_zero_elems_in_freq)


        sigmax = 1 / unfreq[0] * self.relative_scale_factor_x
        sigmay = 1 / unfreq[0] * self.relative_scale_factor_y

        sze = int(np.round(3 * np.max([sigmax, sigmay])))

        mesh_x, mesh_y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

        reffilter = np.exp(-(((np.power(mesh_x, 2)) / (sigmax * sigmax) + (np.power(mesh_y, 2)) / (sigmay * sigmay)))) * np.cos(
            2 * np.pi * unfreq[0] * mesh_x
        )

        filt_rows, filt_cols = reffilter.shape

        angle_range = int(180 / self.angle_inc)

        gabor_filter = np.array(np.zeros((angle_range, filt_rows, filt_cols)))

        for filter_idx in range(0, angle_range):


            rot_filt = scipy.ndimage.rotate(reffilter, -(filter_idx * self.angle_inc + 90), reshape=False)
            gabor_filter[filter_idx] = rot_filt

        maxsze = int(sze)

        temp = self._freq > 0
        validr, validc = np.where(temp)

        temp1 = validr > maxsze
        temp2 = validr < rows - maxsze
        temp3 = validc > maxsze
        temp4 = validc < cols - maxsze

        final_temp = temp1 & temp2 & temp3 & temp4

        finalind = np.where(final_temp)

        maxorientindex = np.round(180 / self.angle_inc)
        orientindex = np.round(self._orientim / np.pi * 180 / self.angle_inc)

        for i in range(0, rows):
            for j in range(0, cols):
                if orientindex[i][j] < 1:
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if orientindex[i][j] > maxorientindex:
                    orientindex[i][j] = orientindex[i][j] - maxorientindex
        _, finalind_cols = np.shape(finalind)
        sze = int(sze)
        for k in range(0, finalind_cols):
            cur_r = validr[finalind[0][k]]
            cur_c = validc[finalind[0][k]]

            img_block = norm_im[cur_r - sze : cur_r + sze + 1][:, cur_c - sze : cur_c + sze + 1]

            newim[cur_r][cur_c] = np.sum(img_block * gabor_filter[int(orientindex[cur_r][cur_c]) - 1])

        self._binim = newim < self.ridge_filter_thresh

    def save_enhanced_image(self, path: str) -> None:

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, (255 * self._binim))

    def enhance(self, img: np.ndarray, resize: bool = True, invert_output=False) -> np.ndarray:

        if resize:
            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)

            new_rows = 350
            new_cols = new_rows / aspect_ratio

            img = cv2.resize(img, (int(new_cols), int(new_rows)))

        self.__ridge_segment(img)
        self.__ridge_orient()
        self.__ridge_freq()
        self.__ridge_filter()
        if invert_output:
            self._binim ^= True
        return self._binim


def enhance_fingerprint(
    img: np.ndarray,
    resize: bool = False,
    ridge_segment_blksze: int = 16,
    ridge_segment_thresh: float = 0.1,
    gradient_sigma: int = 1,
    block_sigma: int = 7,
    orient_smooth_sigma: int = 7,
    ridge_freq_blksze: int = 38,
    ridge_freq_windsze: int = 5,
    min_wave_length: int = 5,
    max_wave_length: int = 15,
    relative_scale_factor_x: float = 0.65,
    relative_scale_factor_y: float = 0.65,
    angle_inc: float = 3.0,
    ridge_filter_thresh: int = -3,
    invert_output: bool = False,
) -> np.ndarray:

    image_enhancer = FingerprintImageEnhancer(
        ridge_segment_blksze,
        ridge_segment_thresh,
        gradient_sigma,
        block_sigma,
        orient_smooth_sigma,
        ridge_freq_blksze,
        ridge_freq_windsze,
        min_wave_length,
        max_wave_length,
        relative_scale_factor_x,
        relative_scale_factor_y,
        angle_inc,
        ridge_filter_thresh,
    )
    enhanced_output = image_enhancer.enhance(img, resize, invert_output=invert_output)
    return enhanced_output

