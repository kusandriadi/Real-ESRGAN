from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(grount_truth_img_arr, hr_img_arr):
    return psnr(grount_truth_img_arr, hr_img_arr, data_range=255)