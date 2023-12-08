import numpy as np
from skimage import color
from utils import normal_vec2img

# no sorting
def _myPMS(data, mask):
    # shape of the image
    height, width = data['mask'].shape
    num_images = len(data['imgs'])
    # number of valid pixels
    valid_num = len(mask[0])
    
    gt_intensities = np.zeros((valid_num, num_images))
    # transform 3 channels to 1 channel
    for i in range(num_images):
        img_gray = color.rgb2gray(data['imgs'][i])
        img_gray = img_gray[mask[0], mask[1]]
        gt_intensities[:, i] = img_gray
    gt_intensities = gt_intensities.T
    print('ground truth intensities obtained')
    
    directions = data['s']
    # least square
    valid_normals = np.linalg.lstsq(directions, gt_intensities, rcond=None)[0].T
    # normalize
    valid_normals = np.array([row / np.linalg.norm(row) for row in valid_normals])
    # retain the original shape
    normals = np.zeros((height, width, 3))
    normals[mask[0], mask[1], :] = valid_normals
    return normals
    
def myPMS(data, mask):
    proportion = .2
    render_direction = np.array([0, 0, 1])
    render_intensity = np.array([.8, 1, .8])
    # shape of the image
    height, width = data['mask'].shape
    num_images = len(data['imgs'])
    # number of valid pixels
    valid_num = len(mask[0])
    
    gt_intensities = np.zeros((valid_num, num_images))
    # transform 3 channels to 1 channel
    for i in range(num_images):
        img_gray = color.rgb2gray(data['imgs'][i])
        img_gray = img_gray[mask[0], mask[1]]
        gt_intensities[:, i] = img_gray
    gt_intensities = gt_intensities.T
    print('ground truth intensities obtained')
    
    # sort on each pixel
    sorted_idx = np.argsort(gt_intensities, axis=0)
    sorted_idx = sorted_idx[int(num_images * proportion): int(num_images * (1-proportion)), :]
    directions = data['s']
    # least square
    valid_normals = np.zeros((valid_num, 3))
    valid_albedo = np.zeros((valid_num))
    valid_render = np.zeros((valid_num, 3))
    for i in range(valid_num):
        valid_normals[i] = np.linalg.lstsq(directions[sorted_idx[:, i]], gt_intensities[sorted_idx[:, i], i], rcond=None)[0]  # rho * n
        render_I = max(0, np.dot(render_direction, valid_normals[i]))
        valid_render[i] =  render_I * render_intensity
        valid_albedo[i] = np.linalg.norm(valid_normals[i])
        # normalize
        valid_normals[i] = valid_normals[i] / (valid_albedo[i] + 1e-8)
    
    # retain the original shape
    normals = np.zeros((height, width, 3))
    normals[mask[0], mask[1], :] = valid_normals
    albedo = np.zeros((height, width))
    albedo[mask[0], mask[1]] = valid_albedo
    render = np.zeros((height, width, 3))
    render[mask[0], mask[1], :] = valid_render
    return albedo, normals, render