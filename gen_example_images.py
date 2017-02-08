'''
Generate example images to illustrate different pipeline stages' outputs
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz


# Read camera calibration coefficients
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

# Create example pipeline images for all test images
image_files = os.listdir('test_images')
for image_file in image_files:
	out_image_file = image_file.split('.')[0] + '.png'  # write to png format
	img = mpimg.imread('test_images/' + image_file)

	# Undistort image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(img)
	plt.savefig('example_images/undistort_' + out_image_file)

	# Thresholded binary image
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.savefig('example_images/binary_' + out_image_file)

	# Perspective transform
	img, binary_unwarped, m, m_inv = perspective_transform(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.savefig('example_images/warped_' + out_image_file)

	# Polynomial fit
	ret = line_fit(img)
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	save_file = 'example_images/polyfit_' + out_image_file
	viz2(img, ret, save_file=save_file)

	# Do full annotation on original image
	# Code is the same as in 'line_fit_video.py'
	orig = mpimg.imread('test_images/' + image_file)
	undist = cv2.undistort(orig, mtx, dist, None, mtx)
	left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	vehicle_offset *= xm_per_pix

	img = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
	plt.imshow(img)
	plt.savefig('example_images/annotated_' + out_image_file)

