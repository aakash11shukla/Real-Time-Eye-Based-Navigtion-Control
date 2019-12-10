import numpy as np
import cv2

# units in cm
screen_w = 34.59
screen_h = 19.46
screen_aspect = screen_w / screen_h
camera_l = 18.05
camera_t = 0.8
screen_t = 1.6
screen_l = 0.78
pc_w = 36.1
pc_h = 24.3
screen_from_camera = [screen_t - camera_t, screen_l - camera_l]

camera_coords_percentage = [camera_t / pc_h, camera_l / pc_w]

screenW = 1920
screenH = 1080

pc_w_to_screen = pc_w / screen_w
pc_h_to_screen = pc_h / screen_h

def render_gaze(full_image, camera_center, cm_to_px, output):
	xScreen = output[1]
	yScreen = output[0]
	pixelGaze = [round(camera_center[0] - yScreen * cm_to_px), round(camera_center[1] + xScreen * cm_to_px)]
	
	cv2.circle(full_image,(int(pixelGaze[1]), int(pixelGaze[0])), 6, (0, 0, 255), -1)


def render_gazes(img, outputs):
	full_image = np.ones((round(img.shape[0] * 1), round(img.shape[1] * 1), 3), dtype=np.uint8)

	full_image_center = [round(full_image.shape[0] * 0.5), round(full_image.shape[1] *0.5)]
	camera_center = full_image_center

	cm_to_px = img.shape[0] * 1. / screen_h

	screen_from_camera_px = [round(screen_from_camera[0] * cm_to_px), round(screen_from_camera[1] * cm_to_px)]

	screen_start = [camera_center[0] + screen_from_camera_px[0], camera_center[1] + screen_from_camera_px[1]]
	
	full_image[:, :, :] = img[:, :, :]

	cv2.circle(full_image,(camera_center[1],camera_center[0]), 7, (255, 0, 0), -1)
	
	for output in outputs:
		if output is not None:
			render_gaze(full_image, camera_center, cm_to_px, output)

	return full_image