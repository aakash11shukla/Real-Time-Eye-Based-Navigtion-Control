import os
import sys
import time
import cv2, dlib
import numpy as np
import tensorflow as tf
import itracker_adv
from skimage import transform
from imutils import face_utils
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import label_map_util
from scipy.spatial import distance as dist
from utils import visualization_utils_color as vis_util

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

PATH_TO_META_GRAPH = 'model/model-23'

NUM_CLASSES = 2

LEFT_EYE = None
RIGHT_EYE = None
FACE_MASK = None
FACE = None

img_size = (64, 64)
mask_size = (25, 25)

fig = plt.figure()
plt.axis('off')

ax_face = fig.add_subplot(141)
ax_face_mask = fig.add_subplot(142)
ax_l_eye = fig.add_subplot(143)
ax_r_eye = fig.add_subplot(144)

ax_face.axis('off')
ax_face_mask.axis('off')
ax_l_eye.axis('off')
ax_r_eye.axis('off')

ax_face.set_title('Face')
ax_face_mask.set_title('Face Mask')
ax_l_eye.set_title('Left Eye')
ax_r_eye.set_title('Right Eye')

im_face = ax_face.imshow(np.zeros(img_size))
im_face_mask = ax_face_mask.imshow(np.zeros(mask_size))
im_l_eye = ax_l_eye.imshow(np.zeros(img_size))
im_r_eye = ax_r_eye.imshow(np.zeros(img_size))
plt.ion()

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
	def __init__(self, PATH_TO_CKPT):
		"""Tensorflow detector
		"""
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		with self.detection_graph.as_default():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(graph=self.detection_graph, config=config)
			self.windowNotSet = True


	def run(self, image):
		"""image: bgr image
		return (boxes, scores, classes, num_detections)
		"""

		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
		# Actual detection.
		start_time = time.time()
		(boxes, scores, classes, num_detections) = self.sess.run(
						[boxes, scores, classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})
		elapsed_time = time.time() - start_time
		#print('inference time cost: {}'.format(elapsed_time))

		return (boxes, scores, classes, num_detections)

def crop_eye(img, eye_points):
	x1, y1 = np.amin(eye_points, axis=0)
	x2, y2 = np.amax(eye_points, axis=0)
	cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

	w = (x2 - x1) * 1.2
	h = w * IMG_SIZE[1] / IMG_SIZE[0]

	margin_x, margin_y = w / 2, h / 2

	min_x, min_y = int(cx - margin_x), int(cy - margin_y)
	max_x, max_y = int(cx + margin_x), int(cy + margin_y)

	eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

	eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

	return eye_img, eye_rect

def get_coordinates(sess, val_ops, val_data):

	eye_left, eye_right, face, face_mask, pred = val_ops

	LEFT_EYE, RIGHT_EYE, FACE, FACE_MASK = val_data

	y_pred = sess.run(pred, feed_dict={eye_left: LEFT_EYE, eye_right: RIGHT_EYE, face: FACE, face_mask: FACE_MASK})
	return y_pred


def main():
	# open the camera,load the cnn model 
	camera = cv2.VideoCapture(0)

	# blinks is the number of total blinks ,close_counter
	# the counter for consecutive close predictions
	# and mem_counter the counter of the previous loop 
	close_counter = blinks = mem_counter= 0
	state = ''
	tDetector = TensoflowFaceDector(PATH_TO_CKPT)
 
	model = None
	val_ops = None
	sess = tf.Session()
	val_ops = itracker_adv.load_model(sess, PATH_TO_META_GRAPH)

	if model == None:	
		model = load_model('./model/2018_12_17_22_58_35.h5')
	
	while True:

		flag=True
		ret, frame = camera.read()

		[h, w] = frame.shape[:2]
		frame = cv2.flip(frame, 1)
		(boxes, scores, classes, num_detections) = tDetector.run(frame)

		vis_util.visualize_boxes_and_labels_on_image_array(
								frame,
								np.squeeze(boxes),
								np.squeeze(classes).astype(np.int32),
								np.squeeze(scores),
								category_index,
								use_normalized_coordinates=True,
						max_boxes_to_draw=1,
						min_score_thresh=0.4,
						line_thickness=4)

		# detect eyes
		if scores[0][0] > 0.2:

			# keep the face region from the whole frame
			face_rect = dlib.rectangle(left = int(boxes[0,0,1]*w), top = int(boxes[0,0,0]*h),
													right = int(boxes[0,0,3]*w), bottom = int(boxes[0,0,2]*h))

			FACE_RECT = np.rint([int(boxes[0,0,1]*w), int(boxes[0,0,0]*h), int(boxes[0,0,3]*w), int(boxes[0,0,2]*h)]).astype(np.int)
			FACE = np.array(cv2.cvtColor(frame[FACE_RECT[1]:FACE_RECT[3], FACE_RECT[0]:FACE_RECT[2]], cv2.COLOR_BGR2RGB))

			im_face.set_data(FACE)

			input_image = np.zeros(frame.shape)
			input_image[FACE_RECT[1]:FACE_RECT[3], FACE_RECT[0]:FACE_RECT[2]] = np.ones((FACE_RECT[3]-FACE_RECT[1], FACE_RECT[2]-FACE_RECT[0], 3))

			FACE = transform.resize(FACE, output_shape=img_size)
			FACE_MASK = transform.resize(input_image, output_shape=mask_size)
			im_face_mask.set_data(FACE_MASK)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			shape = predictor(gray, face_rect)
			shape = face_utils.shape_to_np(shape)

			eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shape[36:42])
			eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shape[42:48])

			LEFT_EYE = transform.resize(np.array(cv2.cvtColor(eye_img_l, cv2.COLOR_GRAY2RGB)), output_shape=img_size)
			RIGHT_EYE = transform.resize(np.array(cv2.cvtColor(eye_img_r, cv2.COLOR_GRAY2RGB)), output_shape=img_size)

			im_l_eye.set_data(LEFT_EYE)
			im_r_eye.set_data(RIGHT_EYE)

			plt.pause(0.0001)
			plt.show()

			LEFT_EYE = LEFT_EYE.reshape(1, *img_size, 3)
			RIGHT_EYE = RIGHT_EYE.reshape(1, *img_size, 3)
			FACE = FACE.reshape(1, *img_size, 3)
			FACE_MASK = FACE_MASK[:, :, :1]
			FACE_MASK = FACE_MASK.reshape(1, 625)

			
			val_data = [LEFT_EYE, RIGHT_EYE, FACE, FACE_MASK]
			coord = get_coordinates(sess, val_ops, val_data)
			coord *= 255.
			x, y = coord[0][0], coord[0][1]
			x += frame.shape[0]/2
			y += frame.shape[1]/2
			print(x,y)

			eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
			eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
			eye_img_r = cv2.flip(eye_img_r, flipCode=1)


			eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
			eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

			pred_l = model.predict(eye_input_l)
			pred_r = model.predict(eye_input_r)

			# visualize
			state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
			state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

			state_l = state_l % pred_l
			state_r = state_r % pred_r

			pred = (pred_l+pred_r)/2

			cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
			cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

			cv2.putText(frame, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
			cv2.putText(frame, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

			# blinks
			# if the eyes are open reset the counter for close eyes
			
			if pred > 0.3 :
				state = 'open'
				close_counter = 0
			else:
				state = 'close'
				close_counter += 1
			
			# if the eyes are open and previousle were closed
			# for sufficient number of frames then increcement 
			# the total blinks
			if state == 'open' and mem_counter > 1:
				blinks += 1
			# keep the counter for the next loop 
			mem_counter = close_counter 

			# draw the total number of blinks on the frame along with
			# the state for the frame
			cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "State: {}".format(state), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# show the frame
		cv2.imshow('Blinks Counter', frame)

		if cv2.waitKey(1) == ord('q'):
			break

	cv2.destroyAllWindows()
	del(camera)

if __name__ == "__main__":
	main()