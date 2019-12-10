import os
import sys
import time
import gaze
import glob
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

dataset_path = './Dataset/'

print('Reading Dataset files...')
img_files = glob.glob(dataset_path+'*.png')
print('Done')

lables_files = 'labels.txt'

train_labels = dict()

print('Creating a map from img to coordinates')
with open(lables_files, 'r') as f:
	for line in f:
		img = line.split()[0]
		label = [float(line.split()[1][1:-1]), float(line.split()[2][:-1])]
		train_labels[img] = label
print('Done')


NUM_CLASSES = 2

LEFT_EYE = None
RIGHT_EYE = None
FACE_MASK = None
FACE = None

img_size = (64, 64)
mask_size = (25, 25)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
	def __init__(self, PATH_TO_CKPT):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		with self.detection_graph.as_default():
			config = tf.compat.v1.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
			self.windowNotSet = True


	def run(self, image):
		
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
		# start_time = time.time()
		(boxes, scores, classes, num_detections) = self.sess.run(
						[boxes, scores, classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})
		# elapsed_time = time.time() - start_time
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

def main():

	tDetector = TensoflowFaceDector(PATH_TO_CKPT)
	sess = tf.compat.v1.Session()
	val_ops = itracker_adv.load_model(sess, PATH_TO_META_GRAPH)
	model = load_model('./model/2018_12_17_22_58_35.h5')
	
	final = dict()

	final['train_eye_left'] = np.array([])
	final['train_eye_right'] = np.array([])
	final['train_face'] = np.array([])
	final['train_face_mask'] = np.array([])
	final['train_y'] = np.array([])

	numpy_file = 'np.npz'

	if os.path.exists(numpy_file):
		numpy_array = np.load(numpy_file)
		final['train_eye_left'] = numpy_array['train_eye_left']
		final['train_eye_right'] = numpy_array['train_eye_right']
		final['train_face'] = numpy_array['train_face']
		final['train_face_mask'] = numpy_array['train_face_mask']
		final['train_y'] = numpy_array['train_y']

	print('Starting the process...')

	for file in img_files:

		file = ('/').join(file.split('\\'))
		print(file)
		frame = cv2.imread(file)

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
			FACE = np.array(cv2.cvtColor(frame[FACE_RECT[1]:FACE_RECT[3], FACE_RECT[0]:FACE_RECT[2]], cv2.COLOR_BGR2RGB), dtype=np.uint8)

			input_image = np.zeros(frame.shape)
			input_image[FACE_RECT[1]:FACE_RECT[3], FACE_RECT[0]:FACE_RECT[2]] = np.ones((FACE_RECT[3]-FACE_RECT[1], FACE_RECT[2]-FACE_RECT[0], 3))

			FACE = transform.resize(FACE, output_shape=img_size, preserve_range=True)
			FACE_MASK = input_image
			FACE_MASK = transform.resize(input_image, output_shape=mask_size, preserve_range=True)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			shape = predictor(gray, face_rect)
			shape = face_utils.shape_to_np(shape)

			eye_img_l, eye_rect_l = crop_eye(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), eye_points=shape[36:42])
			eye_img_r, eye_rect_r = crop_eye(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), eye_points=shape[42:48])

			LEFT_EYE = transform.resize(eye_img_l, output_shape=img_size)
			RIGHT_EYE = transform.resize(eye_img_r, output_shape=img_size)

			LEFT_EYE = LEFT_EYE.reshape(1, *img_size, 3)
			RIGHT_EYE = RIGHT_EYE.reshape(1, *img_size, 3)
			FACE = FACE.reshape(1, *img_size, 3)
			FACE_MASK = FACE_MASK[:, :, :1]

			final['train_eye_left'] = np.append(final['train_eye_left'], LEFT_EYE)
			final['train_eye_right'] = np.append(final['train_eye_right'], RIGHT_EYE)
			final['train_face'] = np.append(final['train_face'], FACE)
			final['train_face_mask'] = np.append(final['train_face_mask'], FACE_MASK)
			final['train_y'] = np.append(final['train_y'], train_labels[file])
			print('Done')
	print('Done')
	np.savez(numpy_file, final)

if __name__ == "__main__":
	main()