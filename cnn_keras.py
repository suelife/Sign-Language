import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()

	# Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	# 16是kernel數量 
	# (2,2)是kernelsize 
	# input_shape=(image_x, image_y, 1)是輸入為 image_x 乘上 image_y channel=1
	model.add(Conv2D(32, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))

	# MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
	# pool_size=(2, 2) maxpooling size
	# strides=(2, 2) 步距
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(64, (4,4), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Conv2D(128, (4,4), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

	# 平坦化
	model.add(Flatten())

	# 連接 Fully Connected Layer，接著一層 ReLU 的 Activation 函數
	model.add(Dense(128, activation='relu'))

	# Dropout Layer
	model.add(Dropout(0.4))

	# 連接 Fully Connected Layer，接著一層 Softmax 的 Activation 函數
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	
	# 設定模型的 Loss 函數、Optimizer 以及用來判斷模型好壞的依據（metrics）
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras2.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint1]
	from keras.utils import plot_model
	plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	test_labels = np_utils.to_categorical(test_labels)

	model, callbacks_list = cnn_model()
	model.summary()
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=500, callbacks=callbacks_list)
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()
K.clear_session();
