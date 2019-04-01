from natsort import natsorted,ns
import os
import glob
import numpy as np
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Activation,Dense,Flatten,Reshape,Permute,concatenate,Lambda
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
from keras.utils.generic_utils import get_custom_objects
import cv2
import pyflow
import tensorflow as tf 
from keras.losses import mean_squared_error

def dice(im1,im2):
	intersection = np.logical_and(im1, im2)
	return 2. * intersection.sum() / (im1.sum() + im2.sum())


def warp_flow_2d_output_shape(input_shape):
	shape = list(input_shape)
	# assert len(shape) == 2  # only valid for 2D tensors
	# shape[-1] *= 2
	shape[-1]-=2
	return tuple(shape)


def input_for_warp(input):
	image1=input[:,:,:,0]
	image2=input[:,:,:,1]
	image3=input[:,:,:,2]
	image=tf.stack((image1,image2,image3),axis=-1)
	flow1=input[:,:,:,3]
	flow2=input[:,:,:,4]
	flow=tf.stack((flow1,flow2),axis=-1)
	# image=image[...,tf.newaxis]
	xrange = tf.range(0,256,dtype=tf.float32)
	yrange = tf.range(0,256,dtype=tf.float32)
	gridx,gridy= tf.meshgrid(xrange,yrange)
	c1=tf.add(flow[:,:,:,0] , gridx)
	c2=tf.add(flow[:,:,:,1] , gridy)
	# print ('C shapes ',c1.shape,c2.shape)
	C=tf.stack([c1,c2],-1)

	res=tf.contrib.resampler.resampler(image,C)
	return res


class myUnet(object):

	def __init__(self, img_rows = 256, img_cols = 256):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		imgdatas=np.load('./imgdatas_double.npy')


		pyflows=np.load('./newflow_double.npy')

		next_frames=np.load('./next_frames_double_.npy')


		masks=np.load('./masks_double.npy')

		test_images=np.load('./imgdatas_double_test.npy')
		test_images_masks=np.load('./masks_double_test_multi.npy')
		test_next_frames_masks=np.load('./next_frames_double_test.npy')
		test_flows=np.load('./newflow_testing.npy')

		return imgdatas,pyflows,next_frames,masks,test_images,test_images_masks,test_next_frames_masks,test_flows
		
	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols,1))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		model = Model(input = inputs, output = conv10)

		return model

	def create_net(self):
		
		frame_t=Input((self.img_rows, self.img_cols,1),name='Frame_1_input')
		frame_t1=Input((self.img_rows, self.img_cols,1),name='Frame_2_input')
		flows = Input((self.img_rows,self.img_cols,2),name='Flow')
		
		########################## Create Multi-Class Unet ###################################
		unet_model=self.get_unet()
		x=unet_model.get_layer('conv2d_22').output
		conv11 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		activation=Activation('softmax')(conv11)
		unet_model = Model(input = unet_model.input, output =activation)
		
		########################## Create Double Net #########################################
		encoded_t=unet_model(frame_t)
		encoded_t1=unet_model(frame_t1)


		merge1 = merge([encoded_t1,flows], mode = 'concat', concat_axis = -1)
		final=Lambda(input_for_warp,output_shape=warp_flow_2d_output_shape)(merge1)


		model=Model(input=[frame_t,frame_t1,flows],outputs=[final,encoded_t])

		model.compile(optimizer = Adam(lr = 1e-5), loss = ['categorical_crossentropy','categorical_crossentropy'], metrics = ['accuracy'])
		return model
		
	def train(self):
		print("loading data")
		imgdatas,pyflows,next_frames,masks,test_images,test_images_masks,test_next_frames_masks,test_flows = self.load_data()
		print("loading data done")

		model=self.create_net()
		print("got unet")
		model.load_weights('unet_double_multi_newflows_12.hdf5')
		print("Loaded Weights...")
		model_checkpoint = ModelCheckpoint('unet_double_multi.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit({'Frame_1_input':imgdatas,'Frame_2_input':next_frames,'Flow':pyflows}, [masks,masks], batch_size=4, epochs=100,verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
		print('predict test data')
		imgs_mask_test_1,imgs_mask_test_2 = model.predict({'Frame_1_input':test_images,'Frame_2_input':test_next_frames_masks,'Flow':test_flows}, batch_size=1, verbose=1)
		np.save('./results/imgs_mask_test_1.npy', imgs_mask_test_1)
		np.save('./results/imgs_mask_test_2.npy', imgs_mask_test_2)


	def save_img(self):

		print("array to image")
		imgs = np.load('./results/imgs_mask_test_1.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save('./results/multi_images/'+str(i)+'_1.png')
		imgs = np.load('./results/imgs_mask_test_2.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save('./results/multi_images/'+str(i)+'_2.png')


	def get_quant_measure(self):
		masks=np.load('./masks_double_test_multi.npy')
		imgs = np.load('./results/imgs_mask_test_1.npy')
		dices=0
		for i in range(imgs.shape[0]):
			img = imgs[i]
			mask=masks[i]
			dices+=dice(img,mask)
		length_of_images=imgs.shape[0]
		print ('Average Dice_1=',dices/length_of_images)
		imgs = np.load('./results/imgs_mask_test_2.npy')
		dices=0
		for i in range(imgs.shape[0]):
			img = imgs[i]
			mask=masks[i]
			dices+=dice(img,mask)
		length_of_images=imgs.shape[0]
		print ('Average Dice_2 =',dices/length_of_images)


			

if __name__ == '__main__':
	tf.set_random_seed(1)
	myunet=myUnet()
	myunet.train()	
	myunet.save_img()
	myunet.get_quant_measure()

