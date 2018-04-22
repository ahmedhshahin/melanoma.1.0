import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, Multiply, RepeatVector, Flatten, Dense, Reshape, Concatenate, Maximum, merge, Add, Average, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as keras
from data import *
from sklearn.utils import class_weight
import numpy as np

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

class myUnet(object):

	def __init__(self, learning_rate, img_rows = 256, img_cols = 256):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.lr = learning_rate

	# def test(self, y_true, y_pred, smooth=1):
	# 	y_true_f = K.flatten(y_true)
	# 	y_pred_f = K.flatten(y_pred)
	# 	# y_pred_f[y_pred_f >= 0.5] = 1
	# 	# y_pred_f[y_pred_f < 0.5] = 0
	# 	intersection = K.sum(y_true_f * y_pred_f)
	# 	# print(np.unique(y_pred))
	# 	# print(np.unique(y_true))
	# 	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

	# def test_loss(self, y_true, y_pred):
	# 	return - self.test(y_true, y_pred)

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		# these two lines by A Hassaan
		mydata.create_train_data()
		mydata.create_test_data()

		###
		imgs_fft, imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test_fft, imgs_test = mydata.load_test_data()
		return imgs_fft, imgs_train, imgs_mask_train, imgs_test_fft, imgs_test

	def get_unet(self):

		inputs = [Input((self.img_rows, self.img_cols,1)), Input((self.img_rows, self.img_cols, 3))]
		# inputs = inputs[:, :, :, :2]
		# input_fft = inputs[:, :, :, 2]
		# print("++++++++++++++++++++++++++++")
		# print(inputs.shape)
		# print("++++++++++++++++++++++++++++")

		# i = Reshape((self.img_rows, self.img_cols, 1), input_shape=(self.img_rows, self.img_cols))(inputs[...,3])
		i = inputs[0]
		p1 = MaxPooling2D(pool_size=(2,2))(i)
		p2 = MaxPooling2D(pool_size=(2,2))(p1)
		p3 = MaxPooling2D(pool_size=(2,2))(p2)

		D1 = Flatten()(p3)
		D1 = Dense(256, use_bias=True, kernel_initializer='he_normal')(D1)
		D1 = BatchNormalization()(D1)
		D1 = Activation('relu')(D1)
		D2 = Dense(512, use_bias=True, kernel_initializer='he_normal')(D1)
		D2 = BatchNormalization()(D2)
		D2 = Activation('relu')(D2)
		# o = keras.ones((8,16,16,512))
		# D2 = Multiply()([D2, o])
		# F = keras.repeat_elements(D2, 16, axis=1)
		# F = MaxPooling2D(pool_size=(2,2))(D2)
		D2 = RepeatVector(256)(D2)
		F = Reshape((16,16,512))(D2)
		print("==========================")
		print("Shape is:", F.shape)
		print("==========================")
		
		
		# c1 = Conv2D(64, 3, padding= 'same', kernel_initializer = 'he_normal')(i)
		# c1 = BatchNormalization()(c1)
		# c1 = Activation('relu')(c1)
		# c1 = Conv2D(64, 3, padding= 'same', kernel_initializer = 'he_normal')(c1)
		# c1 = BatchNormalization()(c1)
		# c1 = Activation('relu')(c1)
		# p1 = MaxPooling2D(pool_size=(2,2))(c1)
		# print("p1 shape", p1.shape)

		# c2 = Conv2D(128, 3, padding= 'same', kernel_initializer = 'he_normal')(p1)
		# c2 = BatchNormalization()(c2)
		# c2 = Activation('relu')(c2)
		# c2 = Conv2D(128, 3, padding= 'same', kernel_initializer = 'he_normal')(c2)
		# c2 = BatchNormalization()(c2)
		# c2 = Activation('relu')(c2)
		# p2 = MaxPooling2D(pool_size=(2,2))(c2)
		# print("p2 shape", p2.shape)

		# c3 = Conv2D(256, 3, padding= 'same', kernel_initializer = 'he_normal')(p2)
		# c3 = BatchNormalization()(c3)
		# c3 = Activation('relu')(c3)
		# c3 = Conv2D(256, 3, padding= 'same', kernel_initializer = 'he_normal')(c3)
		# c3 = BatchNormalization()(c3)
		# c3 = Activation('relu')(c3)
		# p3 = MaxPooling2D(pool_size=(2,2))(c3)
		# print("p3 shape", p3.shape)

		# c4 = Conv2D(512, 3, padding= 'same', kernel_initializer = 'he_normal')(p3)
		# c4 = BatchNormalization()(c4)
		# c4 = Activation('relu')(c4)
		# c4 = Conv2D(512, 3, padding= 'same', kernel_initializer = 'he_normal')(c4)
		# c4 = BatchNormalization()(c4)
		# c4 = Activation('relu')(c4)
		# p4 = MaxPooling2D(pool_size=(2,2))(c4)
		# print("p4 shape", p4.shape)

		# c5 = Conv2D(1024, 3, padding= 'same', kernel_initializer = 'he_normal')(p4)
		# c5 = Conv2D(1024, 3, padding= 'same', kernel_initializer = 'he_normal')(c5)
		# print("p5 shape", c5.shape)
		# print("============================================================")
		
		'''
		unet with crop(because padding = valid) 

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
		print "conv1 shape:",conv1.shape
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
		print "conv1 shape:",conv1.shape
		crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
		print "crop1 shape:",crop1.shape
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
		print "conv2 shape:",conv2.shape
		crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
		print "crop2 shape:",crop2.shape
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
		print "conv3 shape:",conv3.shape
		crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
		print "crop3 shape:",crop3.shape
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		'''

		j = inputs[1]
		conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(j)
		print("conv1 shape:",conv1.shape)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		print("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print("conv2 shape:",conv2.shape)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		print("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print("conv3 shape:",conv3.shape)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		print("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		drop5 = Dropout(0.5)(conv5)
		print("DROP 5", drop5.shape)

		merge_fft = Concatenate()([drop5, F])
		print("MERGE FFT", merge_fft.shape)
		up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge_fft))
		up6 = BatchNormalization()(up6)
		up6 = Activation('relu')(up6)
		# merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		merge6 = Concatenate()([drop4, up6])
		conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(up6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)

		up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		up7 = BatchNormalization()(up7)
		up7 = Activation('relu')(up7)
		# merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		merge7 = Concatenate()([conv3, up7])
		conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(up7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)

		up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		up8 = BatchNormalization()(up8)
		up8 = Activation('relu')(up8)
		# merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		merge8 = Concatenate()([conv2, up8])
		conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(up8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)

		up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		up9 = BatchNormalization()(up9)
		up9 = Activation('relu')(up9)
		# merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		merge9 = Concatenate()([conv1, up9])
		conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(up9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		print("conv10 shape:",conv10.shape)

		model = Model(input = inputs, output = conv10)

		def dice_coef(y_true, y_pred, smooth=1):
			y_true_f = K.flatten(y_true)
			y_pred_f = K.flatten(y_pred)
			intersection = K.sum(y_true_f * y_pred_f)
			return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

		def dice_coef_loss(y_true, y_pred):
			return 1-dice_coef(y_true, y_pred)

		def Jac(y_true, y_pred):
			y_pred_f = K.flatten(K.round(y_pred))
			y_true_f = K.flatten(y_true)
			num = K.sum(y_true_f * y_pred_f)
			den = K.sum(y_true_f) + K.sum(y_pred_f) - num
			return num / den
		def soft_dice(y_pred, y_true):
    			# y_pred is softmax output of shape (num_samples, num_classes)
    			# y_true is one hot encoding of target (shape= (num_samples, num_classes))
			print("================================================") 
			print(K.min(y_pred))
			print("================================================")
			# te
			intersect = K.sum(y_pred * y_true, 0)
			denominator = K.sum(y_pred, 0) + K.sum(y_true, 0)
			dice_scores = -2 * intersect / (denominator + (1e-6))
			return K.mean(dice_scores[..., 0])

		def binary_crossentropy_wt(y_true, y_pred, from_logits=False):
		    """Binary crossentropy between an output tensor and a target tensor.

		    # Arguments
		        target: A tensor with the same shape as `output`.
		        output: A tensor.
		        from_logits: Whether `output` is expected to be a logits tensor.
		            By default, we consider that `output`
		            encodes a probability distribution.

		    # Returns
		        A tensor.
		    """
		    # Note: tf.nn.sigmoid_cross_entropy_with_logits
		    # expects logits, Keras expects probabilities.
		    t = class_weight.compute_class_weight('balanced', np.unique(y_true), K.flatten(y_true.eval(session = tf.keras.backend.get_session())))
		    weight_map = np.zeros(y_true.shape)
		    if not from_logits:
		        # transform back to logits
		        _epsilon = _to_tensor(epsilon(), y_pred.dtype.base_dtype)
		        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
		        y_pred = tf.log(y_pred / (1 - y_pred))
		    loss_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
		    weight_map[y_true == 0] = t[0]
		    weight_map[y_true == 1] = t[1]
		    weighted_loss = tf.multiply(loss_map, weight_map)
		    return tf.reduce_mean(weighted_loss)



		# model.compile(optimizer = Adam(lr = 1e-4), loss = ['binary_crossentropy'], metrics = [Jac, 'acc'])
		model.compile(optimizer = Adam(lr = self.lr), loss = ['binary_crossentropy'], metrics = [Jac, 'acc'])

		return model



	def train(self):

		print("loading data")
		imgs_fft, imgs_train, imgs_mask_train, imgs_test_fft, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		# t = class_weight.compute_class_weight('balanced', np.unique(imgs_mask_train), imgs_mask_train.flatten())
		history = model.fit([imgs_fft, imgs_train], imgs_mask_train, batch_size=8, epochs=150, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
		print('predict test data')
		imgs_mask_test = model.predict([imgs_test_fft, imgs_test], batch_size=1, verbose=1)
		np.save('/content/unet-keras/results/imgs_mask_test.npy', imgs_mask_test)
		np.save('/content/unet-keras/results/tr_loss.npy', history.history['loss'])
		np.save('/content/unet-keras/results/val_Jac.npy', history.history['val_Jac'])
		np.save('/content/unet-keras/results/val_loss.npy', history.history['val_loss'])
		


	def save_img(self):

		print("array to image")
		imgs = np.load('/content/unet-keras/results/imgs_mask_test.npy')
		names = np.load("/content/unet-keras/names.npy")
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			n = names[i]
			img.save("/content/unet-keras/results/{0}.jpg".format(n))




if __name__ == '__main__':
	myunet = myUnet(learning_rate=7e-4)
	myunet.train()
	myunet.save_img()








