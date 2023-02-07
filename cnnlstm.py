import os as os
os.environ['PYTHONHASHSEED']='0'
import numpy as np
np.random.seed(123)
import random as rn
rn.seed(123)
from tensorflow import set_random_seed
set_random_seed(123)

from keras import backend as K
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten, TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import adam, sgd, adadelta
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

num_classes = 14
bs=16;
path='DHG\\DHG2016CROP2\\'
keyPath='DHG\\DHG2016\\informations_troncage_sequences.txt'
keyPoints=np.genfromtxt(keyPath,delimiter=" ")
#The line format is as follows: #gesture - #finger - #subject - #essai - # frame of the the effective beginning of the gesture - # frame of the the effective end of the gesture.


#keypoints
kpt=np.zeros((14,2,20,5,2))

for kp in range(0,len(keyPoints)):
	for g in range(1,15):
		if keyPoints[kp][0]==g:
			for f in range(1,3):
				if keyPoints[kp][1]==f:
					for s in range (1,21):
						if keyPoints[kp][2]==s:
							for e in range (1, 6):
								if keyPoints[kp][3]==e:
									kpt[g-1][f-1][s-1][e-1][0]=keyPoints[kp][4]
									kpt[g-1][f-1][s-1][e-1][1]=keyPoints[kp][5]

t=os.listdir(path)
precnn=1
modelP=[]
trueP=[]
storeSet=[]
storeLbl=[]
for subTest in range(1,21):
	depthTrainLbl=[]
	depthTestLbl=[]
	depthTestSet=[]
	depthTrainSet=[]
	cnnTestSet=[]
	cnnTrainSet=[]
	cnnTestLbl=[]
	cnnTrainLbl=[]

    #load data once
	if subTest==1:
		for fLen in range(0,len(t)):
			fingerPath=path + 'finger_' + str(fLen+1) + '\\'
			fingerDir=os.listdir(fingerPath)
			for sLen in range(0,len(fingerDir)):
				subjectPath=fingerPath + 'subject_' + str(sLen+1) + '\\'
				subjectDir=os.listdir(subjectPath)
				for eLen in range (0,len(subjectDir)):
					essaiPath=subjectPath + 'essai_' + str(eLen+1) + '\\'
					essaiDir=os.listdir(essaiPath)
					for gLen in range (0,len(essaiDir)):
						gesturePath = essaiPath + 'gesture_' + str(gLen+1) + '\\'
						gestureDir = os.listdir(gesturePath)

						st=kpt[gLen][fLen][sLen][eLen][0]
						en=kpt[gLen][fLen][sLen][eLen][1]
						depthImg=[]

						for dLen in range (int(st)+1,int(en)):
							depthPath= gesturePath + 'depth_' + str(dLen) + '.png'
							img=load_img(depthPath,grayscale=1)
							x=img_to_array(img)
							x=x.astype('float16')
							x=x/255.0
							depthImg.append(x)
							if precnn==1:
								if subTest == sLen+1:
									cnnTestSet.append(x)
									cnnTestLbl.append(gLen)
								else:
									cnnTrainSet.append(x)
									cnnTrainLbl.append(gLen)
						storeSet.append(depthImg)
						storeLbl.append(gLen)

		storeSet[0:len(storeSet)//2]=sequence.pad_sequences(storeSet[0:len(storeSet)//2],maxlen=2**5,truncating='pre',dtype='float16',padding='pre',value=0)
		storeSet[len(storeSet)//2:len(storeSet)]=sequence.pad_sequences(storeSet[len(storeSet)//2:len(storeSet)],maxlen=2**5,truncating='pre',dtype='float16',padding='pre',value=0)

        #manually separate data into train/test sets based on subject number
		depthTrainSet=storeSet[(subTest)*70:len(storeSet)//2]+storeSet[(len(storeSet)//2)+((subTest)*70):len(storeSet)]
		depthTestSet=storeSet[(subTest-1)*70:subTest*70]+storeSet[(len(storeSet)//2)+((subTest-1)*70):(len(storeSet)//2)+(subTest*70)]
		depthTrainLbl=storeLbl[(subTest)*70:len(storeSet)//2]+storeLbl[(len(storeSet)//2)+((subTest)*70):len(storeSet)]
		depthTestLbl=storeLbl[(subTest-1)*70:subTest*70]+storeLbl[(len(storeSet)//2)+((subTest-1)*70):(len(storeSet)//2)+(subTest*70)]
	else:
		depthTrainSet=storeSet[0:(subTest-1)*70]+storeSet[subTest*70:(len(storeSet)//2)+((subTest-1)*70)]+storeSet[(len(storeSet)//2)+subTest*70:len(storeSet)]
		depthTestSet=storeSet[(subTest-1)*70:subTest*70]+storeSet[(len(storeSet)//2)+((subTest-1)*70):(len(storeSet)//2)+(subTest*70)]
		depthTrainLbl=storeLbl[0:(subTest-1)*70]+storeLbl[subTest*70:(len(storeSet)//2)+((subTest-1)*70)]+storeLbl[(len(storeSet)//2)+subTest*70:len(storeSet)]
		depthTestLbl=storeLbl[(subTest-1)*70:subTest*70]+storeLbl[(len(storeSet)//2)+((subTest-1)*70):(len(storeSet)//2)+(subTest*70)]

	depthTestSet=np.array(depthTestSet)
	depthTrainSet=np.array(depthTrainSet)

	depthTestSv=depthTestLbl

    #convert to one-hot-encoded labels
	depthTrainLbl=to_categorical(depthTrainLbl, num_classes=None)
	depthTestLbl=to_categorical(depthTestLbl, num_classes=None)

    #enable pre-training based on cnn
	if precnn==1:
		cnnTrainLbl=to_categorical(cnnTrainLbl, num_classes=None)
		cnnTestLbl=to_categorical(cnnTestLbl, num_classes=None)
		cnnmodel=Sequential()
		cnnmodel.add(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='relu',input_shape=(227,227,1)))
		cnnmodel.add(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='relu'))
		cnnmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		cnnmodel.add(Dropout(0.20))

		cnnmodel.add(Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu'))
		cnnmodel.add(Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu'))
		cnnmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		cnnmodel.add(Dropout(0.20))

		cnnmodel.add(Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu'))
		cnnmodel.add(Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu'))
		cnnmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		cnnmodel.add(Dropout(0.20))
		cnnmodel.add(Flatten())

		cnnmodel.add(Dense(256, activation='relu'))
		cnnmodel.add(Dropout(0.50))
		cnnmodel.add(Dense(512, activation='relu'))
		cnnmodel.add(Dropout(0.50))
		cnnmodel.add(Dense(256, activation='relu'))
		cnnmodel.add(Dropout(0.50))
		cnnmodel.add(Dense(num_classes, activation='softmax'))

		cnnmodel.compile(loss='categorical_crossentropy',
				  optimizer='adadelta',
				  metrics=['accuracy'])
		cnnmodel.summary()

		filepath="model\\sv3\\loo" + str(subTest) + "-best.hdf5"

		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
		early=EarlyStopping(monitor='val_acc', patience=5,verbose=0,mode='auto')
		callbacks_list=[early,checkpoint]

		cnnTestSet=np.array(cnnTestSet)
		cnnTrainSet=np.array(cnnTrainSet)

		cnnmodel.fit(cnnTrainSet, cnnTrainLbl, batch_size=bs*2, epochs=100, shuffle=True, validation_data=(cnnTestSet, cnnTestLbl), verbose=1, callbacks=callbacks_list)
		cnnmodel.load_weights(filepath)
		scores=cnnmodel.evaluate(cnnTestSet,cnnTestLbl, batch_size=bs)
		f=open('acc.txt','a')
		f.write("LOOCV: %2.0f, BatchSize: %2.0f, Accuracy: %.2f%%\n" % (subTest, 2*bs, scores[1]*100))
		f.close()
		del cnnTestSet
		del cnnTrainSet
		del cnnTestLbl
		del cnnTrainLbl


	for timesteps in range(5,6):
		# expected input data shape: (batch_size, timesteps, data_dim)
		model = Sequential()

		model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='relu'),input_shape=(2**timesteps, 227,227,1)))
		model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1,1), padding='same', activation='relu')))
		model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))
		model.add(Dropout(0.20))

		model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu')))
		model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu')))
		model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))
		model.add(Dropout(0.20))

		model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')))
		model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')))
		model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))
		model.add(Dropout(0.20))

		model.add(TimeDistributed(Flatten()))
		model.add(Dropout(0.50))
		model.add(LSTM(256,return_sequences=True))
		model.add(Dropout(0.50))
		model.add(LSTM(256))
		model.add(Dropout(0.50))

		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.50))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.50))
		model.add(Dense(num_classes, activation='softmax'))
		model.summary()

        #load pre-trained cnn weights into CNN+LSTM model
		if precnn==1:
			for index in range(0,12):
				w=cnnmodel.layers[index].get_weights()
				model.layers[index].set_weights(w)
			del cnnmodel

		model.compile(loss='categorical_crossentropy',
					  optimizer='adadelta',
					  metrics=['accuracy'])

		filepath="model\\sv3\\loo" + str(subTest) + 'ts' + str(2**timesteps)  + "-weights-improvement-{epoch:04d}-{val_acc:.5f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
		early=EarlyStopping(monitor='val_acc', patience=10,verbose=0,mode='auto')
		log=CSVLogger('model\\sv3\\loo' + str(subTest) + 'ts' + str(2**timesteps)  + 'training.log')
		callbacks_list=[early,checkpoint,log]

		model.fit(depthTrainSet, depthTrainLbl, batch_size=bs, epochs=100, shuffle=False, validation_data=(depthTestSet, depthTestLbl), verbose=1, callbacks=callbacks_list)
		model.load_weights(filepath)
		scores=model.evaluate(depthTestSet,depthTestLbl, batch_size=bs)

		pred=model.predict_classes(depthTestSet)
		s='predictions'+ str(subTest) +'.txt'
		f=open(s,'w')
		for p in range(0,len(pred)):
			f.write("%d %d\n" % (pred[p], depthTestSv[p]))
		f.close()

		modelP.append(pred)
		trueP.append(depthTestSv)
		f=open('acc.txt','a')
		f.write("TS: %3.0f, LOOCV: %2.0f, BatchSize: %2.0f, Accuracy: %.2f%%\n" % (timesteps,subTest, bs, scores[1]*100))
		f.close()
		print("Accuracy: %.2f%%" % (scores[1]*100))

		del model
		K.clear_session()
