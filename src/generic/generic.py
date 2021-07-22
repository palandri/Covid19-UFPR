def auto_gpu_selection(usage_max=0.01, mem_max=0.05):
    import os
    import subprocess
    """Auto set CUDA_VISIBLE_DEVICES for gpu

    COPIED FROM https://stackoverflow.com/questions/42219848/tensorflow-automatically-choose-least-loaded-gpu

    :param mem_max: max percentage of GPU utility
    :param usage_max: max percentage of GPU memory
    :return:
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    gpu = 0

    # Maximum of GPUS, 8 is enough for most
    for i in range(8):
        idx = i*4 + 3
        if idx > log.__len__()-1:
            break
        inf = log[idx].split("|")
        if inf.__len__() < 3:
            break
        usage = int(inf[3].split("%")[0].strip())
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
        # print("GPU-%d : Usage:[%d%%]" % (gpu, usage))
        if usage < 100*usage_max and mem_now < mem_max*mem_all:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("\nAuto choosing vacant GPU-%d : Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]\n" %
                  (gpu, mem_now, mem_all, usage))
            return
        print("GPU-%d is busy: Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]" %
              (gpu, mem_now, mem_all, usage))
        gpu += 1
    print("\nNo vacant GPU, use CPU instead\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import os
if os.name == 'nt':
	print(f"Importing DLLs")
	os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
	os.add_dll_directory("C:/cuda/bin")

import pickle as pkl
import h5py
from src import densenetcrossvalidation as dense, vggcrossvalidation as vgg, efficientnetcrossvalidation as eff, mobilenetcrossvalidation as mobile, resnetcrossvalidation as res
import itertools
import cv2
from imutils import paths
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import os


def save_to_pkl(data, labels, lb, data_path):
    with open(f'{data_path}data.pkl', 'wb') as f:
        pkl.dump(data, f)
        
    with open(f'{data_path}labels.pkl', 'wb') as f:
        pkl.dump(labels, f)
        
    with open(f'{data_path}lb.pkl', 'wb') as f:
        pkl.dump(lb, f)


def load_from_pkl(data_path):
    with open(f'{data_path}data.pkl', 'rb') as f:
        data = pkl.load(f)
    with open(f'{data_path}labels.pkl', 'rb') as f:
        labels = pkl.load(f)
    with open(f'{data_path}lb.pkl', 'rb') as f:
        lb = pkl.load(f)
        
    return data, labels, lb

def process_dense(results_path, unique_id, training_x, training_y, validation_x, validation_y, data, labels, lb, epochs, folds):
    network_name = 'dense'
    model_path = f"{results_path}saved_models/{network_name}_best_{unique_id}.h5"
    print(f"Starting {network_name}_net processing...")
    start_time = time.time()
    dense16_model = dense.generate_network()
    dense_history, dense_training_time = train(   dense16_model,
                                                  model_path,
                                                  training_x,
                                                  training_y,
                                                  data, labels,
                                                  n_epochs=epochs,
                                                  kfolds=folds)
    generate_training_graphs(dense_history, results_path, network=network_name, unique_id=unique_id)
    dense_report, dense_inference_time = predict(dense16_model,
                                                 model_path,
                                                 validation_x,
                                                 validation_y,
                                                 network=network_name,
                                                 results_path=results_path,
                                                 lb=lb,
                                                 unique_id=unique_id)
    save_report(results_path, unique_id, dense_report, dense_training_time, dense_inference_time, network_name)
    print(f"Finished {network_name}_net processing. Time elapsed: {round(time.time() - start_time, 3)}s")

def process_eff(results_path, unique_id, training_x, training_y, validation_x, validation_y, data, labels, lb, epochs, folds):
    network_name = 'eff'
    model_path = f"{results_path}saved_models/{network_name}_best_{unique_id}.h5"
    print(f"Starting {network_name}_net processing...")
    start_time = time.time()
    eff16_model = eff.generate_network()
    eff_history, eff_training_time = train(   eff16_model,
                                                  model_path,
                                                  training_x,
                                                  training_y,
                                                  data, labels,
                                                  n_epochs=epochs,
                                                  kfolds=folds)
    generate_training_graphs(eff_history, results_path, network=network_name, unique_id=unique_id)
    eff_report, eff_inference_time = predict(eff16_model,
                                                 model_path,
                                                 validation_x,
                                                 validation_y,
                                                 network=network_name,
                                                 results_path=results_path,
                                                 lb=lb,
                                                 unique_id=unique_id)
    save_report(results_path, unique_id, eff_report, eff_training_time, eff_inference_time, network_name)
    print(f"Finished {network_name}_net processing. Time elapsed: {round(time.time() - start_time, 3)}s")

def process_mobile(results_path, unique_id, training_x, training_y, validation_x, validation_y, data, labels, lb, epochs, folds):
    network_name = 'mobile'
    model_path = f"{results_path}saved_models/{network_name}_best_{unique_id}.h5"
    print(f"Starting {network_name}_net processing...")
    start_time = time.time()
    mobile16_model = mobile.generate_network()
    mobile_history, mobile_training_time = train(   mobile16_model,
                                                  model_path,
                                                  training_x,
                                                  training_y,
                                                  data, labels,
                                                  n_epochs=epochs,
                                                  kfolds=folds)
    generate_training_graphs(mobile_history, results_path, network=network_name, unique_id=unique_id)
    mobile_report, mobile_inference_time = predict(mobile16_model,
                                                 model_path,
                                                 validation_x,
                                                 validation_y,
                                                 network=network_name,
                                                 results_path=results_path,
                                                 lb=lb,
                                                 unique_id=unique_id)
    save_report(results_path, unique_id, mobile_report, mobile_training_time, mobile_inference_time, network_name)
    print(f"Finished {network_name}_net processing. Time elapsed: {round(time.time() - start_time, 3)}s")

def process_vgg(results_path, unique_id, training_x, training_y, validation_x, validation_y, data, labels, lb, epochs, folds):
    network_name = 'vgg'
    model_path = f"{results_path}saved_models/{network_name}_best_{unique_id}.h5"
    print(f"Starting {network_name}_net processing...")
    start_time = time.time()
    vgg16_model = vgg.generate_network()
    vgg_history, vgg_training_time = train(   vgg16_model,
                                                  model_path,
                                                  training_x,
                                                  training_y,
                                                  data, labels,
                                                  n_epochs=epochs,
                                                  kfolds=folds)
    generate_training_graphs(vgg_history, results_path, network=network_name, unique_id=unique_id)
    vgg_report, vgg_inference_time = predict(vgg16_model,
                                                 model_path,
                                                 validation_x,
                                                 validation_y,
                                                 network=network_name,
                                                 results_path=results_path,
                                                 lb=lb,
                                                 unique_id=unique_id)
    save_report(results_path, unique_id, vgg_report, vgg_training_time, vgg_inference_time, network_name)
    print(f"Finished {network_name}_net processing. Time elapsed: {round(time.time() - start_time, 3)}s")

def process_res(results_path, unique_id, training_x, training_y, validation_x, validation_y, data, labels, lb, epochs, folds):
    network_name = 'res'
    model_path = f"{results_path}saved_models/{network_name}_best_{unique_id}.h5"
    print(f"Starting {network_name}_net processing...")
    start_time = time.time()
    res16_model = res.generate_network()
    res_history, res_training_time = train(   res16_model,
                                                  model_path,
                                                  training_x,
                                                  training_y,
                                                  data, labels,
                                                  n_epochs=epochs,
                                                  kfolds=folds)
    generate_training_graphs(res_history, results_path, network=network_name, unique_id=unique_id)
    res_report, res_inference_time = predict(res16_model,
                                                 model_path,
                                                 validation_x,
                                                 validation_y,
                                                 network=network_name,
                                                 results_path=results_path,
                                                 lb=lb,
                                                 unique_id=unique_id)
    save_report(results_path, unique_id, res_report, res_training_time, res_inference_time, network_name)
    print(f"Finished {network_name}_net processing. Time elapsed: {round(time.time() - start_time, 3)}s")


def generate_training_graphs(model, results_path, network, unique_id="no_id"):
	# plot the training loss and accuracy
	N = len(model.history['loss'])
	# plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), model.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), model.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), model.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), model.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on COVID-19 Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(f"{results_path}graphs/{network}_cross_{unique_id}.png")


def predict(model, model_path, validX, validY, lb, network, results_path, unique_id="no_id"):
	inference_time_start = time.time()
	# make predictions on the validation set
	print("[INFO] evaluating network...")
	model.load_weights(model_path)
	predIdxs = model.predict(np.array(validX), batch_size=8)
	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	predIdxs = np.argmax(predIdxs, axis=1)
	# show a nicely formatted classification report
	cm = confusion_matrix(validY.argmax(axis=1), predIdxs)
	cm_plot_labels = ['COVID-19', 'NORMAL']
	plot_confusion_matrix(cm=cm, classes=cm_plot_labels, process_id=unique_id, title='Matriz de Confusao',
	                      results_path=results_path, network=network)
	report = classification_report(np.array(validY).argmax(axis=1), predIdxs, target_names=lb.classes_)
	return report, time.time() - inference_time_start


def train(model,
          model_path,
          trainX,
          trainY,
          data,
          labels,
          n_epochs=500,
          kfolds=5):

	import gc
	total_training_time_start = time.time()
	folds = KFold(n_splits=kfolds, shuffle=True, random_state=1)
	monitor = 'val_loss'
	count = 0

	for iteration, (train_idx, test_idx) in enumerate(folds.split(trainX)):
		print(f"Fold: {iteration + 1}/{kfolds}")
		# input_sub_data = [data[index] for index in train_idx]
		# input_sub_labels = [labels[index] for index in train_idx]

		sub_trainX = [data[index] for index in train_idx]
		sub_trainY = [labels[index] for index in train_idx]
		sub_testX = [data[index] for index in test_idx]
		sub_testY = [labels[index] for index in test_idx]

		trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
		validAug = ImageDataGenerator()

		# compile our model
		print("[INFO] compiling model...")
		opt = Adam(learning_rate=0.001, decay=0.001 / 10)
		model.compile(loss="binary_crossentropy", optimizer=opt,
		              metrics=["accuracy"])
		# train the head of the network
		print("[INFO] training head...")

		count += 1
		patience=20
		min_delta = 0.01

		# earlystopping_callback = EarlyStopping(monitor='loss',
		#                                       patience=patience,
		#                                       min_delta=min_delta)

		checkpoint_callback = ModelCheckpoint(filepath=model_path,
		                                      save_best_only=True,
		                                      monitor=monitor,
		                                      save_weights_only=False)

		# callbacks = [earlystopping_callback, checkpoint_callback]
		callbacks = [checkpoint_callback]

		start_time = time.time()
		batch_size = 4

		print(f"Training size: {len(trainX)},{len(trainY)}")

		H = model.fit(
			trainAug.flow(np.array(sub_trainX), np.array(sub_trainY), batch_size=batch_size),
			steps_per_epoch=len(sub_trainX) // batch_size,
			validation_data=validAug.flow(np.array(sub_testX), np.array(sub_testY), batch_size=batch_size),
			validation_steps=len(sub_testX) // batch_size,
			epochs=n_epochs,
			callbacks=callbacks)

		print(f"Training time, fold {iteration + 1} --- {time.time() - start_time} seconds ---")
		gc.collect()
	return H, time.time() - total_training_time_start


def plot_confusion_matrix(  cm,
							classes,
							process_id,
							results_path,
                            network,
							normalize=False,
							title='Confusion matrix',
							cmap=plt.cm.Blues
							):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
		         horizontalalignment="center",
		         color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('')
	plt.xlabel('')
	plt.savefig(f"{results_path}graphs/{network}_cross_{process_id}_cm.png")


def save_report(results_path, unique_id, report, training_time, inference_time, network_name):
	print(report)
	with open(f"{results_path}reports/{network_name}_{unique_id}.txt", 'w') as f:
		f.write(f"Execution ID: {unique_id}\n")
		f.write(f"Total training time: {round(training_time,3)}s\n")
		f.write(f"Inference time: {round(inference_time,3)}s")
		f.write(report)
		f.write('\n')


def load_images(root_path):
	print(f"Loading images")
	start_time = time.time()
	imagePaths = list(paths.list_images(root_path))
	# print(imagePaths)
	# label = imagePaths[0].split(os.path.sep)[-2]
	# print(label)
	data = []
	labels = []
	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]
		# load the image, swap color channels, and resize it to be a fixed
		# 224x224 pixels while ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))
		# update the data and labels lists, respectively
		data.append(image)
		labels.append(label)
	# convert the data and labels to NumPy arrays while scaling the pixel
	# intensities to the range [0, 1]
	data = np.array(data) / 255.0
	labels = np.array(labels)

	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	labels = to_categorical(labels)
	print(f"Finished loading images, time: {time.time() - start_time}s")

	return data, lb, labels
