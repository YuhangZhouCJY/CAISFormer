import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils import *
from stegan_block.Network import *

from utils.load_train_setting import *
import numpy as np


'''
train
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(device, batch_size, lr)

train_dataset = SteganDataset(os.path.join(dataset_path, "train/"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

val_dataset = SteganDataset(os.path.join(dataset_path, "val/"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

if train_continue:
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

for epoch in range(epoch_number):

	epoch += train_continue_epoch if train_continue else 0

	running_result = {
		"error_rate": 0.0,
		"psnr_cover": 0.0,
		"ssim_cover": 0.0,
		"psnr_secret": 0.0,
		"ssim_secret": 0.0,
		"MAE_cover": 0.0,
		"MAE_secret": 0.0,
		"RMSE_cover": 0.0,
		"RMSE_secret": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0
	}

	start_time = time.time()

	'''
	train
	'''
	num = 0
	for _, images in enumerate(train_dataloader):
		dataload = images.to(device)
		cover = dataload[dataload.shape[0] // 2:]
		secret = dataload[:dataload.shape[0] // 2]

		result = network.train(cover, secret)

		for key in result:
			running_result[key] += float(result[key])

		num += 1

	'''
	train results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in running_result:
		content += key + "=" + str(running_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/train_log.txt", "a") as file:
		file.write(content)
	print(content)

	'''
	validation
	'''

	val_result = {
		"error_rate": 0.0,
		"psnr_cover": 0.0,
		"ssim_cover": 0.0,
		"psnr_secret": 0.0,
		"ssim_secret": 0.0,
		"MAE_cover": 0.0,
		"MAE_secret": 0.0,
		"RMSE_cover": 0.0,
		"RMSE_secret": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0
	}

	start_time = time.time()

	saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
	saved_all = None


	num = 0
	for i, images in enumerate(val_dataloader):
		dataload = images.to(device)
		cover = dataload[dataload.shape[0] // 2:]
		secret = dataload[:dataload.shape[0] // 2]

		result, (covers, stegan_image, secrets, re_secret) = network.validation(cover, secret)

		for key in result:
			val_result[key] += float(result[key])

		num += 1

		if i in saved_iterations:
			if saved_all is None:
				saved_all = get_random_images(cover, stegan_image, secrets, re_secret)
				save_images(saved_all, epoch, result_folder + "train_images/", resize_to=(W, H))
			else:
				saved_all = concatenate_images(saved_all, cover, stegan_image, secrets, re_secret)
				save_images(saved_all, epoch, result_folder + "train_images/", resize_to=(W, H))

	'''
	validation results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in val_result:
		content += key + "=" + str(val_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/val_log.txt", "a") as file:
		file.write(content)
	print(content)

	'''
	save model
	'''
	path_model = result_folder + "models/"
	path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
	path_discriminator = path_model + "D_" + str(epoch) + ".pth"
	network.save_model(path_encoder_decoder, path_discriminator)
