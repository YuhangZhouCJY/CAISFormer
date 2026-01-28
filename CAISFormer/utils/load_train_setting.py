from .settings import *
import os
import time

'''
params setting
'''
settings = JsonConfig()
settings.load_json_file("train_settings.json")

project_name = settings.project_name
dataset_path = settings.dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
save_images_number = settings.save_images_number
lr = settings.lr
H, W = settings.H, settings.W


'''
file preparing
'''
full_project_name = project_name + "_" + str(H)

result_folder = "results/" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"
if not os.path.exists(result_folder): os.mkdir(result_folder)
if not os.path.exists(result_folder + "train_images/"): os.mkdir(result_folder + "train_images/")
if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
if not os.path.exists(result_folder + "test_images/"): os.mkdir(result_folder + "test_images/")
if not os.path.exists(result_folder + "test_images/cover/"): os.mkdir(result_folder + "test_images/cover/")
if not os.path.exists(result_folder + "test_images/stego/"): os.mkdir(result_folder + "test_images/stego/")
if not os.path.exists(result_folder + "test_images/secret/"): os.mkdir(result_folder + "test_images/secret/")
if not os.path.exists(result_folder + "test_images/rsecret/"): os.mkdir(result_folder + "test_images/rsecret/")
if not os.path.exists(result_folder + "consequence.txt/"): os.mkdir(result_folder + "consequence.txt/")

with open(result_folder + "/train_params.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"

	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"

	print(content)
	file.write(content)
with open(result_folder + "/train_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
with open(result_folder + "/val_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
