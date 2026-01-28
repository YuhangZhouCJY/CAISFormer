from .settings import *
import os
import time

'''
params setting
'''
filename = "test_settings.json"
settings = JsonConfig()
settings.load_json_file(filename)

dataset_path = settings.dataset_path
batch_size = 2
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
save_images_number = settings.save_images_number
lr = 1e-3
H, W = settings.H, settings.W

result_folder = "results/" + settings.result_folder
test_base = "/test_"

test_param = result_folder + test_base + "s{}_params.json".format(strength_factor)
test_log = result_folder + test_base + "s{}_log.txt".format(strength_factor)
with open(test_param, "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)

	with open(filename, "r") as setting_file:
		content = setting_file.read()
		file.write(content)

with open(test_log, "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
