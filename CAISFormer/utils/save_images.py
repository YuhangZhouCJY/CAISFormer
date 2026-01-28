'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def save_images(saved_all, epoch, folder, resize_to=None):
	cover, stegan_image, secrets, re_secret = saved_all
	cover = cover[:cover.shape[0], :, :, :].cpu()
	stegan_image = stegan_image[:stegan_image.shape[0], :, :, :].cpu()
	secrets = secrets[:secrets.shape[0], :, :, :].cpu()
	resize = nn.UpsamplingNearest2d(size=(cover.shape[2], cover.shape[3]))
	re_secret = resize(re_secret)

	if resize_to is not None:
		cover = F.interpolate(cover, size=resize_to)
		stegan_image = F.interpolate(stegan_image, size=resize_to)
		secrets = F.interpolate(secrets, size=resize_to)
	diff_images = (stegan_image - cover + 1) / 2

	# transform to rgb
	diff_images_linear = diff_images.clone()
	R = diff_images_linear[:, 0, :, :]
	G = diff_images_linear[:, 1, :, :]
	B = diff_images_linear[:, 2, :, :]
	diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
	diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

	# maximize diff in every image
	for id in range(diff_images_linear.shape[0]):
		diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
				diff_images_linear[id].max() - diff_images_linear[id].min())

	stacked_images = torch.cat(
		[cover.unsqueeze(0), stegan_image.unsqueeze(0), secrets.unsqueeze(0), re_secret.unsqueeze(0),
		 diff_images_linear.unsqueeze(0)], dim=0)
	shape = stacked_images.shape
	stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
	stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

	saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
	saved_image.save(filename)


def get_random_images(cover, stegan_image, secrets, re_secret):
	selected_id = np.random.randint(1, cover.shape[0]) if cover.shape[0] > 1 else 1
	cover = cover.cpu()[selected_id - 1:selected_id, :, :, :]
	stegan_image = stegan_image.cpu()[selected_id - 1:selected_id, :, :, :]
	secrets = secrets.cpu()[selected_id - 1:selected_id, :, :, :]
	re_secret = re_secret.cpu()[selected_id - 1:selected_id, :, :, :]
	return [cover, stegan_image, secrets, re_secret]


def concatenate_images(saved_all, cover, stegan_image, secrets, re_secret):
	saved = get_random_images(cover, stegan_image, secrets, re_secret)
	if saved_all[3].shape[3] != saved[3].shape[3]:
		return saved_all
	saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
	saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
	saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
	saved_all[3] = torch.cat((saved_all[3], saved[3]), 0)
	return saved_all


def save_stego_images(stegan_images, covers, num, folder, resize_to=None):
	stegan_image = stegan_images
	cover = covers

	if resize_to is not None:
		stegan_image = F.interpolate(stegan_image, size=resize_to)
		cover = F.interpolate(cover, size=resize_to)

	stego_filename = os.path.join(folder, 'stego-{}.png'.format(num))
	cover_filename = os.path.join(folder, 'cover-{}.png'.format(num))

	stegan_image.squeeze_(0)
	cover.squeeze_(0)

	stegan_image = stegan_image.permute(2, 1, 0)
	stegan_image = stegan_image.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	saved_stego_image = Image.fromarray(np.array(stegan_image, dtype=np.uint8)).convert("RGB")

	cover = cover.permute(2, 1, 0)
	cover = cover.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	saved_cover_image = Image.fromarray(np.array(cover, dtype=np.uint8)).convert("RGB")


	saved_stego_image.save(stego_filename)
	saved_cover_image.save(cover_filename)



