from stegan_block.encoder_decoder import *
from stegan_block.Discriminator import Discriminator
import kornia.losses
import numpy as np
from sklearn import metrics


class Network:
	def __init__(self, device, batch_size, lr):
		# device
		self.device = device

		self.encoder_decoder = EncoderDecoder()
		self.discriminator = Discriminator().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)

		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

		self.opt_encoder_decoder = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
		self.criterion_MSE = nn.MSELoss().to(device)

		self.discriminator_weight = 0.0001
		self.encoder_weight = 4
		self.decoder_weight = 10
		self.encoder_ssim_weight = 0.02
		self.decoder_ssim_weight = 0.01


	def train(self, covers: torch.Tensor, secrets: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute
			covers, secrets = covers.to(self.device), secrets.to(self.device)
			stegan_image, re_secret = self.encoder_decoder(covers, secrets)

			'''
			train discriminator
			'''
			self.opt_discriminator.zero_grad()

			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(covers)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(stegan_image.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(stegan_image)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to covers image
			g_loss_on_encoder = self.criterion_MSE(stegan_image, covers)
			g_ssim_loss_on_encoder = 2 * kornia.losses.ssim_loss(stegan_image, covers, window_size=5, reduction="mean")

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(re_secret, secrets)
			g_ssim_loss_on_decoder = 2 * kornia.losses.ssim_loss(re_secret, secrets, window_size=5, reduction="mean")

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder + \
					 self.encoder_ssim_weight * g_ssim_loss_on_encoder + self.decoder_ssim_weight * g_ssim_loss_on_decoder

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr_cover = -kornia.losses.psnr_loss(stegan_image.detach(), covers, 1)
			psnr_secret = -kornia.losses.psnr_loss(re_secret.detach(), secrets, 1)

			# ssim
			ssim_cover = 1 - 2 * kornia.losses.ssim_loss(stegan_image.detach(), covers, window_size=5, reduction="mean")
			ssim_secret = 1 - 2 * kornia.losses.ssim_loss(re_secret.detach(), secrets, window_size=5, reduction="mean")

			# MAE
			MAE_cover = torch.mean(torch.mean(torch.abs((stegan_image.detach()) * 255.0 - covers * 255.0)))
			MAE_secret = torch.mean(torch.mean(torch.abs((re_secret.detach()) * 255.0 - secrets * 255.0)))

			# RMSE
			RMSE_cover = torch.mean(torch.sqrt(torch.mean(torch.square((stegan_image.detach()) * 255.0 - covers * 255.0))))
			RMSE_secret = torch.mean(torch.sqrt(torch.mean(torch.square((re_secret.detach()) * 255.0 - secrets * 255.0))))

		'''
		decoded message error rate
		'''
		# error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)
		re_secret_rounded = re_secret.detach().cpu().numpy()
		secrets_rounded = secrets.detach().cpu().numpy()
		error_rate = np.sum(np.abs(re_secret_rounded - secrets_rounded)) / (
				secrets.shape[0] * secrets.shape[1] * secrets.shape[2] * secrets.shape[3])

		result = {
			"error_rate": error_rate,
			"psnr_cover": psnr_cover,
			"ssim_cover": ssim_cover,
			"psnr_secret": psnr_secret,
			"ssim_secret": ssim_secret,
			"MAE_cover": MAE_cover,
			"MAE_secret": MAE_secret,
			"RMSE_cover": RMSE_cover,
			"RMSE_secret": RMSE_secret,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result

	def validation(self, covers: torch.Tensor, secrets: torch.Tensor):
		self.encoder_decoder.eval()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			covers, secrets = covers.to(self.device), secrets.to(self.device)
			stegan_image, re_secret = self.encoder_decoder(covers, secrets)

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(covers)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(stegan_image.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(stegan_image)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to covers image
			g_loss_on_encoder = self.criterion_MSE(stegan_image, covers)
			g_ssim_loss_on_encoder = 2 * kornia.losses.ssim_loss(stegan_image, covers, window_size=5, reduction="mean")

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(re_secret, secrets)
			g_ssim_loss_on_decoder = 2 * kornia.losses.ssim_loss(re_secret, secrets, window_size=5, reduction="mean")

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder + \
					 self.encoder_ssim_weight * g_ssim_loss_on_encoder + self.decoder_ssim_weight * g_ssim_loss_on_decoder


			# psnr
			psnr_cover = -kornia.losses.psnr_loss(stegan_image.detach(), covers, 1)
			psnr_secret = -kornia.losses.psnr_loss(re_secret.detach(), secrets, 1)

			# ssim
			ssim_cover = 1 - 2 * kornia.losses.ssim_loss(stegan_image.detach(), covers, window_size=5,
														 reduction="mean")
			ssim_secret = 1 - 2 * kornia.losses.ssim_loss(re_secret.detach(), secrets, window_size=5,
														  reduction="mean")

			# MAE
			MAE_cover = torch.mean(torch.mean(torch.abs((stegan_image.detach()) * 255.0 - covers * 255.0)))
			MAE_secret = torch.mean(torch.mean(torch.abs((re_secret.detach()) * 255.0 - secrets * 255.0)))

			# RMSE
			RMSE_cover = torch.mean(torch.sqrt(torch.mean(torch.square((stegan_image.detach()) * 255.0 - covers * 255.0))))
			RMSE_secret = torch.mean(torch.sqrt(torch.mean(torch.square((re_secret.detach()) * 255.0 - secrets * 255.0))))
		'''
		decoded message error rate
		'''
		# error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)
		re_secret_rounded = re_secret.detach().cpu().numpy()
		secrets_rounded = secrets.detach().cpu().numpy()
		error_rate = np.sum(np.abs(re_secret_rounded - secrets_rounded)) / (
				secrets.shape[0] * secrets.shape[1] * secrets.shape[2] * secrets.shape[3])

		result = {
			"error_rate": error_rate,
			"psnr_cover": psnr_cover,
			"ssim_cover": ssim_cover,
			"psnr_secret": psnr_secret,
			"ssim_secret": ssim_secret,
			"MAE_cover": MAE_cover,
			"MAE_secret": MAE_secret,
			"RMSE_cover": RMSE_cover,
			"RMSE_secret": RMSE_secret,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		return result, (covers, stegan_image, secrets, re_secret)

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		error_rate /= batch_size
		return error_rate

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))
