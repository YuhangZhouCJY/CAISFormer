from stegan_block.encoder import Encoder as Encoder
from stegan_block.decoder import Decoder as Decoder
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
	def __init__(self):
		super(EncoderDecoder, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, cover, secret):
		stegan_image = self.encoder(cover, secret)
		re_secret = self.decoder(stegan_image)
		return stegan_image, re_secret
