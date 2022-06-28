"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch, ncolor):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(ncolor+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)


class spatialDnCNN(nn.Module):
	""" Definition of the spatial DnCNN denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=1, num_color_channels=3):
		super(spatialDnCNN, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0, ncolor=num_color_channels)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=num_color_channels)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in1, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in1, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3, num_color_channels=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0, ncolor=num_color_channels)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=num_color_channels)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x
from .builder import MODELS		

@MODELS.register_module
class FastDVDnet(nn.Module):

	def __init__(self, num_input_frames=5, num_color_channels=3):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		# self.spatial = spatialDnCNN(num_input_frames=1, num_color_channels=num_color_channels)
		self.temp1 = DenBlock(num_input_frames=3, num_color_channels=num_color_channels)
		self.temp2 = DenBlock(num_input_frames=3, num_color_channels=num_color_channels)
		# Init weights
		self.reset_params()
		self.num_color_channels = num_color_channels

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_sigma):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		frames,channel,src_height,src_width = x.shape
		if src_height%2 != 0 or src_width%2!=0:
			pad = (0,src_width%2,0,src_height%2)
			x = F.pad(x,pad)
		frames,channel,height,width = x.shape
		batch_size=1
		noise_map = noise_sigma.view(frames, 1, 1, 1).repeat(1, 1, height, width)
		input = torch.zeros(batch_size,frames*self.num_input_frames*channel,height,width).to(x.device)
		for frameidx in range(frames):
			idx = (torch.tensor(range(frameidx, frameidx+self.num_input_frames)) - self.num_input_frames//2) % frames# circular padding
			noisy_seq = x[idx].reshape((-1,height,width))
			# out = denoiser(noisy_seq,sigma)
			input[:,frameidx*self.num_input_frames*channel:(frameidx+1)*self.num_input_frames*channel] = noisy_seq
		x = input.view(frames,self.num_input_frames*channel,height,width)
		C = self.num_color_channels # number of color channels
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, m*C:m*C+C, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)
		x = x[:,:,:src_height,:src_width]
		return x
	def __nchannel__(self):
		return self.num_color_channels