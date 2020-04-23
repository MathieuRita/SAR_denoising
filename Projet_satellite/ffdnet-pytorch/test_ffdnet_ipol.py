"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_ffdnet(**args):
	r"""Denoises an input image with FFDNet
	"""
	# Init logger
	logger = init_logger_ipol()

	# Check if input exists and if it is RGB
	try:
		rgb_den = is_rgb(args['input'])
	except:
		raise Exception('Could not open the input image')

	# Open image as a CxHxW torch.Tensor
	if rgb_den:
		in_ch = 3
		model_fn = 'models/net_rgb.pth'
		imorig = cv2.imread(args['input'])
		# from HxWxC to CxHxW, RGB image
		imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		in_ch = 1
		model_fn = 'models/net_gray.pth'
		imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
		imorig = np.expand_dims(imorig, 0)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	if sh_im[2]%2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3]%2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = normalize(imorig)
	imorig = torch.Tensor(imorig)

	# Absolute path to model file
	model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
				model_fn)

	# Create model
	print('Loading model ...\n')
	net = FFDNet(num_input_channels=in_ch)

	# Load saved weights
	if args['cuda']:
		state_dict = torch.load(model_fn)
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
	else:
		state_dict = torch.load(model_fn, map_location='cpu')
		# CPU mode: remove the DataParallel wrapper
		state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	model.load_state_dict(state_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# Add noise
	if args['add_noise']:
		noise = torch.FloatTensor(imorig.size()).\
				normal_(mean=0, std=args['noise_sigma'])
		imnoisy = imorig + noise
	else:
		imnoisy = imorig.clone()

        # Test mode
	with torch.no_grad(): # PyTorch v0.4.0
	    imorig, imnoisy = Variable(imorig.type(dtype)), \
	    				Variable(imnoisy.type(dtype))
	    nsigma = Variable(
	    		torch.FloatTensor([args['noise_sigma']]).type(dtype))

	# Measure runtime
	start_t = time.time()

	# Estimate noise and subtract it to the input image
	im_noise_estim = model(imnoisy, nsigma)
	outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
	stop_t = time.time()

	if expanded_h:
		imorig = imorig[:, :, :-1, :]
		outim = outim[:, :, :-1, :]
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imorig = imorig[:, :, :, :-1]
		outim = outim[:, :, :, :-1]
		imnoisy = imnoisy[:, :, :, :-1]

	# Compute PSNR and log it
	if rgb_den:
		logger.info("### RGB denoising ###")
	else:
		logger.info("### Grayscale denoising ###")
	if args['add_noise']:
		psnr = batch_psnr(outim, imorig, 1.)
		psnr_noisy = batch_psnr(imnoisy, imorig, 1.)

		logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
		logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
	else:
		logger.info("\tNo noise was added, cannot compute PSNR")
	logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))

	# Compute difference
	diffout   = 2*(outim - imorig) + .5
	diffnoise = 2*(imnoisy-imorig) + .5

	# Save images
	if not args['dont_save_results']:
		noisyimg = variable_to_cv2_image(imnoisy)
		outimg = variable_to_cv2_image(outim)
		cv2.imwrite("noisy.png", noisyimg)
		cv2.imwrite("ffdnet.png", outimg)
		if args['add_noise']:
 			cv2.imwrite("noisy_diff.png", variable_to_cv2_image(diffnoise))
 			cv2.imwrite("ffdnet_diff.png", variable_to_cv2_image(diffout))

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="FFDNet_Test")
	parser.add_argument('--add_noise', type=str, default="True")
	parser.add_argument("--input", type=str, default="", \
						help='path to input image')
	parser.add_argument("--suffix", type=str, default="", \
						help='suffix to add to output name')
	parser.add_argument("--noise_sigma", type=float, default=25, \
						help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', \
						help="don't save output images")
	parser.add_argument("--no_gpu", action='store_true', \
						help="run model on CPU")
	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# String to bool
	argspar.add_noise = (argspar.add_noise.lower() == 'true')

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_ffdnet(**vars(argspar))
