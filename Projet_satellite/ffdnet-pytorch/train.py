"""
Trains a FFDNet model

By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
described in the FFDNet paper is performed (--no_orthog to set it off).

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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from models import FFDNet
from dataset import Dataset
from utils import weights_init_kaiming, batch_psnr, init_logger, \
			svd_orthogonalization

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
	r"""Performs the main training loop
	"""
	# Load dataset
	print('> Loading dataset ...')
	dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
	dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
	loader_train = DataLoader(dataset=dataset_train, num_workers=6, \
							   batch_size=args.batch_size, shuffle=True)
	print("\t# of training samples: %d\n" % int(len(dataset_train)))

	# Init loggers
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	writer = SummaryWriter(args.log_dir)
	logger = init_logger(args)

	# Create model
	if not args.gray:
		in_ch = 3
	else:
		in_ch = 1
	net = FFDNet(num_input_channels=in_ch)
	# Initialize model with He init
	net.apply(weights_init_kaiming)
	# Define loss
	criterion = nn.MSELoss(size_average=False)

	# Move to GPU
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# Resume training or start anew
	if args.resume_training:
		resumef = os.path.join(args.log_dir, 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = args.epochs
			new_milestone = args.milestone
			current_lr = args.lr
			args = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			args.epochs = new_epoch
			args.milestone = new_milestone
			args.lr = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = vars(checkpoint['args'])
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			args.resume_training = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0
		training_params['no_orthog'] = args.no_orthog

	# Training
	for epoch in range(start_epoch, args.epochs):
		# Learning rate value scheduling according to args.milestone
		if epoch > args.milestone[1]:
			current_lr = args.lr / 1000.
			training_params['no_orthog'] = True
		elif epoch > args.milestone[0]:
			current_lr = args.lr / 10.
		else:
			current_lr = args.lr

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('learning rate %f' % current_lr)

		# train
		for i, data in enumerate(loader_train, 0):
			# Pre-training step
			model.train()
			model.zero_grad()
			optimizer.zero_grad()

			# inputs: noise and noisy image
			img_train = data
			noise = torch.zeros(img_train.size())
			stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
							size=noise.size()[0])
			for nx in range(noise.size()[0]):
				sizen = noise[0, :, :, :].size()
				noise[nx, :, :, :] = torch.FloatTensor(sizen).\
									normal_(mean=0, std=stdn[nx])
			imgn_train = img_train + noise
			# Create input Variables
			img_train = Variable(img_train.cuda())
			imgn_train = Variable(imgn_train.cuda())
			noise = Variable(noise.cuda())
			stdn_var = Variable(torch.cuda.FloatTensor(stdn))

			# Evaluate model and optimize it
			out_train = model(imgn_train, stdn_var)
			loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
			loss.backward()
			optimizer.step()

			# Results
			model.eval()
			out_train = torch.clamp(imgn_train-model(imgn_train, stdn_var), 0., 1.)
			psnr_train = batch_psnr(out_train, img_train, 1.)
			# PyTorch v0.4.0: loss.data[0] --> loss.item()

			if training_params['step'] % args.save_every == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Log the scalar values
				writer.add_scalar('loss', loss.data[0], training_params['step'])
				writer.add_scalar('PSNR on training data', psnr_train, \
					  training_params['step'])
				print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %\
					(epoch+1, i+1, len(loader_train), loss.data[0], psnr_train))
			training_params['step'] += 1
		# The end of each epoch
		model.eval()

		# Validation
		psnr_val = 0
		for valimg in dataset_val:
			img_val = torch.unsqueeze(valimg, 0)
			noise = torch.FloatTensor(img_val.size()).\
					normal_(mean=0, std=args.val_noiseL)
			imgn_val = img_val + noise
			img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
			sigma_noise = Variable(torch.cuda.FloatTensor([args.val_noiseL]))
			out_val = torch.clamp(imgn_val-model(imgn_val, sigma_noise), 0., 1.)
			psnr_val += batch_psnr(out_val, img_val, 1.)
		psnr_val /= len(dataset_val)
		print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
		writer.add_scalar('PSNR on validation data', psnr_val, epoch)
		writer.add_scalar('Learning rate', current_lr, epoch)

		# Log val images
		try:
			if epoch == 0:
				# Log graph of the model
				writer.add_graph(model, (imgn_val, sigma_noise), )
				# Log validation images
				for idx in range(2):
					imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
											nrow=2, normalize=False, scale_each=False)
					imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
											nrow=2, normalize=False, scale_each=False)
					writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
					writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
			for idx in range(2):
				imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
										nrow=2, normalize=False, scale_each=False)
				writer.add_image('Reconstructed validation image {}'.format(idx), \
								imrecons, epoch)
			# Log training images
			imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
						 scale_each=True)
			writer.add_image('Training patches', imclean, epoch)

		except Exception as e:
			logger.error("Couldn't log results: {}".format(e))

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))
		save_dict = { \
			'state_dict': model.state_dict(), \
			'optimizer' : optimizer.state_dict(), \
			'training_params': training_params, \
			'args': args\
			}
		torch.save(save_dict, os.path.join(args.log_dir, 'ckpt.pth'))
		if epoch % args.save_every_epochs == 0:
			torch.save(save_dict, os.path.join(args.log_dir, \
									  'ckpt_e{}.pth'.format(epoch+1)))
		del save_dict

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="FFDNet")
	parser.add_argument("--gray", action='store_true',\
						help='train grayscale image denoising instead of RGB')

	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	#Training parameters
	parser.add_argument("--batch_size", type=int, default=128, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	argspar = parser.parse_args()
	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noiseIntL[0] /= 255.
	argspar.noiseIntL[1] /= 255.

	print("\n### Training FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(argspar)
