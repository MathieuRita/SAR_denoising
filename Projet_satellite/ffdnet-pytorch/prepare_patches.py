"""
Construction of the training and validation databases

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import argparse
from dataset import prepare_data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=\
								  "Building the training patch database")
	parser.add_argument("--gray", action='store_true',\
						help='prepare grayscale database instead of RGB')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=44, \
					 help="Patch size")
	parser.add_argument("--stride", "--s", type=int, default=20, \
					 help="Size of stride")
	parser.add_argument("--max_number_patches", "--m", type=int, default=None, \
						help="Maximum number of patches")
	parser.add_argument("--aug_times", "--a", type=int, default=1, \
						help="How many times to perform data augmentation")
	# Dirs
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	args = parser.parse_args()

	if args.gray:
		if args.trainset_dir is None:
			args.trainset_dir = 'data/gray/train'
		if args.valset_dir is None:
			args.valset_dir = 'data/gray/Set12'
	else:
		if args.trainset_dir is None:
			args.trainset_dir = 'data/rgb/CImageNet_expl'
		if args.valset_dir is None:
			args.valset_dir = 'data/rgb/Kodak24'

	print("\n### Building databases ###")
	print("> Parameters:")
	for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	prepare_data(args.trainset_dir,\
					args.valset_dir,\
					args.patch_size,\
					args.stride,\
					args.max_number_patches,\
					aug_times=args.aug_times,\
					gray_mode=args.gray)
