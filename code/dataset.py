from torch.utils.data import Dataset
# from load_blender import pose_spherical
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
import os
import time
import numpy as np
import pickle
import sys
import glob
import pdb
import torch
import random
import natsort
from load_llff_batch import _minify, recenter_poses, poses_avg, normalize, spherify_poses
import os, imageio
import natsort
from run_nerf_helpers_batch import get_rays, get_rays_np
# sys.path.append('/home/prao/windows/nerf/torch_sampling')
# from torch_sampling import choice
import datetime
import pathlib
from os.path import dirname as osdir
from os.path import basename as osbname
import cv2
from PIL import Image

trans_t = lambda t: torch.Tensor([
	[1, 0, 0, t / 2],
	[0, 1, 0, 0],
	[0, 0, 1, t],
	[0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
	[1, 0, 0, 0],
	[0, np.cos(phi), -np.sin(phi), 0],
	[0, np.sin(phi), np.cos(phi), 0],
	[0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
	[np.cos(th), 0, -np.sin(th), 0],
	[0, 1, 0, 0],
	[np.sin(th), 0, np.cos(th), 0],
	[0, 0, 0, 1]]).float()

rot_z = lambda z: torch.Tensor([
	[np.cos(z), -np.sin(z), 0, 0],
	[np.sin(z), np.cos(z), 0, 0],
	[0, 0, 1, 0],
	[0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, z, radius, device, mean_pose=None):
	if mean_pose is None:
		c2w = trans_t(radius)
	else:
		c2w = np.vstack((mean_pose, [0, 0, 0, 1]))
		c2w = torch.Tensor(c2w)
	c2w = rot_phi(phi / 180. * np.pi) @ c2w
	c2w = rot_theta(theta / 180. * np.pi) @ c2w
	c2w = rot_z(z / 180. * np.pi) @ c2w
	# c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
	return c2w


def getInterpolatedPose(pose1, pose2, ratio):
	out = pose1 * 0

	rotMat1 = R.from_matrix(pose1[:, 0:3])
	rotMat2 = R.from_matrix(pose2[:, 0:3])
	angles1 = R.as_rotvec(rotMat1)
	angles2 = R.as_rotvec(rotMat2)
	angle = (angles1 * ratio) + (angles2 * (1 - ratio))
	rotMat = R.from_rotvec(angle)

	trans1 = pose1[:, 3]
	trans2 = pose2[:, 3]
	trans = (trans1 * ratio) + (trans2 * (1 - ratio))

	out[:, 0:3] = rotMat.as_matrix()
	out[:, 3] = trans

	return out


def getRenderPoses(_poses, samp_img, fit, prefix):
	if prefix == 'OLAT_images':
		ID = int(osbname(osdir(samp_img))[2:])
	else:
		ID = int(osbname(osdir(osdir(samp_img)))[2:])
	poseSamples = 10
	out = None
	if len(_poses) == 16:
		poses = np.delete(_poses, [0, 13], axis=0)
		assert len(poses) == 14
	elif (len(_poses) > 14 and ID in [2, 7, 53]): # has CAM 0 and missing CAM 13
		poses = np.delete(_poses, [0], axis=0)
		assert len(poses) == 14
	elif (len(_poses) > 14 and ID in [11, 15, 18, 19, 23, 24, 25, 33, 52]): # has CAM 13 and missing CAM 0
		poses = np.delete(_poses, [13], axis=0)
		assert len(poses) == 14
	else:
		poses = _poses

	poses = np.load('data_utils/render_poses_full.npy')
	# mean_pose =  poses[6][:3,:4] if len(poses) >= 7 else poses[3][:3,:4]
	if len(poses) >= 7:
		mean_pose = poses[6][:3, :4]
	elif 7 > len(poses) >= 4:
		mean_pose = poses[3][:3, :4]
	else:
		mean_pose = poses[0][:3, :4]

	for i in range(poses.shape[0] - 1):
		start_pose = poses[i][:, 0:4]
		end_pose = poses[i + 1][:, 0:4]
		for j in range(poseSamples):
			ratio = 1 - (j / poseSamples)
			tt = getInterpolatedPose(start_pose, end_pose, ratio)
			tt = tt.reshape(1, 3, 4)
			if out is None:
				out = tt
			else:
				out = np.concatenate((out, tt))
	return out, mean_pose


class ImagesDataset(Dataset):

	def __init__(self, args, data_fol, aux_data_fol, train, device, prefix, perceptual_loss=False, perceptual_loss_prob=0.5):
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		self.args = args
		self.data_fol = data_fol
		self.aux_data_fol = aux_data_fol
		self.train = train
		self.device = device
		self.prefix = prefix
		self.OLAT_config = args.OLAT_config
		self.init(train)
		self.ii = None
		self.jj = None
		self.perceptual_loss = perceptual_loss
		self.init_perceptual_loss = perceptual_loss
		self.perceptual_loss_prob = perceptual_loss_prob

	def init(self, train):
		self.all_imgs = self.getAllImgs(self.data_fol,self.aux_data_fol, train)

	def __len__(self):
		return len(self.all_imgs)

	def getAllImgs(self, data_fol, aux_data_fol, train):
		print(self.prefix)
		import pdb; pdb.set_trace()
		image_ids_poor_mask = []
		# image_ids_poor_mask = [1, 7, 11, 13, 18, 26, 30, 45, 49, 64, 67, 83, 89, 100, 112, 121, 127]
		if train:
			if self.prefix == 'OLAT_images':
				# if self.args.test:
				# 	with open('./OLATs/{}.txt'.format(osbname(self.data_fol)), 'r') as f:
				# 		all_imgs = f.read().splitlines()
				# else:
				if ('3dpr' in self.data_fol) or ('h3ds' in self.data_fol) or ('celeba' in self.data_fol):
					print('./OLATs/{}{}.txt'.format(osbname(self.data_fol), self.OLAT_config))
					with open('./OLATs/{}{}.txt'.format(osbname(self.data_fol), self.OLAT_config), 'r') as f:
						all_imgs = f.read().splitlines()
				else:
					print('./OLATs/{}{}.txt'.format(osbname(self.data_fol[:-3]), self.OLAT_config))
					with open('./OLATs/{}{}.txt'.format(osbname(self.data_fol[:-3]), self.OLAT_config), 'r') as f:
						all_imgs = f.read().splitlines()
			else:
				if self.args.num_train_IDs == 50:
					all_imgs = sorted(glob.glob(data_fol + '/ID000[0-4]*/'+self.prefix+'/*png'))
					all_imgs.extend(sorted(glob.glob(data_fol + '/ID0005[0-3]/'+self.prefix+'/*png')))
				elif self.args.num_train_IDs == 100:
					# print([ os.path.basename(os.path.dirname(x)) for x in sorted(glob.glob(data_fol + '/ID000[0-9]*/'))])
					# print([ os.path.basename(os.path.dirname(x)) for x in sorted(glob.glob(data_fol + '/ID0010[0-3]/'))])
					all_imgs = sorted(glob.glob(data_fol + '/ID000[0-9]*/' + self.prefix + '/*png'))
					all_imgs.extend(glob.glob(data_fol + '/ID0010[0-3]/' + self.prefix + '/*png'))
				elif self.args.num_train_IDs == -1:
					all_imgs = sorted(glob.glob(data_fol + '/*/'+self.prefix+'/*png'))
				else:
					raise NotImplementedError
				print(len(all_imgs))

				for aux_dir in aux_data_fol:
					if 'i3DMM' in aux_dir:
						all_imgs.extend(sorted(glob.glob(aux_dir + '/*/' + self.prefix + '/*png'))*75)
					else:
						if self.args.num_train_IDs == 50:
							all_imgs.extend(sorted(glob.glob(aux_dir + '/ID000[0-4]*/'+self.prefix+'/*png')))
							all_imgs.extend(sorted(glob.glob(aux_dir + '/ID005[0-3]/'+self.prefix+'/*png')))
						elif self.args.num_train_IDs == 100:
							all_imgs.extend(sorted(glob.glob(aux_dir + '/ID000[0-9]*/' + self.prefix + '/*png')))
							all_imgs.extend(sorted(glob.glob(aux_dir + '/ID0010[0-3]/' + self.prefix + '/*png')))
						elif self.args.num_train_IDs == -1:
							all_imgs.extend(sorted(glob.glob(aux_dir + '/*/'+self.prefix+'/*png')))
						else:
							raise NotImplementedError

					print(aux_dir, len(all_imgs))

				# all_imgs = [a for a in all_imgs if
			    #         os.path.isfile(osdir(osdir(a)) + '/' + self.args.poses_bounds_fn)]

				if self.args.num_relit_emaps == 15:
					SELECT_EMAPS = [
						'EMAP-000', 'EMAP-003', 'EMAP-012', 'EMAP-015', 'EMAP-028',
						'EMAP-030', 'EMAP-035', 'EMAP-043', 'EMAP-048', 'EMAP-054',
						'EMAP-061', 'EMAP-064', 'EMAP-075', 'EMAP-078', 'EMAP-081',
						'EMAP-999'  # testing
					]
				if len(SELECT_EMAPS)>0 and 'EMAP' in osbname(all_imgs[0]):
					all_imgs = [i for i in all_imgs if osbname(i).split('_')[-1][:8] in SELECT_EMAPS]
				print("Using total {} images".format(len(all_imgs)))
		else:
			all_imgs = sorted(glob.glob(data_fol + '/'+self.prefix+'/*png'))

		# out = []
		# for img in all_imgs:
		# 	img_id = osbname(img)
		# 	# img_id = int(img_id.split('.')[0])
		# 	img_id = int(img_id.split('_')[0][3:])
		# 	if img_id not in image_ids_poor_mask:
		# 		out.append(img)

		return all_imgs

	def getSampleData(self, samp_img):

		args = self.args
		switch_prob = self.perceptual_loss_prob if self.init_perceptual_loss else 1 # [1 -> no perceptual loss]
		self.perceptual_loss = np.random.choice([False, True], 1, p=[ switch_prob, 1 - switch_prob])[0]
		# print(self.perceptual_loss, samp_img)

		if args.dataset_type == 'llff':
			bd_factor = None  # 0.75 #TODO: make this configurable
			images, poses, bds, lights_arr, render_train_poses, render_poses = self.load_llff_data(samp_img, args.factor,
			                                                                                 recenter=args.recenter_poses,
			                                                                                 bd_factor=bd_factor,
			                                                                                 spherify=args.spherify,
			                                                                                 load_mask=args.load_mask)
			if images is None:
				return None, None, None, None
			train_poses = poses.copy()

			poses = torch.Tensor(poses).to(self.device)
			poses = poses.view(1, poses.shape[0], poses.shape[1])

			images = np.transpose(images, (1, 2, 0))
			images = torch.Tensor(images).to(self.device)
			# images = images/255
			images = images.view(1, images.shape[0], images.shape[1], images.shape[2])

			hwf = poses[0, :3, -1]
			poses = poses[:, :3, :4]
			# print('Loaded llff', images.shape, hwf, args.datadir, bds)

			# print('DEFINING BOUNDS')
			if args.no_ndc:
				near = np.ndarray.min(bds) * .9
				far = np.ndarray.max(bds) * 1.
			elif args.custom_bds:
				near = args.near
				far = args.far
			else:
				near = 0.
				far = 1.
			# print('NEAR FAR', near, far)

			if args.load_mask:
				mask = images[..., -1:]
				if not args.acc_loss:
					mask = torch.ones_like(mask) #DEBUG

			if args.use_light_dirs:
				lights = images[..., 3:-1]

			if args.white_bkgd:
				images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
			else:
				images = images[..., :3]

		else:
			print('Unknown dataset type', args.dataset_type, 'exiting')
			return

		# Cast intrinsics to right types
		H, W, focal = hwf
		H, W = int(H), int(W)
		hwf = [H, W, focal]


		# H, W ,focal = 1030, 1300, torch.tensor(2591.7990502629186)
		# hwf = [H, W, focal]
		# print("HWF: ", hwf)

		if self.ii is None or (self.ii.shape[0] != H and self.ii.shape[1] != W):
			# full image
			i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
			                      torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
			self.ii = i.t().to(self.device)
			self.jj = j.t().to(self.device)

		if self.perceptual_loss:
			# Random crops within bounding box : [x_pos, y_pos, width, height] = [200, 200, 700, 500]
			offset_W = np.random.randint(0, 700//abs(args.factor)) + 200//abs(args.factor)
			offset_H = np.random.randint(0, 500//abs(args.factor)) + 200//abs(args.factor)
			crop_W, crop_H = self.args.crop_dim, self.args.crop_dim
			images = images[:, offset_H:offset_H + crop_H, offset_W:offset_W + crop_W] # images : [H, W]
			lights = lights[:, :crop_W, :crop_H, :]
			# cv2.imwrite('images/{}_images.png'.format(os.path.basename(samp_img)), images.squeeze(0).data.cpu().numpy()[:, :, ::-1] * 255)

		# Prepare raybatch tensor if batching random rays
		N_rand = args.N_rand
		use_batching = not args.no_batching
		if use_batching:
			# For random ray batching
			if self.perceptual_loss:
				rays_o, rays_d = get_rays(H, W, focal, poses[0, :3, :4], self.ii[:crop_W, :crop_W] + offset_W, self.jj[:crop_H, :crop_H] + offset_H)
			else:
				rays_o, rays_d = get_rays(H, W, focal, poses[0, :3, :4], self.ii, self.jj)
			rays = torch.stack([rays_o, rays_d], 0)  # [N, ro+rd, H, W, 3]
			rays_rgb = torch.cat([rays, images], 0)  # [N, ro+rd+rgb, H, W, 3]
			info_rays_rgb = 3  # ro+rd+rgb

			if args.use_light_dirs:
				rays_rgb = torch.cat([rays_rgb, lights], 0)
				info_rays_rgb += 1  # [N, ro+rd+rgb+light, H, W, 3]

			if args.acc_loss:
				# acc_mask = torch.tile(mask, (1, 1, 1, 3))
				acc_mask = mask.repeat((1, 1, 1, 3))
				rays_rgb = torch.cat([rays_rgb, acc_mask], 0)  # [N, ro+rd+rgb+mask, H, W, 3]
				info_rays_rgb += 1  # ro+rd+rgb+light+mask

			if not self.perceptual_loss:
				mask = mask.view(W * H)
				# fg_inds = torch.logical_and(mask!=2, mask==1).nonzero(as_tuple=True)[0]
				# bg_inds = torch.logical_and(mask!=2, mask==0).nonzero(as_tuple=True)[0]
				fg_inds = (mask == 1).nonzero(as_tuple=True)[0]
				bg_inds = (mask == 0).nonzero(as_tuple=True)[0]

				if fg_inds.shape[0] > 0:
					if args.acc_loss:
						fg_pixels = int(N_rand * 0.8)
						bg_pixels = N_rand - fg_pixels
					else:
						fg_pixels = int(N_rand * 1.0)

					fg_sel_inds = torch.multinomial(torch.ones(fg_inds.shape[0]), num_samples=fg_pixels, replacement=False)
					if args.acc_loss:
						bg_sel_inds = torch.multinomial(torch.ones(bg_inds.shape[0]), num_samples=bg_pixels, replacement=False)

					sel_inds = torch.cat([fg_sel_inds, bg_sel_inds], 0) if args.acc_loss else fg_sel_inds
				else:
					fg_pixels = 0
					bg_pixels = N_rand - fg_pixels
					sel_inds = torch.multinomial(torch.ones(bg_inds.shape[0]), num_samples=bg_pixels, replacement=True)

				rays_rgb = rays_rgb.permute(1, 2, 0, 3)  # [N, H, W, ro+rd+rgb, 3]
				rays_rgb = rays_rgb.view(W * H, info_rays_rgb, 3)
				rays_rgb = rays_rgb[sel_inds]
			else:
				rays_rgb = rays_rgb.permute(1, 2, 0, 3)  # [N, H, W, ro+rd+rgb, 3]
				rays_rgb = rays_rgb.view(crop_W * crop_H, info_rays_rgb, 3)
				rays_rgb = rays_rgb

		batch = torch.transpose(rays_rgb, 0, 1)
		return batch, hwf, train_poses, render_train_poses, render_poses, lights_arr

	def getSampInd(self, samp_img):
		# fls = sorted(glob.glob(osdir(samp_img)+'/*png'))

		basedir = osdir(samp_img)
		if not os.path.isdir(basedir):
			return None
		if self.prefix == 'OLAT_images':
			fls = self.all_imgs
		else:
			fls = [os.path.join(basedir, f) for f in natsort.natsorted(os.listdir(basedir)) if
		       f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
		return fls.index(samp_img)

	def _load_data(self, samp_img, factor=None, width=None, height=None, load_imgs=True, load_mask=False):
		# print(samp_img)
		if self.prefix == 'OLAT_images':
			ID = osbname(osdir(samp_img))
			basedir = os.path.join(self.data_fol, ID)
		else:
			basedir = osdir(osdir(samp_img))
			# ID = osbname(osdir(osdir(samp_img)))
			# basedir = os.path.join(self.data_fol, ID)

		samp_ind_in_scan = self.getSampInd(samp_img)
		arr_ind = None
		if samp_ind_in_scan is None:
			print("Image Not Found!")
			return None, None, None

		poses_arr = np.load(os.path.join(basedir, self.args.poses_bounds_fn))
		poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
		bds = poses_arr[:, -2:].transpose([1, 0])

		if self.prefix == 'OLAT_images':
			img0 = self.all_imgs[0]
			sh = np.rot90(imageio.imread(img0)).shape
		else:
			img0 = \
			[os.path.join(basedir, self.prefix, f) for f in natsort.natsorted(os.listdir(os.path.join(basedir, self.prefix))) \
			 if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
			sh = imageio.imread(img0).shape

		sfx = ''

		if factor is not None:
			sfx = '_{}'.format(factor)
			_minify(basedir, factors=[factor])
			factor = factor
		elif height is not None:
			raise ValueError('height not supported')
			factor = sh[0] / float(height)
			width = int(sh[1] / factor)
			_minify(basedir, resolutions=[[height, width]])
			sfx = '_{}x{}'.format(width, height)
		elif width is not None:
			raise ValueError('width not supported')
			factor = sh[1] / float(width)
			height = int(sh[0] / factor)
			_minify(basedir, resolutions=[[height, width]])
			sfx = '_{}x{}'.format(width, height)
		else:
			factor = 1

		if not self.prefix == 'OLAT_images':
			imgdir = os.path.join(basedir, self.prefix + sfx)
			if not os.path.exists(imgdir):
				print(imgdir, 'does not exist, returning')
				return

		if load_mask:
			maskdir = os.path.join(basedir, 'masks' + sfx)
			if not os.path.exists(maskdir):
				print(maskdir, 'does not exist, returning')
				return

		# natsort sorts images according to the image idx, ['10.png', '2.png', '1.png'] --> ['1.png', '2.png', '10.png']
		if self.prefix == 'OLAT_images':
			imgfiles = self.all_imgs
		else:
			imgfiles = [os.path.join(imgdir, f) for f in natsort.natsorted(os.listdir(imgdir))
		            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
		if load_mask:
			maskfiles = [os.path.join(maskdir, f) for f in natsort.natsorted(os.listdir(maskdir))
			             if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
		# Sanity Check
		# if poses.shape[-1] != len(imgfiles):
		# 	print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
		# 	return
		# if load_mask:
		# 	if ((poses.shape[-1] != len(imgfiles)) or (poses.shape[-1] != len(maskfiles))):
		# 		print('Mismatch between imgs {}, masks {} and poses {} !!!!'.format(len(imgfiles), len(maskfiles),
		# 		                                                                    poses.shape[-1]))
		# 		return
		# else:
		# 	if (poses.shape[-1] != len(imgfiles)):
		# 		print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
		# 		return
		# sh = imageio.imread(imgfiles[0]).shape
		sh = (np.ceil(sh[0]/factor).astype(int), np.ceil(sh[1]/factor).astype(int), sh[2])

		def imread(f, load_mask = False, factor=1):
			if f.endswith('png'):
				if (self.prefix == 'OLAT_images' and load_mask == False):
					img = np.float32(cv2.cvtColor(cv2.imread(f, -1), cv2.COLOR_BGR2RGB))
					assert factor == 1 or factor == 8, "Only 1/8 rescaling possible!"
					# img = np.array(Image.fromarray(img).resize((129, 163)) if factor == 8 else img)
					img = cv2.resize(img, (129, 163)) if factor == 8 else img
					return np.rot90(img)
				elif load_mask and factor==4:
					img = imageio.imread(f, ignoregamma=True)
					img = img[:,:,:3] if len(img.shape) > 2 else img
					_sh = img.shape
					img = np.resize(img, (_sh[0]//factor, _sh[1]//factor, 3))
					return img[:,:,0]
				else:
					img = imageio.imread(f, ignoregamma=True)
					img = img[:,:,:3] if len(img.shape) > 2 else img
					return img
			else:
				assert f.endswith('png'), "Invalid File Extension"

		filename = imgfiles[samp_ind_in_scan]
		assert osbname(filename) == osbname(samp_img), print(filename, samp_img)
		# print(filename)
		if self.prefix == 'OLAT_images':
			CAM_ID = osbname(osdir(osdir(filename)))
			OLAT_ID = osbname(filename).split('.')[0]
			imgs = imread(filename, factor=factor)[..., :3] / 65535.
		else:
			CAM_ID = osbname(filename).split('-')[0][:5]
			OLAT_ID = osbname(imgfiles[samp_ind_in_scan]).split('.')[0].split('_')[-1]
			imgs = imread(filename, factor=factor)[..., :3] / 255.
		if 'EMAP' in OLAT_ID:
			OLAT_ID = -1
		# elif 'OLAT' in OLAT_ID:
		elif len(OLAT_ID) == 3:
			#TODO: Create OLAT images with consistent format
			# OLAT_ID = int(OLAT_ID[5:8])
			OLAT_ID = int(OLAT_ID)
		else:
			OLAT_ID = 0 #just single illumn


		if self.args.use_light_dirs:
			lights_arr = np.load(os.path.join(basedir, 'light_dirs.npy'))

			if OLAT_ID == -1: #uniform illuminations
				lights_arr = np.zeros_like(lights_arr[OLAT_ID])
			else:
				from data_utils import olat_indices
				if self.OLAT_config == '50_OLATs':
					assert OLAT_ID in olat_indices.config_50, AttributeError
				elif self.OLAT_config == '100_OLATs':
					assert OLAT_ID in olat_indices.config_100, AttributeError
				lights_arr = lights_arr[OLAT_ID]

			lights_arr = lights_arr[np.newaxis, np.newaxis, :]
			lights = np.tile(lights_arr, (sh[0], sh[1], 1))
			if len(imgs.shape)==2:
				imgs = np.zeros_like(imread(img0, factor=factor))
			imgs = np.concatenate((imgs, lights), axis=2)
			lights_arr_out = np.squeeze(lights_arr.T)
		else:
			lights_arr_out = None

		if load_mask:
			mask_filename = [f for f in maskfiles if CAM_ID in f]
			# print(maskfiles, mask_filename, CAM_ID)
			mask_filename = mask_filename[0]
			arr_ind = maskfiles.index(mask_filename)
			mask = imread(mask_filename, load_mask, factor=factor)[..., np.newaxis] >= 1
			# import pdb; pdb.set_trace()
			imgs = np.concatenate((imgs, mask), axis=2)

		sh = imgs.shape
		if len(sh) < 3:
			return None, None, None, None

		poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
		poses[2, 4, :] = poses[2, 4, :] * 1. / factor

		if not load_imgs:
			raise ValueError('load_imgs=False not supported')
			return poses, bds

		if arr_ind is None:
			arr_ind = samp_ind_in_scan
		s_bds = np.take(bds, arr_ind, 1)
		# print(poses.shape, imgs.shape, s_bds, lights_arr.shape, arr_ind)
		# return poses, s_bds, imgs, arr_ind, distortion_mask, depths
		return poses, s_bds, imgs, lights_arr_out, arr_ind

	def load_llff_data(self, samp_img, factor=None, recenter=True, bd_factor=None, spherify=False, path_zflat=False,
	                   load_mask=False):
		factor = factor if (factor != -1) else None
		ID = osbname(osdir(osdir(samp_img)))

		# change factor to 1/4 for i3DMM
		if self.prefix == 'images':
			factor = 4 if int(ID[3:]) >= 600 else factor

		poses, bds, imgs, lights_arr, arr_ind = self._load_data(samp_img, factor=factor, load_mask=load_mask)  # factor=8 downsamples original imgs by 8x
		if poses is None:
			return None, None, None, None
		# print('Loaded', samp_img, bds.min(), bds.max())

		# Correct rotation matrix ordering and move variable dim to axis 0
		poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
		poses = np.moveaxis(poses, -1, 0).astype(np.float32)
		imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
		images = imgs
		bds = np.moveaxis(bds, -1, 0).astype(np.float32)

		# Rescale if bd_factor is provided
		sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
		poses[:, :3, 3] *= sc
		bds *= sc

		if recenter:
			o_poses = recenter_poses(poses)
		else:
			o_poses = poses
		interpolated_poses, mean_pose = getRenderPoses(o_poses, samp_img, self.args.fit, self.prefix)
		# reference_interp = np.load("reference_poses/10IDs_ID00000_interp_poses.npy")
		# mean_pose = np.take(interpolated_poses, 50, 0)

		poses = np.take(o_poses, arr_ind, 0)
		if spherify:
			poses, render_poses, bds = spherify_poses(poses, bds)
		else:
			# replaced render spriral
			# z_angle =
			# render_poses = torch.stack([pose_spherical(0, -90, 0, 3.0) for angle in np.linspace(0,3.5,40+1)], 0)
			# render_poses = torch.stack([pose_spherical(theta=0, phi=angle, z=0) for angle in np.linspace(0, 180, 100+1)[:-1]], radius= 3.0)
			# render_poses = torch.stack([pose_spherical(theta=0, phi=angle, z=0, radius=2.732, mean_pose=mean_pose) for angle in np.linspace(-180, 180, 180 + 1)[:-1]], 0)
			if not self.args.render_single:
				render_poses = torch.stack([pose_spherical(theta=0, phi=angle, z=0, radius=2.862, mean_pose=mean_pose, device=self.device)
				                            # for angle in np.linspace(-60, 50, 100 + 1)[:-1]], 0)
				                            for angle in np.linspace(-60, 50, 10 + 1)[:-1]], 0)
			else:
				render_poses = torch.stack([pose_spherical(theta=0, phi=angle, z=0, radius=2.862, mean_pose=mean_pose, device=self.device)
				                            # for angle in np.linspace(0, 0, 1 + 1)[:-1]], 0)
				                            for angle in np.linspace(-60, 45, 50 + 1)[:-1]], 0)

		# render_poses = np.array(render_poses.data.cpu()).astype(np.float32)
		images = images.astype(np.float32)
		poses = poses.astype(np.float32)
		render_train_poses = o_poses.astype(np.float32)
		# return images, poses, bds, render_poses, distortion_mask, depths
		return images, poses, bds, lights_arr, render_train_poses, render_poses

	def __getitem__(self, index):
		# Get sample id
		samp_img = self.all_imgs[index]
		
		if self.prefix == 'OLAT_images':
			scan_id = osbname(osdir(samp_img))
			img_name = osbname(samp_img)
			cam_id = osbname(osdir(osdir(samp_img)))
		else:
			scan_id = osbname(osdir(osdir(samp_img)))
			img_name = osbname(samp_img)
			cam_id = img_name[:5]

		# Get image, pose, bds, etc
		# print('after _load', time.time())
		batch, hwf, o_poses, render_train_poses, render_poses, lights_arr = self.getSampleData(samp_img)
		# print('before _load', time.time())

		while batch is None:
			index = random.randint(0, len(self.all_imgs) - 1)
			samp_img = self.all_imgs[index]
			scan_id = osbname(osdir(osdir(samp_img)))
			img_name = osbname(samp_img)
			cam_id = img_name[:5]
			# Get image, pose, bds, etc
			batch, hwf, o_poses, render_train_poses, render_poses, lights_arr = self.getSampleData(samp_img)

		return batch, hwf, o_poses, scan_id, render_train_poses, render_poses, img_name, lights_arr, cam_id, self.perceptual_loss


class data_prefetcher():
	def __init__(self, loader):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.preload()

	def preload(self):
		try:
			self.next_batch, self.next_hwf, self.next_o_poses, self.next_scan_id, self.next_render_train_poses, \
			self.next_render_poses,  self.next_img_name, self.next_lights_arr, self.next_cam_id, self.next_perceptual_loss = next(self.loader)
		except StopIteration:
			self.next_batch = None
			self.next_hwf = None
			self.next_o_poses = None
			self.next_scan_id = None
			self.next_render_train_poses = None
			self.next_render_poses = None
			self.next_img_name = None
			self.next_lights_arr = None
			self.next_cam_id = None
			self.next_perceptual_loss = None
			return

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		batch = self.next_batch
		hwf = self.next_hwf
		o_poses = self.next_o_poses
		scan_id = self.next_scan_id
		render_train_poses = self.next_render_train_poses
		render_poses = self.next_render_poses
		img_name = self.next_img_name
		lights_arr = self.next_lights_arr
		cam_id =self.next_cam_id
		perceptual_loss =self.next_perceptual_loss
		self.preload()
		return batch, hwf, o_poses, scan_id, render_train_poses, render_poses, img_name, lights_arr, cam_id, perceptual_loss
