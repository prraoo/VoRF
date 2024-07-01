import os
import imageio
import time
from tqdm import tqdm, trange
import shutil
import scipy.io as sio
import glob
import cv2
import torch
from dataset import ImagesDataset
from dataset import data_prefetcher
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from lpips_pytorch import LPIPS, lpips

from run_nerf_helpers_batch import *
from torch.utils.tensorboard import SummaryWriter
import natsort
import itertools
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs, use_OLAT):
        return torch.cat([fn(inputs[i:i+chunk], use_OLAT) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, latent_vector, viewdirs, light_dirs, fn, embed_fn,
                embeddirs_fn, embed_light_dirs_fn, use_shading_nw, use_OLAT, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    latent_vector_flat = torch.reshape(latent_vector, [-1, latent_vector.shape[-1]])
    embedded = embed_fn(inputs_flat)
    # [ {input + latent_code} + view_dirs]
    embedded = torch.cat([embedded, latent_vector_flat], -1)

    if viewdirs is not None:
        # input_dirs = viewdirs[:,None].expand(inputs[:,:,:3].shape)
        input_dirs = viewdirs[:, None].repeat(1, inputs.shape[1], 1)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    if use_shading_nw:
        light_dirs = light_dirs[:, None].repeat(1, inputs.shape[1], 1)
        light_dirs_flat = torch.reshape(light_dirs, [-1, light_dirs.shape[-1]])
        embedded_light_dirs = embed_light_dirs_fn(light_dirs_flat)
        embedded = torch.cat([embedded, embedded_light_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded, use_OLAT)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, latent_vector, use_OLAT,  chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], latent_vector[i:i+chunk], use_OLAT, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, use_OLAT, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., latent_vector = None,
                  use_viewdirs=False, c2w_staticcam=None,light_vector=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays2render(H, W, focal, c2w)

        if kwargs['use_light_dirs']:
            rays_l = light_vector * torch.ones_like(rays_d)

        latent_vector = latent_vector[0].repeat(rays_o.shape[0] * rays_o.shape[1], 1)
    else:
        # use provided ray batch
        if kwargs['use_light_dirs']:
            rays_o, rays_d, rays_l = rays
        else:
            rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    if kwargs['use_light_dirs']:
        rays_l = torch.reshape(rays_l, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if kwargs['use_light_dirs']:
        viewdirs = torch.cat([viewdirs, rays_l], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)


    # Render and reshape
    all_ret = batchify_rays(rays, latent_vector, use_OLAT, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'weights']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, use_OLAT, chunk, render_kwargs, latent_vector=None, gt_imgs=None, savedir=None, render_factor=0,
                tone_mapping=False, light_vector=None, prefix=None, camera_idx=None):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    rgbs_raw = []
    disps = []
    if light_vector is not None:
        # light_vector = torch.Tensor(light_vector.float()).to(device)
        light_vector = light_vector.to(device)

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses, disable=True)):
        # print(i, time.time() - t)
        t = time.time()
        filename = os.path.join(savedir, prefix + '_{:03d}.png'.format(i if camera_idx is None else camera_idx[i]))
        if os.path.exists(filename):
            continue
        rgb, disp, acc, _, _ = render(H, W, focal, use_OLAT, chunk=chunk, c2w=c2w[:3,:4],light_vector=light_vector, latent_vector=latent_vector, **render_kwargs)
        if tone_mapping and use_OLAT:
            rgbs_raw.append(rgb.cpu().numpy())
            rgbs.append(np.rot90(np.rot90(np.rot90(tonemap(rgb.cpu().numpy())))))
        else:
            # rgbs.append(rgb.cpu().numpy())
            rgbs.append(np.rot90(np.rot90(np.rot90(rgb.cpu().numpy()))))

        disps.append(disp.cpu().numpy())
        # if i==0:
            # print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            if use_OLAT:
                filename = os.path.join(savedir, prefix + '_{:03d}.npy'.format(i if camera_idx is None else camera_idx[i]))
                np.save(filename, rgbs_raw[-1]*65535 )
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, prefix+'_{:03d}.png'.format(i if camera_idx is None else camera_idx[i]))
                # imageio.imwrite(filename, cv2.resize(rgb8, (1300//4, 1030//4)))
                imageio.imwrite(filename, rgb8)
            else:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, prefix+'_{:03d}.png'.format(i))
                # imageio.imwrite(filename, cv2.resize(rgb8, (1300//4, 1030//4)))
                imageio.imwrite(filename, rgb8)

    if len(rgbs)>0:
        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, latent_embedding, num_latent_modules=1):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, args.input_dims)

    input_ch_views = 0
    embeddirs_fn = None

    latent_dim = 600 + args.latent_vec_size if args.use_emaps and num_latent_modules > 1 \
                                            else args.latent_vec_size * num_latent_modules

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, args.viewdirs_dims)

    # if args.use_shading_nw:
        embed_light_dirs_fn, input_light_dirs_ch = get_embedder(args.multires_light_dirs, args.i_embed, args.viewdirs_dims)
    # else:
    #     input_light_dirs_ch = 3

    output_ch = 5 if args.N_importance > 0 else 4
    skips = args.skips

    # model = MultiResLatentNeRF(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, D_l=args.sh_netdepth,
    #              latent_dim=args.latent_vec_size*num_latent_modules,
    #              input_ch_light=input_light_dirs_ch, fit=args.fit,
    #              output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
    #              use_viewdirs=args.use_viewdirs, use_shading_nw = args.use_shading_nw,
    #              use_custom_mapping=args.use_custom_mapping, use_sine=args.use_sine, N_rays=args.N_rand*args.N_samples).to(device)

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, D_l=args.sh_netdepth,
                 latent_dim=latent_dim,
                 input_ch_light=input_light_dirs_ch, fit=args.fit,
                 output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
                 use_viewdirs=args.use_viewdirs, use_shading_nw = args.use_shading_nw,
                 use_custom_mapping=args.use_custom_mapping, use_sine=args.use_sine, N_rays=args.N_rand*args.N_samples).to(device)
    grad_vars = list(model.parameters())
    if args.render_multi_gpu:
        model = nn.DataParallel(model)
    # print(model)
    model_fine = None
    if args.N_importance > 0:

        # model_fine = MultiResLatentNeRF(D=args.netdepth_fine, W=args.netwidth_fine,
        #                   input_ch=input_ch, D_l=args.sh_netdepth_fine,
        #                   latent_dim=args.latent_vec_size*num_latent_modules,
        #                   output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
        #                   input_ch_light=input_light_dirs_ch, fit=args.fit,
        #                   use_viewdirs=args.use_viewdirs, use_shading_nw = args.use_shading_nw,
        #                   use_custom_mapping=args.use_custom_mapping, use_sine=args.use_sine, N_rays=args.N_rand).to(device)

        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, D_l=args.sh_netdepth_fine,
                          latent_dim=latent_dim,
                          output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
                          input_ch_light=input_light_dirs_ch, fit=args.fit,
                          use_viewdirs=args.use_viewdirs, use_shading_nw = args.use_shading_nw,
                          use_custom_mapping=args.use_custom_mapping, use_sine=args.use_sine, N_rays=args.N_rand).to(device)
        grad_vars += list(model_fine.parameters())
    if args.render_multi_gpu:
        model_fine = nn.DataParallel(model_fine)
    # result = lambda args : expression(args)
    network_query_fn = lambda inputs, latent_vector, viewdirs, light_dirs, use_OLAT, network_fn: run_network(inputs, latent_vector,
                                                                viewdirs, light_dirs, fn=network_fn, use_OLAT=use_OLAT,
                                                                embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,embed_light_dirs_fn=embed_light_dirs_fn,
                                                                use_shading_nw=args.use_shading_nw, netchunk=args.netchunk)
    # Create optimizer

    # if args.use_shading_nw:
    #     # freeze weights view-linears network
    #     for name, param in model.views_linears.named_parameters():
    #         param.requires_grad = False
    #     for name, param in model_fine.views_linears.named_parameters():
    #         param.requires_grad = False
    #     # freeze weights rgb network
    #     for name, param in model.rgb_linear.named_parameters():
    #         param.requires_grad = False
    #     for name, param in model_fine.rgb_linear.named_parameters():
    #         param.requires_grad = False
        # for name, param in model.alpha_linear.named_parameters():
        #     param.requires_grad = False
        # for name, param in model_fine.alpha_linear.named_parameters():
        #     param.requires_grad = False

    geom_params = []; shading_params = []
    for name, params in model.named_parameters():
        shading_params.append(params) if 'shading' in name else geom_params.append(params)

    geom_fine_params = []; shading_fine_params = []
    for name, params in model_fine.named_parameters():
        shading_fine_params.append(params) if 'shading' in name else geom_fine_params.append(params)

    if (args.latent_lrate != args.lrate) or args.use_shading_nw:
        if args.use_custom_mapping:
            optimizer = torch.optim.Adam([
                {"params": latent_embedding.parameters(), "lr":args.latent_lrate, 'name':'latent_embedding'},
                {"params": model.parameters(), "lr":args.lrate, 'name':'coarse_network'},
                {"params": model_fine.parameters(), "lr":args.lrate, 'name':'fine_network'},
            ],
                lr=args.lrate, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam([
                {"params": latent_embedding.parameters(), "lr":args.latent_lrate},
                {"params": model.parameters(), "lr": args.lrate},
                {"params": model_fine.parameters(), "lr": args.lrate},
                # {"params": geom_params,"lr" : args.lrate},
                # {"params": geom_fine_params,"lr" : args.lrate},
                # {"params": shading_params,"lr" : args.lrate},
                # {"params": shading_fine_params,"lr" : args.lrate},
            ],
                lr=args.lrate, betas=(0.9, 0.999))
    else:
        grad_vars += list(latent_embedding.parameters())
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None' and not args.fit:
        ckpts = [args.ft_path]
    elif args.ft_path is None and args.ft_path!='None' and args.fit:
        ckpts = [os.path.join(basedir, expname,'geom_wts', f) for f in sorted(os.listdir(os.path.join(basedir, expname, 'geom_wts'))) if 'tar' in f]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    # print('Found ckpts', ckpts)
    ckpts = natsort.natsorted(ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if args.test:
            ckpt_path = ckpts[args.ckpt_idx]
        else:
            ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        if not (args.use_shading_nw or args.fit or args.test or args.use_env_embedding):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        if args.use_shading_nw or (args.test == True):
            # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16
            model_dict = model.state_dict()
            model_load_dict = {k:v for k,v in ckpt['network_fn_state_dict'].items() if k in model_dict}
            model_dict.update(model_load_dict)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            if args.use_shading_nw or (args.test == True):
                model_fine_dict = model_fine.state_dict()
                model_fine_load_dict = {k: v for k, v in ckpt['network_fine_state_dict'].items() if k in model_fine_dict}
                model_fine_dict.update(model_fine_load_dict)
                model_fine.load_state_dict(model_fine_dict)
            else:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, latent_vector_batch, use_OLAT, network_fn, network_query_fn, N_samples,
                retraw=False, lindisp=False, perturb=0., N_importance=0, network_fine=None,
                white_bkgd=False, raw_noise_std=0., verbose=False, pytest=False,
                use_light_dirs=False, viewdirs_dim=3, use_shading_nw=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each

    if ray_batch.shape[-1] == 14:
        light_dirs = ray_batch[:,11:14]
        if viewdirs_dim==3:
            viewdirs = ray_batch[:, 8:11]
        elif viewdirs_dim ==6:
            viewdirs = ray_batch[:, 8:14]
        else:
            viewdirs = None
    elif ray_batch.shape[-1] == 11:
        viewdirs = ray_batch[:,-3:]

    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    latent_vector_batch_expanded = latent_vector_batch[..., None, :].repeat(1, N_samples, 1)

    # if ray_batch.shape[-1] == 14:
        # pts_l = rays_l[...,None,:] * torch.ones_like(z_vals[...,:,None])
        # pts = torch.cat([pts, pts_l], -1)


    # raw = run_network(pts)
    raw = network_query_fn(pts, latent_vector_batch_expanded, viewdirs, light_dirs, use_OLAT, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, weights_0 = rgb_map, disp_map, acc_map, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        latent_vector_batch_expanded_fine = latent_vector_batch[..., None, :].repeat(1, N_samples+N_importance, 1)

        # if ray_batch.shape[-1] == 14:
        #     pts_l = rays_l[..., None, :] * torch.ones_like(z_vals[..., :, None])
        #     pts = torch.cat([pts, pts_l], -1)

        run_fn = network_fn if network_fine is None else network_fine
        # raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, latent_vector_batch_expanded_fine, viewdirs, light_dirs, use_OLAT, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['wt0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def init_latent_embedding(data_dir, args):
    """
    Embeddings for per number of Identities and OLATs
    """
    if args.num_train_IDs == 50:
        identities = glob.glob1(data_dir, 'ID000[0-4]*')
        identities.extend(glob.glob1(data_dir, 'ID0005[0-3]'))
    elif args.num_train_IDs == 100:
        identities = glob.glob1(data_dir, 'ID000[0-9]*')
        identities.extend(glob.glob1(data_dir, 'ID0010[0-3]'))
    elif args.num_train_IDs == -1:
        identities = natsort.natsorted(os.listdir(data_dir))
    else:
        raise NotImplementedError

    if len(args.aux_datadir) > 0:
        if len([dir for dir in args.aux_datadir if 'i3DMM' in dir]) > 0:
            identities.extend([os.listdir(dir) for dir in args.aux_datadir if 'i3DMM' in dir][0])

    map_elements = identities
    EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir,identities[0], 'images'), 'Cam07_*_EMAP*'))

    if len(EMAPS) == 0:
        EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir, identities[0], 'images'), 'Cam06_*_EMAP*'))
        if len(EMAPS) == 0:
            EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir, identities[0], 'images'), 'Cam08_*_EMAP*'))
            if len(EMAPS) == 0:
                EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir, identities[0], 'images'), 'Cam10_*_EMAP*'))
    # auxilary data
    if len(args.aux_datadir)>0:
        for aux_data_dir in args.aux_datadir:
            EMAPS.extend(natsort.natsorted(glob.glob1(os.path.join(aux_data_dir, identities[0], 'images'), 'Cam07_*_EMAP*')))

    if len(EMAPS)==0:
        EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir, identities[0], 'images'), 'Cam00_*_EMAP*'))
        if len(EMAPS)==0:
            EMAPS = natsort.natsorted(glob.glob1(os.path.join(data_dir, identities[0], 'images'), 'Cam01_*_EMAP*'))

    EMAPS = list(dict.fromkeys([s.split('_')[-1][0:8] for s in EMAPS]))
    if len(EMAPS)>5:
        print("Too Many environment maps! Selecting only {} outdoor environment maps".format(NUM_RELIT_EMAPs))
        if NUM_RELIT_EMAPs == 15:
            EMAPS = [
            'EMAP-000', 'EMAP-003', 'EMAP-012', 'EMAP-015', 'EMAP-028',
            'EMAP-030', 'EMAP-035', 'EMAP-043', 'EMAP-048', 'EMAP-054',
            'EMAP-061', 'EMAP-064', 'EMAP-075', 'EMAP-078', 'EMAP-081',
            'EMAP-999'  # testing
            ]
        elif NUM_RELIT_EMAPs == 2:
            EMAPS = [
                'EMAP-250', 'EMAP-900',

                'EMAP-999'
            ]
        elif NUM_RELIT_EMAPs == 50:
            EMAPS = [
                'EMAP-000', 'EMAP-001', 'EMAP-003', 'EMAP-006', 'EMAP-012',
                'EMAP-015', 'EMAP-018', 'EMAP-019', 'EMAP-023', 'EMAP-024', 'EMAP-025', 'EMAP-028',
                'EMAP-030', 'EMAP-035', 'EMAP-036', 'EMAP-043', 'EMAP-046', 'EMAP-047', 'EMAP-048',
                'EMAP-061', 'EMAP-064', 'EMAP-075', 'EMAP-078', 'EMAP-081', 'EMAP-086', 'EMAP-106',
                'EMAP-108', 'EMAP-110', 'EMAP-117', 'EMAP-128', 'EMAP-140',
                'EMAP-141', 'EMAP-144', 'EMAP-148', 'EMAP-157', 'EMAP-153', 'EMAP-154', 'EMAP-158',
                'EMAP-180', 'EMAP-182', 'EMAP-183', 'EMAP-184', 'EMAP-193', 'EMAP-194', 'EMAP-195',
                'EMAP-008', 'EMAP-010',
                'EMAP-114', 'EMAP-116',
                'EMAP-054',
                'EMAP-999',  # testing
            ]
        elif NUM_RELIT_EMAPs == 30:
            EMAPS = [
                'EMAP-860', 'EMAP-861', 'EMAP-862', 'EMAP-863', 'EMAP-864', 'EMAP-865', 'EMAP-866', 'EMAP-867',
                'EMAP-868', 'EMAP-869',
                'EMAP-870', 'EMAP-871', 'EMAP-872', 'EMAP-873', 'EMAP-874', 'EMAP-875', 'EMAP-876', 'EMAP-877',
                'EMAP-878', 'EMAP-879',
                'EMAP-880', 'EMAP-881', 'EMAP-882', 'EMAP-883', 'EMAP-884', 'EMAP-885', 'EMAP-886', 'EMAP-887',
                'EMAP-888', 'EMAP-889',
                'EMAP-999',  # testing
            ]
        elif NUM_RELIT_EMAPs == 100:
            EMAPS = [
                'EMAP-250', 'EMAP-251', 'EMAP-252', 'EMAP-253', 'EMAP-254', 'EMAP-255', 'EMAP-256', 'EMAP-257', 'EMAP-258', 'EMAP-259',
                'EMAP-260', 'EMAP-261', 'EMAP-262', 'EMAP-263', 'EMAP-264', 'EMAP-265', 'EMAP-266', 'EMAP-267', 'EMAP-268', 'EMAP-269',
                'EMAP-270', 'EMAP-271', 'EMAP-272', 'EMAP-273', 'EMAP-274', 'EMAP-275', 'EMAP-276', 'EMAP-277', 'EMAP-278', 'EMAP-279',
                'EMAP-280', 'EMAP-281', 'EMAP-282', 'EMAP-283', 'EMAP-284', 'EMAP-285', 'EMAP-286', 'EMAP-287', 'EMAP-288', 'EMAP-289',
                'EMAP-290', 'EMAP-291', 'EMAP-292', 'EMAP-293', 'EMAP-294', 'EMAP-295', 'EMAP-296', 'EMAP-297', 'EMAP-298', 'EMAP-299',
                'EMAP-999',  # testing
            ]
        elif NUM_RELIT_EMAPs == 253:
            EMAPS = [
                'EMAP-250', 'EMAP-251', 'EMAP-252', 'EMAP-253', 'EMAP-254', 'EMAP-255', 'EMAP-256', 'EMAP-257', 'EMAP-258', 'EMAP-259',
                'EMAP-260', 'EMAP-261', 'EMAP-262', 'EMAP-263', 'EMAP-264', 'EMAP-265', 'EMAP-266', 'EMAP-267', 'EMAP-268', 'EMAP-269',
                'EMAP-270', 'EMAP-271', 'EMAP-272', 'EMAP-273', 'EMAP-274', 'EMAP-275', 'EMAP-276', 'EMAP-277', 'EMAP-278', 'EMAP-279',
                'EMAP-280', 'EMAP-281', 'EMAP-282', 'EMAP-283', 'EMAP-284', 'EMAP-285', 'EMAP-286', 'EMAP-287', 'EMAP-288', 'EMAP-289',
                'EMAP-290', 'EMAP-291', 'EMAP-292', 'EMAP-293', 'EMAP-294', 'EMAP-295', 'EMAP-296', 'EMAP-297', 'EMAP-298', 'EMAP-299',

                'EMAP-350', 'EMAP-351', 'EMAP-352', 'EMAP-353', 'EMAP-354', 'EMAP-355', 'EMAP-356', 'EMAP-357', 'EMAP-358', 'EMAP-359',
                'EMAP-360', 'EMAP-361', 'EMAP-362', 'EMAP-363', 'EMAP-364', 'EMAP-365', 'EMAP-366', 'EMAP-367', 'EMAP-368', 'EMAP-369',
                'EMAP-370', 'EMAP-371', 'EMAP-372', 'EMAP-373', 'EMAP-374', 'EMAP-375', 'EMAP-376', 'EMAP-377', 'EMAP-378', 'EMAP-379',
                'EMAP-380', 'EMAP-381', 'EMAP-382', 'EMAP-383', 'EMAP-384', 'EMAP-385', 'EMAP-386', 'EMAP-387', 'EMAP-388', 'EMAP-389',
                'EMAP-390', 'EMAP-391', 'EMAP-392', 'EMAP-393', 'EMAP-394', 'EMAP-395', 'EMAP-396', 'EMAP-397', 'EMAP-398', 'EMAP-399',
                'EMAP-400', 'EMAP-401', 'EMAP-402', 'EMAP-403', 'EMAP-404', 'EMAP-405', 'EMAP-406', 'EMAP-407', 'EMAP-408', 'EMAP-409',
                'EMAP-410', 'EMAP-411', 'EMAP-412', 'EMAP-413', 'EMAP-414', 'EMAP-415', 'EMAP-416', 'EMAP-417',

                'EMAP-418', 'EMAP-419',
                'EMAP-420', 'EMAP-421', 'EMAP-422', 'EMAP-423', 'EMAP-424', 'EMAP-425', 'EMAP-426', 'EMAP-427', 'EMAP-428', 'EMAP-429',
                'EMAP-430', 'EMAP-431', 'EMAP-432', 'EMAP-433', 'EMAP-434', 'EMAP-435', 'EMAP-436', 'EMAP-437', 'EMAP-438', 'EMAP-439',
                'EMAP-440', 'EMAP-441', 'EMAP-442', 'EMAP-443', 'EMAP-444', 'EMAP-445', 'EMAP-446', 'EMAP-447', 'EMAP-448', 'EMAP-449',
                'EMAP-450', 'EMAP-451', 'EMAP-452', 'EMAP-453', 'EMAP-454', 'EMAP-455', 'EMAP-456', 'EMAP-457', 'EMAP-458', 'EMAP-459',
                'EMAP-460', 'EMAP-461', 'EMAP-462', 'EMAP-463', 'EMAP-464', 'EMAP-465', 'EMAP-466', 'EMAP-467', 'EMAP-468', 'EMAP-469',
                'EMAP-470', 'EMAP-471', 'EMAP-472', 'EMAP-473', 'EMAP-474', 'EMAP-475', 'EMAP-476', 'EMAP-477', 'EMAP-478', 'EMAP-479',
                'EMAP-480', 'EMAP-481', 'EMAP-482', 'EMAP-483', 'EMAP-484', 'EMAP-485',

                'EMAP-486', 'EMAP-487', 'EMAP-488', 'EMAP-489',
                'EMAP-490', 'EMAP-491', 'EMAP-492', 'EMAP-493', 'EMAP-494', 'EMAP-495', 'EMAP-496', 'EMAP-497', 'EMAP-498', 'EMAP-499',
                'EMAP-500', 'EMAP-501', 'EMAP-502', 'EMAP-503', 'EMAP-504', 'EMAP-505', 'EMAP-506', 'EMAP-507', 'EMAP-508', 'EMAP-509',
                'EMAP-510', 'EMAP-511', 'EMAP-512', 'EMAP-513', 'EMAP-514', 'EMAP-515', 'EMAP-516', 'EMAP-517', 'EMAP-518', 'EMAP-519',
                'EMAP-520', 'EMAP-521', 'EMAP-522', 'EMAP-523', 'EMAP-524', 'EMAP-525', 'EMAP-526', 'EMAP-527', 'EMAP-528', 'EMAP-529',
                'EMAP-530', 'EMAP-531', 'EMAP-532', 'EMAP-533', 'EMAP-534', 'EMAP-535', 'EMAP-536', 'EMAP-537', 'EMAP-538', 'EMAP-539',
                'EMAP-540', 'EMAP-541', 'EMAP-542', 'EMAP-543', 'EMAP-544', 'EMAP-545', 'EMAP-546', 'EMAP-547', 'EMAP-548', 'EMAP-549',
                'EMAP-550', 'EMAP-551', 'EMAP-552',
                'EMAP-553',

                'EMAP-900',

                'EMAP-999',  # testing
            ]
        elif NUM_RELIT_EMAPs == (676):
            EMAPS = [
                # Hard Shadows
                'EMAP-000', 'EMAP-001', 'EMAP-003', 'EMAP-006', 'EMAP-012', 'EMAP-015', 'EMAP-018', 'EMAP-019',
                'EMAP-023', 'EMAP-024',
                'EMAP-025', 'EMAP-028', 'EMAP-030', 'EMAP-035', 'EMAP-036', 'EMAP-043', 'EMAP-046', 'EMAP-047',
                'EMAP-048', 'EMAP-061',
                'EMAP-064', 'EMAP-075', 'EMAP-078', 'EMAP-081', 'EMAP-086', 'EMAP-106', 'EMAP-108', 'EMAP-110',
                'EMAP-117', 'EMAP-128',
                'EMAP-140', 'EMAP-141', 'EMAP-144', 'EMAP-148', 'EMAP-157', 'EMAP-153', 'EMAP-154', 'EMAP-158',
                'EMAP-180', 'EMAP-182',
                'EMAP-183', 'EMAP-184', 'EMAP-193', 'EMAP-194', 'EMAP-195', 'EMAP-008', 'EMAP-010', 'EMAP-114',
                'EMAP-116', 'EMAP-054',

                # Outdoor
                'EMAP-250', 'EMAP-251', 'EMAP-252', 'EMAP-253', 'EMAP-254', 'EMAP-255', 'EMAP-256', 'EMAP-257',
                'EMAP-258', 'EMAP-259',
                'EMAP-260', 'EMAP-261', 'EMAP-262', 'EMAP-263', 'EMAP-264', 'EMAP-265', 'EMAP-266', 'EMAP-267',
                'EMAP-268', 'EMAP-269',
                'EMAP-270', 'EMAP-271', 'EMAP-272', 'EMAP-273', 'EMAP-274', 'EMAP-275', 'EMAP-276', 'EMAP-277',
                'EMAP-278', 'EMAP-279',
                'EMAP-280',

                'EMAP-281', 'EMAP-282', 'EMAP-283', 'EMAP-284', 'EMAP-285', 'EMAP-286', 'EMAP-287', 'EMAP-288',
                'EMAP-289',
                'EMAP-290', 'EMAP-291', 'EMAP-292', 'EMAP-293', 'EMAP-294', 'EMAP-295', 'EMAP-296', 'EMAP-297',
                'EMAP-298', 'EMAP-299',
                'EMAP-300', 'EMAP-301', 'EMAP-302', 'EMAP-303', 'EMAP-304', 'EMAP-305', 'EMAP-306', 'EMAP-307',
                'EMAP-308', 'EMAP-309',
                'EMAP-310', 'EMAP-311',

                'EMAP-312', 'EMAP-313', 'EMAP-314', 'EMAP-315', 'EMAP-316', 'EMAP-317', 'EMAP-318', 'EMAP-319',
                'EMAP-320', 'EMAP-321', 'EMAP-322', 'EMAP-323', 'EMAP-324', 'EMAP-325', 'EMAP-326', 'EMAP-327',
                'EMAP-328', 'EMAP-329',
                'EMAP-330', 'EMAP-331', 'EMAP-332', 'EMAP-333', 'EMAP-334', 'EMAP-335', 'EMAP-336', 'EMAP-337',
                'EMAP-338', 'EMAP-339',
                'EMAP-340', 'EMAP-341',

                # indoor
                'EMAP-350', 'EMAP-351', 'EMAP-352', 'EMAP-353', 'EMAP-354', 'EMAP-355', 'EMAP-356', 'EMAP-357',
                'EMAP-358', 'EMAP-359',
                'EMAP-360', 'EMAP-361', 'EMAP-362', 'EMAP-363', 'EMAP-364', 'EMAP-365', 'EMAP-366', 'EMAP-367',
                'EMAP-368', 'EMAP-369',
                'EMAP-370', 'EMAP-371', 'EMAP-372', 'EMAP-373', 'EMAP-374', 'EMAP-375', 'EMAP-376', 'EMAP-377',
                'EMAP-378', 'EMAP-379',
                'EMAP-380', 'EMAP-381', 'EMAP-382', 'EMAP-383', 'EMAP-384', 'EMAP-385', 'EMAP-386', 'EMAP-387',
                'EMAP-388', 'EMAP-389',
                'EMAP-390', 'EMAP-391', 'EMAP-392', 'EMAP-393', 'EMAP-394', 'EMAP-395', 'EMAP-396', 'EMAP-397',
                'EMAP-398', 'EMAP-399',
                'EMAP-400', 'EMAP-401', 'EMAP-402', 'EMAP-403', 'EMAP-404', 'EMAP-405', 'EMAP-406', 'EMAP-407',
                'EMAP-408', 'EMAP-409',
                'EMAP-410', 'EMAP-411', 'EMAP-412', 'EMAP-413', 'EMAP-414', 'EMAP-415', 'EMAP-416', 'EMAP-417',

                'EMAP-418', 'EMAP-419',
                'EMAP-420', 'EMAP-421', 'EMAP-422', 'EMAP-423', 'EMAP-424', 'EMAP-425', 'EMAP-426', 'EMAP-427',
                'EMAP-428', 'EMAP-429',
                'EMAP-430', 'EMAP-431', 'EMAP-432', 'EMAP-433', 'EMAP-434', 'EMAP-435', 'EMAP-436', 'EMAP-437',
                'EMAP-438', 'EMAP-439',
                'EMAP-440', 'EMAP-441', 'EMAP-442', 'EMAP-443', 'EMAP-444', 'EMAP-445', 'EMAP-446', 'EMAP-447',
                'EMAP-448', 'EMAP-449',
                'EMAP-450', 'EMAP-451', 'EMAP-452', 'EMAP-453', 'EMAP-454', 'EMAP-455', 'EMAP-456', 'EMAP-457',
                'EMAP-458', 'EMAP-459',
                'EMAP-460', 'EMAP-461', 'EMAP-462', 'EMAP-463', 'EMAP-464', 'EMAP-465', 'EMAP-466', 'EMAP-467',
                'EMAP-468', 'EMAP-469',
                'EMAP-470', 'EMAP-471', 'EMAP-472', 'EMAP-473', 'EMAP-474', 'EMAP-475', 'EMAP-476', 'EMAP-477',
                'EMAP-478', 'EMAP-479',
                'EMAP-480', 'EMAP-481', 'EMAP-482', 'EMAP-483', 'EMAP-484', 'EMAP-485',

                'EMAP-486', 'EMAP-487', 'EMAP-488', 'EMAP-489',
                'EMAP-490', 'EMAP-491', 'EMAP-492', 'EMAP-493', 'EMAP-494', 'EMAP-495', 'EMAP-496', 'EMAP-497',
                'EMAP-498', 'EMAP-499',
                'EMAP-500', 'EMAP-501', 'EMAP-502', 'EMAP-503', 'EMAP-504', 'EMAP-505', 'EMAP-506', 'EMAP-507',
                'EMAP-508', 'EMAP-509',
                'EMAP-510', 'EMAP-511', 'EMAP-512', 'EMAP-513', 'EMAP-514', 'EMAP-515', 'EMAP-516', 'EMAP-517',
                'EMAP-518', 'EMAP-519',
                'EMAP-520', 'EMAP-521', 'EMAP-522', 'EMAP-523', 'EMAP-524', 'EMAP-525', 'EMAP-526', 'EMAP-527',
                'EMAP-528', 'EMAP-529',
                'EMAP-530', 'EMAP-531', 'EMAP-532', 'EMAP-533', 'EMAP-534', 'EMAP-535', 'EMAP-536', 'EMAP-537',
                'EMAP-538', 'EMAP-539',
                'EMAP-540', 'EMAP-541', 'EMAP-542', 'EMAP-543', 'EMAP-544', 'EMAP-545', 'EMAP-546', 'EMAP-547',
                'EMAP-548', 'EMAP-549',
                'EMAP-550', 'EMAP-551', 'EMAP-552', 'EMAP-553',

                # Indoor Extra
                'EMAP-560', 'EMAP-561', 'EMAP-562', 'EMAP-563', 'EMAP-564', 'EMAP-565', 'EMAP-566', 'EMAP-567',
                'EMAP-568', 'EMAP-569',
                'EMAP-570', 'EMAP-571', 'EMAP-572', 'EMAP-573', 'EMAP-574', 'EMAP-575', 'EMAP-576', 'EMAP-577',
                'EMAP-578', 'EMAP-579',
                'EMAP-580', 'EMAP-581', 'EMAP-582', 'EMAP-583', 'EMAP-584', 'EMAP-585', 'EMAP-586', 'EMAP-587',
                'EMAP-588', 'EMAP-589',
                'EMAP-590', 'EMAP-591', 'EMAP-592', 'EMAP-593', 'EMAP-594', 'EMAP-595', 'EMAP-596', 'EMAP-597',
                'EMAP-598', 'EMAP-599',

                'EMAP-600', 'EMAP-601', 'EMAP-602', 'EMAP-603', 'EMAP-604', 'EMAP-605', 'EMAP-606', 'EMAP-607',
                'EMAP-608', 'EMAP-609',
                'EMAP-610', 'EMAP-611', 'EMAP-612', 'EMAP-613', 'EMAP-614', 'EMAP-615', 'EMAP-616', 'EMAP-617',
                'EMAP-618', 'EMAP-619',
                'EMAP-620', 'EMAP-621', 'EMAP-622', 'EMAP-623', 'EMAP-624', 'EMAP-625', 'EMAP-626', 'EMAP-627',
                'EMAP-628', 'EMAP-629',
                'EMAP-630', 'EMAP-631', 'EMAP-632', 'EMAP-633', 'EMAP-634', 'EMAP-635', 'EMAP-636', 'EMAP-637',
                'EMAP-638', 'EMAP-639',
                'EMAP-640', 'EMAP-641', 'EMAP-642', 'EMAP-643', 'EMAP-644', 'EMAP-645', 'EMAP-646', 'EMAP-647',
                'EMAP-648', 'EMAP-649',
                'EMAP-650', 'EMAP-651', 'EMAP-652', 'EMAP-653', 'EMAP-654', 'EMAP-655', 'EMAP-656', 'EMAP-657',
                'EMAP-658', 'EMAP-659',
                'EMAP-660', 'EMAP-661', 'EMAP-662', 'EMAP-663', 'EMAP-664', 'EMAP-665', 'EMAP-666', 'EMAP-667',
                'EMAP-668', 'EMAP-669',
                'EMAP-670', 'EMAP-671', 'EMAP-672', 'EMAP-673', 'EMAP-674', 'EMAP-675', 'EMAP-676', 'EMAP-677',
                'EMAP-678', 'EMAP-679',
                'EMAP-680', 'EMAP-681', 'EMAP-682', 'EMAP-683', 'EMAP-684', 'EMAP-685', 'EMAP-686', 'EMAP-687',
                'EMAP-688', 'EMAP-689',
                'EMAP-690', 'EMAP-691', 'EMAP-692', 'EMAP-693', 'EMAP-694', 'EMAP-695', 'EMAP-696', 'EMAP-697',
                'EMAP-698', 'EMAP-699',

                'EMAP-700', 'EMAP-701', 'EMAP-702', 'EMAP-703', 'EMAP-704', 'EMAP-705', 'EMAP-706', 'EMAP-707',
                'EMAP-708', 'EMAP-709',
                'EMAP-710', 'EMAP-711', 'EMAP-712', 'EMAP-713', 'EMAP-714', 'EMAP-715', 'EMAP-716', 'EMAP-717',
                'EMAP-718', 'EMAP-719',
                'EMAP-720', 'EMAP-721', 'EMAP-722', 'EMAP-723', 'EMAP-724', 'EMAP-725', 'EMAP-726', 'EMAP-727',
                'EMAP-728', 'EMAP-729',
                'EMAP-730', 'EMAP-731', 'EMAP-732', 'EMAP-733', 'EMAP-734', 'EMAP-735', 'EMAP-736', 'EMAP-737',
                'EMAP-738', 'EMAP-739',
                'EMAP-740', 'EMAP-741', 'EMAP-742', 'EMAP-743', 'EMAP-744', 'EMAP-745', 'EMAP-746', 'EMAP-747',
                'EMAP-748', 'EMAP-749',
                'EMAP-750', 'EMAP-751', 'EMAP-752', 'EMAP-753', 'EMAP-754', 'EMAP-755', 'EMAP-756', 'EMAP-757',
                'EMAP-758', 'EMAP-759',
                'EMAP-760', 'EMAP-761', 'EMAP-762', 'EMAP-763', 'EMAP-764', 'EMAP-765', 'EMAP-766', 'EMAP-767',
                'EMAP-768', 'EMAP-769',
                'EMAP-770', 'EMAP-771', 'EMAP-772', 'EMAP-773', 'EMAP-774', 'EMAP-775', 'EMAP-776', 'EMAP-777',
                'EMAP-778', 'EMAP-779',
                'EMAP-780', 'EMAP-781', 'EMAP-782', 'EMAP-783', 'EMAP-784', 'EMAP-785', 'EMAP-786', 'EMAP-787',
                'EMAP-788', 'EMAP-789',
                'EMAP-790', 'EMAP-791', 'EMAP-792', 'EMAP-793', 'EMAP-794', 'EMAP-795', 'EMAP-796', 'EMAP-797',
                'EMAP-798', 'EMAP-799',

                'EMAP-800', 'EMAP-801', 'EMAP-802', 'EMAP-803', 'EMAP-804', 'EMAP-805', 'EMAP-806', 'EMAP-807',
                'EMAP-808', 'EMAP-809',
                'EMAP-810', 'EMAP-811', 'EMAP-812', 'EMAP-813', 'EMAP-814', 'EMAP-815', 'EMAP-816', 'EMAP-817',
                'EMAP-818', 'EMAP-819',
                'EMAP-820', 'EMAP-821', 'EMAP-822', 'EMAP-823', 'EMAP-824', 'EMAP-825', 'EMAP-826', 'EMAP-827',
                'EMAP-828', 'EMAP-829',
                'EMAP-830', 'EMAP-831', 'EMAP-832', 'EMAP-833', 'EMAP-834', 'EMAP-835', 'EMAP-836', 'EMAP-837',
                'EMAP-838', 'EMAP-839',
                'EMAP-840', 'EMAP-841', 'EMAP-842', 'EMAP-843', 'EMAP-844', 'EMAP-845', 'EMAP-846', 'EMAP-847',
                'EMAP-848', 'EMAP-849',
                'EMAP-850', 'EMAP-851', 'EMAP-852', 'EMAP-853', 'EMAP-854', 'EMAP-855', 'EMAP-856', 'EMAP-857',
                'EMAP-858', 'EMAP-859',
                'EMAP-860', 'EMAP-861', 'EMAP-862', 'EMAP-863', 'EMAP-864', 'EMAP-865', 'EMAP-866', 'EMAP-867',
                'EMAP-868', 'EMAP-869',
                'EMAP-870', 'EMAP-871', 'EMAP-872', 'EMAP-873', 'EMAP-874', 'EMAP-875', 'EMAP-876', 'EMAP-877',
                'EMAP-878', 'EMAP-879',
                'EMAP-880', 'EMAP-881', 'EMAP-882', 'EMAP-883', 'EMAP-884', 'EMAP-885', 'EMAP-886', 'EMAP-887',
                'EMAP-888', 'EMAP-889',

                'EMAP-999',  # testing
            ]

    # MAP Identities + Env Maps
    if not args.use_single_embedding:
        if len(EMAPS) > 0:
            map_ID = torch.zeros([len(identities)+len(EMAPS), 2], dtype=torch.int64)
            map_elements = map_elements + EMAPS
        else:
            map_ID = torch.zeros([len(identities), 1], dtype=torch.int64)

        for i, val in enumerate(map_elements):
            if 'ID' in val:
                idx_ID = identities.index(val)
                map_ID[i][0] = idx_ID  # 0 for Identity embedding
            elif 'EMAP' in val:
                idx_EMAP = EMAPS.index(val)
                map_ID[i][1] = len(identities)+idx_EMAP #1 for Environment Map embedding
    else:
        if len(EMAPS) > 0:
            map_ID = torch.zeros([(len(identities))*(len(EMAPS)), 1], dtype=torch.int64)
            map_elements = list(itertools.product(identities,EMAPS))
        else:
            map_ID = torch.zeros([len(identities), 1], dtype=torch.int64)

        for i, val in enumerate(map_elements):
            if 'ID' in val[0]:
                idx_ID = identities.index(val[0])+1 # map_ID does not start with zero!
                if 'EMAP' in val[1]:
                    idx_EMAP = EMAPS.index(val[1])+1
                    map_ID[i][0] = idx_ID * idx_EMAP  # 0 for Identity*Enviroment Map embedding

    num_latent_vecs = len(identities)+len(EMAPS) if not args.use_single_embedding else (len(identities)) * (len(EMAPS))

    return identities, EMAPS, num_latent_vecs , map_ID, len(identities), len(EMAPS)

def get_latent_vector(identities, EMAPs, meta_data, embedding, scan_id, image_names='', N_rays=8, args=None):
    latent_vector = None
    N_images = len(scan_id) # useful if batch size > 1

    if args.use_emaps:
        # NOTE: identities contains Identites + ENV Maps in this mode
        # if len(EMAPs) > 0 and args.use_latent_emap:
        #     inds = torch.ones(2 * N_images * N_rays, dtype=torch.long)
        #     step=2
        # else:
        inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)
        step=1

        for i in range(N_images):
            if len(EMAPs)>0:
                idx_ID = identities.index(scan_id[i])
                # emap_vector = np.load(os.path.join(args.emap_dir, image_names[i].split('_')[-1][0:8]+'.npy'))
                emap_vector = cv2.imread(os.path.join(args.emap_dir, image_names[i].split('_')[-1][0:8]+'.exr'), -1)[:,:,::-1]
                emap_vector = emap_vector.reshape(-1)
                emap_vector = emap_vector/emap_vector.max()
                assert emap_vector.shape[0] == 600, emap_vector.shape

                emap_vector = torch.tensor(np.tile(emap_vector, (N_rays, 1)))
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays*step:(i+1)*N_rays*step] = tt
            else:
                idx_ID = identities.index(scan_id[i])
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays:(i+1)*N_rays*step] = tt
    else:
        if (len(EMAPs) > 0) and not args.use_single_embedding:
            inds = torch.ones(2 * N_images * N_rays, dtype=torch.long)
        else:
            inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)

        for i in range(N_images):
            if len(EMAPs)>0:
                # idx_ID = identities.index(scan_id[i])
                # if test_EMAPs is None:
                # idx_EMAP = len(identities)+EMAPs.index(image_names[i].split('_')[-1][0:8])
                # else:
                #     idx_EMAP = len(identities) + EMAPs.index(test_EMAPs)
                if args.use_single_embedding:
                    idx_ID = identities.index(scan_id[i]) + 1
                    idx_EMAP = EMAPs.index(image_names[i].split('_')[-1][0:8]) + 1
                    tt = torch.Tensor([meta_data[(idx_ID)*(idx_EMAP) - 1][0]]).type(torch.long).repeat(N_rays)
                    inds[i*N_rays*2:(i+1)*N_rays*2] = tt
                    inds = inds if len(identities) > 1 else inds - 1 # fix to start index from 0 instead of 1
                else:
                    idx_ID = identities.index(scan_id[i])
                    idx_EMAP = len(identities) + EMAPs.index(image_names[i].split('_')[-1][0:8])
                    tt = torch.Tensor([meta_data[idx_ID][0], meta_data[idx_EMAP][1]]).type(torch.long).repeat(N_rays)
                    inds[i*N_rays*2:(i+1)*N_rays*2] = tt
            else:
                idx_ID = identities.index(scan_id[i])
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays:(i+1)*N_rays] = tt

    inds = inds.to(device)
    tt = embedding(inds)

    # if args.use_emaps and not args.use_latent_emap:
    if args.use_emaps:
        identity_vector = tt.view(N_images * N_rays, -1)
        latent_vector = torch.cat([identity_vector, emap_vector], -1)
    else:
        latent_vector = tt.view(N_images * N_rays, -1)

    return latent_vector

def get_latent_vector_fit(identities, EMAPs, meta_data, latent_embedding, emap_embedding,
                          scan_id, image_names='', N_rays=8, args=None):
    latent_vector = None
    N_images = len(scan_id) # useful if batch size > 1

    inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)
    step = 1

    for i in range(N_images):
        if len(EMAPs)>0:
            idx_ID = identities.index(scan_id[i])
            tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
            inds[i * N_rays * step:(i + 1) * N_rays * step] = tt

            emap_vector = cv2.imread(os.path.join(args.datadir, identities[0] , 'EMAP-999.exr'), -1)[:, :,::-1]
            emap_vector = emap_vector.reshape(-1)
            emap_vector = emap_vector / emap_vector.max()
            assert emap_vector.shape[0] == 600, emap_vector.shape

            emap_vector = torch.tensor(np.tile(emap_vector, (N_rays, 1)))

        else:
            idx_ID = identities.index(scan_id[i])
            tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
            inds[i*N_rays:(i+1)*N_rays] = tt

    inds = inds.to(device)
    tt = latent_embedding(inds)

    if args.use_emaps and not args.use_latent_emap:
        identity_vector = tt.view(N_images * N_rays, -1)
        latent_vector = torch.cat([identity_vector, emap_vector], -1)
    elif args.use_emaps and args.use_latent_emap:
        identity_vector = tt.view(N_images * N_rays, -1)
        inds_emap = torch.zeros_like(inds)
        tt_emap = emap_embedding(inds_emap)

        emap_vector_delta = tt_emap.view(N_images*N_rays, -1)
        if args.test or args.fit:
            emap_vector = emap_vector + emap_vector_delta
        else:
            emap_vector = emap_vector_delta

        latent_vector = torch.cat([identity_vector, emap_vector], -1)
    else:
        raise NotImplementedError

    return latent_vector

def get_latent_vector_random(identities, EMAPs, meta_data, embedding, scan_id, image_names='', N_rays=8, use_env_embedding=False):
    latent_vector = None
    N_images = len(scan_id) # useful if batch size > 1
    # import pdb; pdb.set_trace()
    if len(EMAPs) > 0:
        inds = torch.ones(2 * N_images * N_rays, dtype=torch.long)
    else:
        inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)

    for i in range(N_images):
        if len(EMAPs)>0:
            idx_ID = identities.index(scan_id[i])
            idx_EMAP =len(identities) +  torch.randperm(len(EMAPs))[0].item()
            tt = torch.Tensor([meta_data[idx_ID][0], meta_data[idx_EMAP][1]]).type(torch.long).repeat(N_rays)
            inds[i*N_rays*2:(i+1)*N_rays*2] = tt
        else:
            idx_ID = identities.index(scan_id[i])
            tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
            inds[i*N_rays:(i+1)*N_rays] = tt

    inds = inds.to(device)
    tt = embedding(inds)
    latent_vector = tt.view(N_images*N_rays, -1)
    return latent_vector

def get_latent_vector_test(identities, EMAPs, meta_data, embedding, scan_id, image_names='', N_rays=8, test_EMAPs=None, args=None):
    latent_vector = None
    N_images = len(scan_id) # useful if batch size > 1

    if args.use_emaps:
        # NOTE: identities contains Identites + ENV Maps in this mode
        if len(EMAPs) > 0 and args.use_latent_emap:
            inds = torch.ones(2 * N_images * N_rays, dtype=torch.long)
            step=2
        else:
            inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)
            step=1

        for i in range(N_images):
            if len(EMAPs)>0:
                idx_ID = identities.index(scan_id[i])
                if test_EMAPs is None:
                    emap_vector = np.load(os.path.join(args.emap_dir, image_names[i].split('_')[-1][0:8]+'.npy'))
                else:
                    print(test_EMAPs)
                    # emap_vector = np.load(os.path.join(args.emap_dir, test_EMAPs+'.npy'))
                    emap_vector = cv2.imread(os.path.join(args.emap_dir, test_EMAPs), -1)[:, :, ::-1]
                    emap_vector = emap_vector.reshape(-1)
                    emap_vector = emap_vector / emap_vector.max()
                    assert emap_vector.shape[0] == 600, emap_vector.shape
                    # emap_vector = np.roll(emap_vector, -3)

                emap_vector = torch.tensor(np.tile(emap_vector, (N_rays, 1)))
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays*step:(i+1)*N_rays*step] = tt
            else:
                idx_ID = identities.index(scan_id[i])
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays:(i+1)*N_rays*step] = tt
    else:
        if (len(EMAPs) > 0) and not args.use_single_embedding:
        # if len(EMAPs) > 0:
            inds = torch.ones(2 * N_images * N_rays, dtype=torch.long)
        else:
            inds = torch.ones(1 * N_images * N_rays, dtype=torch.long)

        for i in range(N_images):
            if len(EMAPs)>0:
                if args.use_single_embedding:
                    idx_ID = identities.index(scan_id[i])
                    if test_EMAPs is None:
                        idx_EMAP = EMAPs.index(image_names[i].split('_')[1][0:8])
                    else:
                        idx_EMAP = EMAPs.index(test_EMAPs)
                    tt = torch.Tensor([meta_data[idx_ID] * meta_data[idx_EMAP]]).type(torch.long).repeat(N_rays)
                    inds[i * N_rays * 2:(i + 1) * N_rays * 2] = tt
                    inds = inds if len(identities) > 1 else inds - 1 # fix to start index from 0 instead of 1
                else:
                    idx_ID = identities.index(scan_id[i])
                    if test_EMAPs is None:
                        idx_EMAP =len(identities)+EMAPs.index(image_names[i].split('_')[1][0:8])
                    else:
                        idx_EMAP = len(identities) + EMAPs.index(test_EMAPs)
                    tt = torch.Tensor([meta_data[idx_ID][0], meta_data[idx_EMAP][1]]).type(torch.long).repeat(N_rays)
                    inds[i*N_rays*2:(i+1)*N_rays*2] = tt
            else:
                idx_ID = identities.index(scan_id[i])
                tt = torch.Tensor([meta_data[idx_ID][0]]).type(torch.long).repeat(N_rays)
                inds[i*N_rays:(i+1)*N_rays] = tt

    inds = inds.to(device)
    tt = embedding(inds)

    if args.use_emaps and not args.use_latent_emap:
        identity_vector = tt.view(N_images * N_rays, -1)
        latent_vector = torch.cat([identity_vector, emap_vector], -1)
    else:
        latent_vector = tt.view(N_images * N_rays, -1)

    # import pdb; pdb.set_trace()
    # TODO: Add new function for random latent vectors
    # mean = torch.mean(embedding.weight.data[-1], dim=0)
    # std = torch.std(embedding.weight.data[-1], dim=0)
    # random_latent_vector = torch.normal(mean, std).repeat(N_rays*N_images, 1)
    # latent_vector[:,8:] = random_latent_vector

    return latent_vector

def get_latent_vector_mean(identities, EMAPs, meta_data, embedding, scan_id, image_names='', N_rays=8, test=False):
    latent_vector = None
    N_images = len(scan_id) # useful if batch size > 1
    N = len(EMAPs)*N_images
    inds = torch.ones(2 * N, dtype=torch.long)

    for i in range(N_images):
        for j, EMAP in enumerate(EMAPs):
            idx_ID = identities.index(scan_id[i])
            idx_EMAP =len(identities)+EMAPs.index(EMAP)
            tt = torch.Tensor([meta_data[idx_ID][0], meta_data[idx_EMAP][1]]).type(torch.long)
            inds[i*N + j*2: i*N + (j+1)*2] = tt

    inds = inds.to(device)
    tt = embedding(inds)
    latent_vector = tt.view(N_images, len(EMAPs),  -1)
    latent_vector = latent_vector.mean(1)
    latent_vector = torch.repeat_interleave(latent_vector, N_rays, 0)
    return latent_vector

def load_latent_vectors(filename, lat_vecs):
    if not os.path.isfile(filename):
        raise Exception('latent state file "{}" does not exist'.format(filename))

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):
        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"]['weight'].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"]['weight'].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"]['weight'].size()[1]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]['weight']):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])
    return data["epoch"]

def save_latent_vectors(experiment_directory, latent_vector, epoch):
    latent_codes_dir = os.path.join(experiment_directory, 'lat_codes')
    if not os.path.exists(latent_codes_dir):
        os.makedirs(latent_codes_dir)
    all_latents = latent_vector.state_dict()
    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, str(epoch).zfill(6)+'.npy'))

def save_emap_vectors(experiment_directory, latent_vector, epoch):
    latent_codes_dir = os.path.join(experiment_directory, 'emap_codes')
    if not os.path.exists(latent_codes_dir):
        os.makedirs(latent_codes_dir)
    all_latents = latent_vector.state_dict()
    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, str(epoch).zfill(6)+'.npy'))

def get_near_far_bounds(datadir, poses_bounds_fn):
    near = float('inf')
    far = float('-inf')
    for basedir in glob.glob(datadir+'/*'):
        poses_arr = np.load(os.path.join(basedir, poses_bounds_fn))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        # print(bds[0].min(), bds[1].max())
        if near > bds[0].min():
            near = bds[0].min()
        if far < bds[1].max():
            far = bds[1].max()

    return near, far


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--decay_rate", type=float, default=0.50,
                        help='multiplication factor for learning rate decay rate')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--input_dims", type=int, default=3,
                        help='number of input dimension, changes when we condition input apart from RGB')
    parser.add_argument("--viewdirs_dims", type=int, default=3,
                        help='number of viewdirs input dimension, changes when we condition input apart from RGB')


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_single",   action='store_true',
                        help='Render only one pose and no video')
    parser.add_argument("--render_all", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the all the train views')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_multi_gpu",   action='store_true',
                        help='Render a multi-gpu trained model')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--n_iterations", type=int, default=100000,
                        help='Number of interations')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--num_relit_emaps", type=int, default=30,
                        help='number of relit env maps')
    parser.add_argument("--num_train_IDs", type=int, default=-1,
                        help='Total number of training identities')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images and -1 for full size')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--use_mat", action='store_true',
                        help='Load poses and bounds from .mat file instead of .npy')
    parser.add_argument("--no_shuffle_rays", action='store_true',
                        help='Shuffle rays before training. Turn off for debugging')
    parser.add_argument("--tone_mapping", action='store_true',
                        help='Perform tone mapping to the images')
    parser.add_argument("--train_all", action='store_true',
                        help='Use all images for training. Important to see how interpolation between train works')
    parser.add_argument("--use_light_dirs",   action='store_true',
                        help='Use OLAT direction')
    parser.add_argument("--light_idx",  nargs='+', default=[0], type=int,
                        help='Use OLAT direction')
    parser.add_argument("--load_mask", action='store_true',
                        help='read alpha channel for input images')
    parser.add_argument("--sample_mask", action='store_true',
                        help='use alpha channel for importance sampling of foreground')
    parser.add_argument("--sample_mask_prob", type=float, default=0.1,
                        help='percentage of background pixels. Set to 1.0 for using full image, 0 for using only foreground')
    parser.add_argument("--acc_loss", action='store_true',
                        help='use alpha channel for opacity loss on background region')
    parser.add_argument("--acc_loss_weight", type=float, default=0.05,
                        help='use alpha channel for opacity loss on background region')
    parser.add_argument("--bds_min_scale", type=float, default=1.0,
                        help='scale factor to extend min bounds')
    parser.add_argument("--bds_max_scale", type=float, default=1.0,
                        help='scale factor to extend max bounds')
    parser.add_argument("--use_hard_loss", action='store_true',
                        help='model beta distribution prior')
    parser.add_argument("--hard_loss_weight", type=float, default=0.1,
                        help='')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=40000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--recenter_poses", action='store_true', default=False,
                        help='recenter poses')
    # Batch training
    parser.add_argument("--poses_bounds_fn", type=str, default='poses_bounds.npy',
                        help='poses_bounds_fn')
    parser.add_argument("--test", action='store_true', default=False,
                        help="Switch to test mode instead of traning")
    parser.add_argument("--batch_size", type=int,default=1,
                        help='set batch size for training ')
    parser.add_argument("--num_workers", type=int,default=0,
                        help='set batch size for training ')
    parser.add_argument("--use_perceptual_loss", action='store_true', default=False,
                        help="Use a patch wise perceptual loss")
    parser.add_argument("--perceptual_loss_weight", type=float, default=0.01,
                        help='')
    parser.add_argument("--crop_dim", type=int, default=32,
                        help='dimensions of square crops for perceptual loss')
    parser.add_argument("--perceptual_loss_prob", default=0.5, type=float,
                        help="frequency of perceptual loss")
    parser.add_argument("--aux_datadir",  nargs='+', default=[], type=str,
                        help='Additional data directory')


    #SIREN and Custom Mapping Layer
    parser.add_argument("--use_custom_mapping", action='store_true', default=False,
                        help="Switch to test mode instead of traning")
    parser.add_argument("--use_sine", action='store_true', default=False,
                        help="Use Sine activation instead of ReLU")
    parser.add_argument("--skips",  nargs='+', default=[4], type=int,
                        help='Skip connections positions to NeRF')
    # Latent Embeddings
    parser.add_argument("--num_latent_vecs", type=int,
                        help='Number of latent vectors')
    parser.add_argument("--latent_vec_size", type=int, default=8,
                        help='latent embedding vector size')
    parser.add_argument("--latent_lrate", type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument("--use_code_regularisation", action='store_true', default=False,
                        help="Regularize the to avoid divergence of latent embeddings" )
    parser.add_argument("--reg_loss_weight", type=float, default=0.01,
                        help='')
    parser.add_argument("--use_env_embedding", action='store_true', default=False,
                        help="use a new embedding to learn different environment maps" )
    parser.add_argument("--use_emaps", action='store_true', default=False,
                        help="Use raw downscaled environment maps for relighting" )
    parser.add_argument("--use_latent_emap", action='store_true', default=False,
                        help="Use raw downscaled environment maps for relighting" )
    parser.add_argument("--emap_dir", type=str, default='//winfs-inf/HPS/prao2/static00/datasets/envmaps',
                        help='path to downsampled env maps')
    parser.add_argument("--use_single_embedding", action='store_true', default=False,
                        help="use a single latent code for identity + illuminaition " )


    #Shading Network
    parser.add_argument("--use_shading_nw", action='store_true', default=False,
                        help="Use Shading network to handle OLATs")
    parser.add_argument("--multires_light_dirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D OLAT direction)')
    parser.add_argument("--sh_netdepth", type=int, default=3,
                        help='layers in shading network')
    parser.add_argument("--sh_netdepth_fine", type=int, default=3,
                        help='layers in shading fine network')
    parser.add_argument("--switch_prob", type=float, default=0.05,
                        help='Switch between shading network and geometry network')
    parser.add_argument("--OLAT_config", type=str, default='',
                        help='Use different number of OLAT images or different data configurations indicated by a suffix')
    parser.add_argument("--use_HDR_loss", action='store_true',
                        help='HDR loss for OLAT images')
    # Fitting Unseen Identity
    parser.add_argument("--fit", action='store_true', default=False,
                        help="Flag to select fitting to unseen identity")

    parser.add_argument("--ckpt_idx", type=int, default=-1,
                        help='select checkpoint index')

    return parser


def train(args):
    global NUM_RELIT_EMAPs
    NUM_RELIT_EMAPs = args.num_relit_emaps

    train_dataset = ImagesDataset(args, args.datadir, args.aux_datadir, train=True, device=device, prefix='images')
    use_shuffle = True if len(train_dataset) > 2 else False
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=use_shuffle,
                                  num_workers=args.num_workers, drop_last=True)
    geom_prefetcher = data_prefetcher(train_dataloader)
    use_OLAT = False
    if args.fit:
        assert args.lrate == 0, "Learning rate should be zero during fitting"

    # if (args.use_shading_nw and not args.fit):
    if (args.use_shading_nw and not args.fit and args.switch_prob < 1):
        print("Initializing OLAT Dataloader ..")
        shading_dataset = ImagesDataset(args, args.datadir, train=True, device=device, prefix='OLAT_images',
                                        # OLAT_config=args.OLAT_config
                                        )
        shading_dataloader = DataLoader(shading_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=True)
        shading_prefetcher = data_prefetcher(shading_dataloader)
        use_OLAT = True

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

   # Create Embeddings
    identities, EMAPs, num_latent_vecs, meta_data, num_ids, num_EMAPs = init_latent_embedding(args.datadir, args)
    # if not args.fit: assert (num_EMAPs == NUM_RELIT_EMAPs + 1), "Check number of env maps loaded! {} {} ".format(num_EMAPs, NUM_RELIT_EMAPs)
    latent_embedding = torch.nn.Embedding(num_latent_vecs, args.latent_vec_size, max_norm=1).to(device)
    torch.nn.init.normal_(latent_embedding.weight.data, 0.0, 1.0)

    # if args.fit:
        # Envirommnet map latent embedding
    # emap_embedding = torch.nn.Embedding(1, 600, max_norm=1).to(device)
    # torch.nn.init.normal_(emap_embedding.weight.data, 0.0, 0.1)

    # print(EMAPs)

    # Create Finetuning Embedding

    ckpts = []
    emap_ckpts = []
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        if os.path.isdir(os.path.join(basedir, expname, 'lat_codes')):
            ckpts = [os.path.join(basedir, expname, 'lat_codes', f) for f in
                     sorted(os.listdir(os.path.join(basedir, expname, 'lat_codes'))) if 'npy' in f]

        if os.path.isdir(os.path.join(basedir, expname, 'emap_codes')):
            emap_ckpts = [os.path.join(basedir, expname, 'emap_codes', f) for f in
                     sorted(os.listdir(os.path.join(basedir, expname, 'emap_codes'))) if 'npy' in f]

    ckpts = natsort.natsorted(ckpts)
    # print('Found ckpts', ckpts)
    if (len(ckpts) > 0 and not args.no_reload) and not args.fit:
        ckpt_path = ckpts[-1]
        print("Loading latent embedding from {}".format(ckpt_path))
        load_latent_vectors(ckpt_path, latent_embedding)

    if (len(emap_ckpts) > 0 and not args.no_reload) and args.use_emaps:
        emap_ckpt_path = emap_ckpts[-1]
        print("Loading environment maps latent embedding from {}".format(emap_ckpt_path))
        load_latent_vectors(emap_ckpt_path, emap_embedding)

    # Create NeRF Model
    near, far = get_near_far_bounds(args.datadir, args.poses_bounds_fn)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, latent_embedding,
                                                                                       num_latent_modules=meta_data.shape[1])
    start = 0 if args.fit else start
    global_step = start


    bds_dict = {
        'near' : args.bds_min_scale * near,
        'far' : args.bds_max_scale * far,
    }
    print(bds_dict)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_kwargs_train['use_light_dirs']=args.use_light_dirs
    render_kwargs_test['use_light_dirs']=args.use_light_dirs

    render_kwargs_train['viewdirs_dim']=args.viewdirs_dims
    render_kwargs_test['viewdirs_dim']=args.viewdirs_dims

    render_kwargs_train['use_shading_nw']=args.use_shading_nw
    render_kwargs_test['use_shading_nw']=args.use_shading_nw

    if use_OLAT:
        batch, hwf, o_poses, scan_id, render_train_poses, render_poses,  \
        img_names, lights_arr, cam_id, use_perceptual_loss = shading_prefetcher.next()
    else:
        batch, hwf, o_poses, scan_id, render_train_poses, render_poses, \
        img_names, lights_arr, cam_id, use_perceptual_loss = geom_prefetcher.next()

    # N_iters = 8000000 + 1
    N_iters = 3000+1
    if args.lrate > 0:
        N_iters = N_iters + 200 + 1

    print('Begin Training')
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters, disable=True):
        global_step += 1

        if batch is None:
            if use_OLAT:
                shading_prefetcher = data_prefetcher(shading_dataloader)
                batch, hwf, o_poses, scan_id, render_train_poses, render_poses, \
                img_names, lights_arr, cam_id, use_perceptual_loss = shading_prefetcher.next()
            else:
                geom_prefetcher = data_prefetcher(train_dataloader)
                batch, hwf, o_poses, scan_id, render_train_poses, render_poses,\
                img_names, lights_arr, cam_id, use_perceptual_loss = geom_prefetcher.next()

        assert len(render_train_poses.shape) == 4
        d0, d1, d2, d3 = render_train_poses.shape
        render_train_poses = render_train_poses.reshape(d0*d1, d2, d3)
        # o_poses = torch.Tensor(o_poses).to(device)
        o_poses = o_poses.to(device)
        time0 = time.time()
        batch = torch.transpose(batch, 0, 1)

        # if not args.use_shading_nw:
        if not use_OLAT:
	        # if args.fit and (args.use_emaps and args.use_latent_emap):
            if (args.use_emaps and args.use_latent_emap):
                batch_latent_vector = get_latent_vector_fit(identities, EMAPs, meta_data, latent_embedding, emap_embedding,
                                                            scan_id, img_names, args.N_rand, args)
            else:
                batch_latent_vector = get_latent_vector(identities, EMAPs, meta_data, latent_embedding, scan_id,
                                                        img_names,
                                                        args.N_rand, args)
        else:
            batch_latent_vector = get_latent_vector_random(identities, EMAPs, meta_data, latent_embedding, scan_id, img_names,
                                                           args.N_rand)

        if args.batch_size > 1:
            H, W, focal = hwf[0][0], hwf[1][0], hwf[2][0]
        else:
            H, W, focal = hwf[0], hwf[1], hwf[2]

        H, W = int(H), int(W)
        hwf = [H, W, focal]

        batch_rays, target_s = batch[:2], batch[2]

        if args.use_light_dirs:
            batch_lights = batch[3]
            batch_rays = torch.cat((batch_rays, batch_lights.unsqueeze(0)), 0)

        if args.acc_loss:
            if not args.use_light_dirs:
                acc_mask = batch[3][..., 0] > 0
            else:
                acc_mask = batch[4][..., 0] > 0

        #####  Core optimization loop  #####
        rgb, disp, acc, weights, extras = render(H, W, focal, use_OLAT=use_OLAT, chunk=args.chunk, rays=batch_rays,latent_vector=batch_latent_vector,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        optimizer.zero_grad()
        trans = extras['raw'][...,-1]
        loss = 0; perceptual_loss = 0

        if args.acc_loss:
            img_loss = img2mse(rgb[acc_mask], target_s[acc_mask])
            acc_loss = img2mse(acc[~acc_mask], 0)
            loss += args.acc_loss_weight * acc_loss
        else:
            if args.use_HDR_loss and use_OLAT:
                img_loss = torch.mean(((rgb - target_s)/(rgb.detach() + 1e-3)) ** 2)
            else:
                img_loss = img2mse(rgb, target_s)

        loss += img_loss
        psnr = mse2psnr(img_loss)

        if args.use_hard_loss:
            beta_distrib = torch.exp(-torch.abs(weights)) + torch.exp(-torch.abs(1-weights))
            hard_loss = torch.mean(-torch.log(beta_distrib))
            loss += args.hard_loss_weight * hard_loss
            # print(hard_loss)

        if 'rgb0' in extras:
            if args.acc_loss:
                img_loss0 = img2mse(extras['rgb0'][acc_mask], target_s[acc_mask])
                acc_loss0 = img2mse(extras['acc0'][~acc_mask], 0)
                loss += args.acc_loss_weight * acc_loss0
            else:
                if args.use_HDR_loss and use_OLAT:
                    img_loss0 = torch.mean(((extras['rgb0'] - target_s)/(extras['rgb0'].detach() + 1e-3)) ** 2)
                else:
                    img_loss0 = img2mse(extras['rgb0'], target_s)

            if args.use_hard_loss:
                weights0 = extras['wt0']
                beta_distrib0 = torch.exp(-torch.abs(weights0)) + torch.exp(-torch.abs(1-weights0))
                hard_loss0 = torch.mean(-torch.log(beta_distrib0))
                loss += args.hard_loss_weight * hard_loss0

            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.use_code_regularisation:
            if args.use_emaps and not args.use_latent_emap:
                l2_size_loss = torch.sum(torch.norm(batch_latent_vector[:, :args.latent_vec_size], dim=1))
            else:
                l2_size_loss = torch.sum(torch.norm(batch_latent_vector, dim=1))
            # reg_loss = (args.reg_loss_weight *  l2_size_loss) / batch_latent_vector.shape[0]
            reg_loss = (args.reg_loss_weight * min(1, i*100/1000000 ) *l2_size_loss) / batch_latent_vector.shape[0]
            loss = loss + reg_loss

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT! ###   update learning rate   ###
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        new_latent_lrate = args.latent_lrate * (decay_rate ** (global_step / decay_steps))

        for idx, param_group in enumerate(optimizer.param_groups):
            if idx==0:
            # if param_group['name']=='latent_embedding':
                param_group['lr'] = new_latent_lrate
            else:
                param_group['lr'] = new_lrate


        dt = time.time() - time0
        writer.add_scalar('training loss', loss.item(), global_step)
        # writer.add_scalar('reg loss', reg_loss.item(), global_step)
        # writer.add_scalar('latent norm', (l2_size_loss/batch_latent_vector.shape[0]).item(), global_step)
        writer.add_scalar('lr', new_lrate, global_step)
        writer.add_scalar('latent lr', new_latent_lrate, global_step)

        if i%args.i_print==0:
            print(f"Step: {global_step}, Loss: {loss}, PSNR: {psnr.item()} LR: {new_lrate} {new_latent_lrate}")

        if i%args.i_weights==0:
            save_latent_vectors(os.path.join(basedir, expname), latent_embedding, i)
            if args.use_emaps and args.use_latent_emap:
                save_emap_vectors(os.path.join(basedir, expname), emap_embedding, i)

            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 or i == 1:
            # for dir in range(lights_arr.shape[0]):
            use_tone_mapping = True if (use_OLAT and args.tone_mapping) else False
            for dir in range(1):
               trainsavedir = os.path.join(basedir, expname,'trainset')
               prefix = '{}_{:06d}'.format(dir, i)
               if os.path.exists(os.path.join(basedir, expname)):
                   os.makedirs(trainsavedir, exist_ok=True)
               print('train poses shape', o_poses.shape)
               with torch.no_grad():
                   render_path(o_poses, hwf, use_OLAT, args.chunk, render_kwargs_test,latent_vector=batch_latent_vector,
                               savedir=trainsavedir,tone_mapping=use_tone_mapping, light_vector=lights_arr[dir], prefix=prefix,
                               render_factor=args.render_factor)
               print('Saved train set')

        if i % (args.i_video) == 0 and i > 0:
            # for dir in range(lights_arr.shape[0]):
            for dir in range(1):
               with torch.no_grad():
                   rgbs, disps = render_path(render_train_poses, hwf, use_OLAT, args.chunk, render_kwargs_test,latent_vector=batch_latent_vector,
                                             tone_mapping=args.tone_mapping, light_vector=lights_arr[dir], render_factor=args.render_factor)
                   # writer.add_image("rendered_test_view", to8b(cast_to_image(rgb)), i)
               print('Done, saving', rgbs.shape, disps.shape)
               os.makedirs(os.path.join(basedir, expname, 'interp'), exist_ok=True)
               moviebase = os.path.join(basedir, expname, 'interp', '{:06d}_'.format(i))
               imageio.mimwrite(moviebase + 'OLAT_{}_rgb.mp4'.format(dir), to8b(rgbs), fps=2, quality=8)

        # Next Batch
        if not args.fit:
            use_OLAT = np.random.choice([False, True], 1, p=[args.switch_prob, 1 - args.switch_prob])[0] if args.use_shading_nw else False
        if use_OLAT:
            batch, hwf, o_poses, scan_id, render_train_poses, \
            render_poses, img_names, lights_arr, cam_id, use_perceptual_loss = shading_prefetcher.next()
        else:
            batch, hwf, o_poses, scan_id, render_train_poses, render_poses,\
            img_names, lights_arr, cam_id, use_perceptual_loss = geom_prefetcher.next()


    ################# End of Batchwise Training ###############

def test(args):
    global NUM_RELIT_EMAPs
    NUM_RELIT_EMAPs = args.num_relit_emaps
    parser = config_parser()
    args = parser.parse_args()

    use_OLAT = False
    prefix = 'images'
    if args.use_shading_nw:
        use_OLAT = True
        prefix = 'OLAT_images'

    test_dataset = ImagesDataset(args, args.datadir, args.aux_datadir, train=True, device=device, prefix=prefix)
    test_dataloader = DataLoader(test_dataset,
                                  # batch_size=args.batch_size,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,  # int(self.opts.workers),
                                  drop_last=True)
    basedir = args.basedir
    expname = args.expname

    # Create Embeddings
    # dirs, num_latent_vecs, meta_data, num_ids = init_latent_embedding(args.datadir)
    identities, EMAPs, num_latent_vecs, meta_data, num_ids, num_EMAPs = init_latent_embedding(args.datadir, args)
    latent_embedding = torch.nn.Embedding(num_latent_vecs, args.latent_vec_size, max_norm=1)
    latent_embedding = latent_embedding.to(device)
    torch.nn.init.normal_(latent_embedding.weight.data, 0.0, 1.0)

    ckpts = []
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        if os.path.isdir(os.path.join(basedir, expname, 'lat_codes')):
            ckpts = [os.path.join(basedir, expname, 'lat_codes', f) for f in
                     sorted(os.listdir(os.path.join(basedir, expname, 'lat_codes'))) if 'npy' in f]

    # print('Found ckpts', ckpts)
    ckpts = natsort.natsorted(ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if args.test:
            ckpt_path = ckpts[args.ckpt_idx]
        else:
            ckpt_path = ckpts[-1]
        print("Loading latent embedding from {}".format(ckpt_path))
        load_latent_vectors(ckpt_path, latent_embedding)

    # Create NeRF Model
    near, far = get_near_far_bounds(args.datadir, args.poses_bounds_fn)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, latent_embedding, meta_data.shape[1])

    # import pdb; pdb.set_trace()
    if not args.render_train:
        bds_dict = {
            # 'near': 0.8 * near,
            # 'far': 1.1 * far,
            # 'near': 0.5* near,
            # 'far': 0.8 * far,
            # 'near' : 1.0,
            # 'far':2.23
            'near': 0.8 * near,
            'far': 1.1 * far,
        }
    else:
        bds_dict = {
            'near': 0.8 * near,
            'far': 1.1 * far,
            # 'near': 3.0,
            # 'far': 2.23
        }

    render_kwargs_test.update(bds_dict)
    render_kwargs_test['use_light_dirs'] = args.use_light_dirs
    render_kwargs_test['viewdirs_dim'] = args.viewdirs_dims

    prefetcher = data_prefetcher(test_dataloader)
    sample_batch = prefetcher.next()
    hwf = sample_batch[1]; img_names = sample_batch[-2]; lights_arr = sample_batch[-1];

    EMAPS_LIST = natsort.natsorted(glob.glob1(os.path.join(args.datadir,identities[0], 'images'), 'Cam07_EMAP*'))
    EMAPS_LIST = list(dict.fromkeys([s.split('_')[1][0:8] for s in EMAPS_LIST]))

    if args.use_emaps:
        eval_EMAPs = sorted(glob.glob1(args.emap_dir, '*.exr'))
        start_idx = eval_EMAPs.index('EMAP-873.exr')

    # ## Sample between two lights directions
    # lights_arr = np.load(os.path.join('data_utils', 'light_dirs.npy'))
    # lights_arr_interp = []
    # for i in range(len(lights_arr) - 1):
    #     lights_arr_interp.extend(np.linspace(lights_arr[i], lights_arr[i + 1], 11 + 1))
    # lights_arr_render = torch.Tensor(np.array(lights_arr_interp))

    # for idx, ID in enumerate(sorted(os.listdir(args.aux_datadir[2]))[0:]):
    for idx, ID in enumerate(sorted(identities,reverse=False)):
        if not args.use_shading_nw:
            sample_image = glob.glob(os.path.join(args.datadir, ID, 'images/*.png'))[0]
            # sample_image = glob.glob(os.path.join(args.aux_datadir[-1], ID, 'images/*.png'))[0]
        else:
            # with open('./OLATs/{}{}.txt'.format(os.path.basename(args.datadir[:-3]), args.OLAT_config), 'r') as f:
            #     all_imgs = f.read().splitlines()
            # if 'h3ds' or 'celeba' in args.datadir:
            if ('h3ds' in args.datadir) or ('celeba' in args.datadir):
                print('./OLATs/{}{}.txt'.format(os.path.basename(args.datadir), args.OLAT_config))
                with open('./OLATs/{}{}.txt'.format(os.path.basename(args.datadir), args.OLAT_config), 'r') as f:
                    all_imgs = f.read().splitlines()
            else:
                print('./OLATs/{}{}.txt'.format(os.path.basename(args.datadir[:-3]), args.OLAT_config))
                with open('./OLATs/{}{}.txt'.format(os.path.basename(args.datadir[:-3]), args.OLAT_config), 'r') as f:
                    all_imgs = f.read().splitlines()

            sample_img_list = [f for f in all_imgs if ID in f]
            sample_image = sample_img_list[0]
        render_poses = test_dataset.getSampleData(sample_image)[-2]
        render_poses = torch.Tensor(render_poses).cuda()
        render_train_poses = test_dataset.getSampleData(sample_image)[3]
        render_train_poses = torch.Tensor(render_train_poses).cuda()

        if args.render_single:
            # mean_latent_vector = get_latent_vector_random(identities, EMAPs, meta_data, latent_embedding, tuple((ID,)),
            #                                               N_rays=args.N_rand)
            iter_range = 150 if args.use_shading_nw else 1
            # iter_range = 150 if args.use_shading_nw else 10
            # iter_range = 150 if args.use_shading_nw else args.num_relit_emaps
            save_dirname = 'render-single-OLATs' if args.use_shading_nw else 'render-single-new'
            rendered_frames = []
            for dir in range(iter_range):
                mean_latent_vector = get_latent_vector_test(identities, EMAPs, meta_data, latent_embedding,
                                                            tuple((ID,)),
                                                            # N_rays=args.N_rand, test_EMAPs=eval_EMAPs[start_idx+dir], args=args)
                                                            N_rays=args.N_rand, test_EMAPs=EMAPs[-1], args=args)
                save_filename = '{}'.format(ID)
                os.makedirs(os.path.join(basedir, expname, save_dirname), exist_ok=True)
                lights = torch.Tensor(np.load(os.path.join('data_utils', 'light_dirs.npy')))
                # savedir = os.path.join(basedir, expname, save_dirname) if not args.use_shading_nw else None
                savedir = os.path.join(basedir, expname, save_dirname)
                with torch.no_grad():
                    # import pdb; pdb.set_trace()
                    # rgbs, disps = render_path(render_poses[[7, 15, 28, 35], :, :], hwf, use_OLAT, args.chunk, render_kwargs_test,
                    rgbs, disps = render_path(render_poses, hwf, use_OLAT, args.chunk, render_kwargs_test,
                                              latent_vector=mean_latent_vector, prefix='{}_{}'.format(str(dir), save_filename),
                                              render_factor=args.render_factor,
                                              savedir=savedir,
                                              tone_mapping=args.tone_mapping, light_vector=lights[dir] if args.use_shading_nw else lights[0])
                rendered_frames.extend(rgbs)
                # print('Done, saving', rgbs.shape, disps.shape)
            if iter_range >= 150:
                moviebase = os.path.join(basedir, expname, 'render-single-video')
                os.makedirs(moviebase, exist_ok=True)
                imageio.mimwrite(moviebase + '/{}_ID_{}_rgb.mp4'.format(str(dir), ID),
                                 to8b(rendered_frames), fps=4, quality=10)

        elif args.render_all:
            print("Rendering all ID {} ... ".format(ID))
            # import pdb; pdb.set_trace()
            save_filename = '{}'.format(ID)
            mean_latent_vector = get_latent_vector_random(identities, EMAPs, meta_data, latent_embedding, tuple((ID,)),
                                                        N_rays=args.N_rand)

            rendered_frames = []
            # iter_range = [7, 10, 12, 23, 24, 27, 32, 44, 55, 101, 147] if args.use_shading_nw else range(15)
            iter_range = range(150) if args.use_shading_nw else [0]
            # iter_range = [11] if args.use_shading_nw else [0]
            for dir in iter_range:
                # save_filename = '{}'.format(ID)
                # os.makedirs(os.path.join(basedir, expname, 'render-all'), exist_ok=True)
                lights = torch.Tensor(np.load(os.path.join('data_utils', 'light_dirs.npy')))

                # mean_latent_vector = get_latent_vector_test(identities, EMAPs, meta_data, latent_embedding, tuple((ID,)),
                #                                             N_rays=args.N_rand, test_EMAPs=EMAPs[dir])
                os.makedirs(os.path.join(basedir, expname, 'render-vd', save_filename), exist_ok=True)
                with torch.no_grad():
                    rgbs, disps = render_path(render_poses, hwf, use_OLAT, args.chunk, render_kwargs_test,
                                              render_factor=args.render_factor,
                                              latent_vector=mean_latent_vector, prefix='{:03d}'.format(dir),
                                              savedir=os.path.join(basedir, expname, 'render-vd', save_filename),
                                              # savedir=None,
                                              tone_mapping=args.tone_mapping, light_vector=lights[dir])
                # print("Saving Rendered output at: {}".format(os.path.join(basedir, expname, 'render')))
                rendered_frames.extend(rgbs)
            moviebase = os.path.join(basedir, expname, 'render')
            imageio.mimwrite(moviebase + '/{}_ID_{}_rgb.mp4'.format(str(dir), ID),
                             to8b(rendered_frames), fps=4, quality=10)

        if args.render_train:
            print("Rendering ID {} ... ".format(ID))
            save_filename = '{}'.format(ID)
            # mean_latent_vector = get_latent_vector_random(identities, EMAPs, meta_data, latent_embedding, tuple((ID,)),
            #                                               N_rays=args.N_rand)

            # iter_range = range(len(lights_arr_interp)) if args.use_shading_nw else [0] # For OLAT interpolations
            iter_range = range(150) if args.use_shading_nw else range(1)
            # pdb.set_trace()
            for dir in iter_range:
                # save_filename = '{}'.format(ID)

                # sub_folder = 'render-train-8' if args.use_shading_nw else 'render-train'
                mean_latent_vector = get_latent_vector_test(identities, EMAPs, meta_data, latent_embedding,
                                                            tuple((ID,)),
                                                            N_rays=args.N_rand, test_EMAPs=EMAPs[-1], args=args)

                # 'render-train-OLAT-00' - post submission evaluation
                sub_folder = 'render-train-OLAT-00' if args.use_shading_nw else 'render-train-88'
                # sub_folder = 'render-train-OLAT-interp' if args.use_shading_nw else 'render-train-88'
                # sub_folder = 'render-train-8' if args.use_shading_nw else 'render-train'
                os.makedirs(os.path.join(basedir, expname, sub_folder , save_filename), exist_ok=True)
                lights = torch.Tensor(np.load(os.path.join('data_utils', 'light_dirs.npy')))

                # import pdb; pdb.set_trace()
                from data_utils import olat_indices
                if '50_OLATs' in args.OLAT_config:
                    lights = lights[olat_indices.config_50]
                elif '100_OLATs' in args.OLAT_config:
                    lights = lights[olat_indices.config_100]

                with torch.no_grad():
                    # camera_idx = torch.LongTensor([0, 4, 7]).to(device)
                    if args.use_shading_nw:
                        # Seen Views
                        camera_idx = torch.LongTensor([0]).to(device) # config 3dpr 811-520
                        # 'render-train-OLAT-00'
                        # camera_idx = torch.LongTensor([4, 6, 8, 10, 11]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([6]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([6, 7]).to(device) # config 3dpr 651-500
                        # camera_idx = torch.LongTensor([7]).to(device) # config 3dpr 761-505
                        # camera_idx = torch.LongTensor([6,7]).to(device) # config 3dpr 811-520
                        camera_idx = torch.LongTensor([10]).to(device) # config 3dpr 820-521
                        camera_idx = torch.LongTensor([6,10]).to(device) # config 3dpr 615-508
                        camera_idx = torch.LongTensor([6,7]).to(device) # config 3dpr 615-508
                        # camera_idx = torch.LongTensor([4, 7]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([7, 8]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([8, 36]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([4, 11, 8, 36]).to(device) # config h3ds 500
                        # camera_idx = torch.LongTensor([12, 36, 2, 6]).to(device) # config h3ds 506
                        # camera_idx = torch.LongTensor([ii for ii in range(len(render_train_poses))]).to(device)


                        # if len(render_train_poses)>20:
                        #     camera_idx = torch.LongTensor([0, 8, 12, 29, 32]).to(device) # h3ds
                        # pdb.set_trace()
                    else:
                        # camera_idx = torch.LongTensor([1]).to(device)
                        camera_idx = torch.LongTensor([ii for ii in range(len(render_train_poses))]).to(device)
                        # camera_idx = torch.LongTensor([2, 3, 6, 7, 8, 10, 11]).to(device)
                    # camera_idx = torch.LongTensor([9, 24, 26, 2c , 47, 49]).to(device)
                    # import pdb; pdb.set_trace()
                    # rgbs, disps = render_path(render_train_poses[camera_idx], hwf, use_OLAT, args.chunk, render_kwargs_test,
                    if dir == 150:
                        print("Generation for cameras ", camera_idx)
                    rgbs, disps = render_path(render_train_poses[camera_idx] if args.use_shading_nw else render_train_poses,
                                              hwf, use_OLAT, args.chunk, render_kwargs_test,
                                              render_factor=args.render_factor,
                                              latent_vector=mean_latent_vector, prefix='{:03d}'.format(dir),
                                              savedir=os.path.join(basedir, expname, sub_folder, save_filename),
                                              # savedir=None,
                                              # tone_mapping=args.tone_mapping, light_vector=lights_arr_render[dir], camera_idx=camera_idx) #WARNING: Interpolation
                                              tone_mapping=args.tone_mapping, light_vector=lights[dir], camera_idx=camera_idx)
                                              # tone_mapping=args.tone_mapping, light_vector=lights[dir], camera_idx=None)

                # moviebase = os.path.join(basedir, expname, 'render-train')
                # imageio.mimwrite(moviebase + '/{}_ID_{}_rgb.mp4'.format(str(dir), ID),
                #                  to8b(rgbs), fps=4, quality=10)


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # import pdb; pdb.set_trace()
    torch.multiprocessing.set_start_method('spawn')
    parser = config_parser()
    args = parser.parse_args()
    print("Started experiment: {} ".format(args.expname))
    if (args.test == False or args.fit):
        train(args=args)
    elif (args.test and not args.fit):
        test(args=args)
    else:
        print("Incorrect Mode!")
