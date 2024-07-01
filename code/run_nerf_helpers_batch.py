import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from siren import CustomMappingNetwork, frequency_init, first_layer_film_sine_init

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : (65535*np.clip(x,0,1)).astype(np.uint16)
tonemap = lambda x :(pow(x/(pow(2,16)),0.5) * 255)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    # Note: If we want to give light dir w/o encoding, add another arg "light" to lambda function below
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, D_l=3, W=256, input_ch=3, input_ch_views=3, input_ch_light=3, latent_dim=8, output_ch=4, skips=[4], use_viewdirs=False,
                 use_custom_mapping=False, use_sine=False, use_shading_nw=False, N_rays=24, fit=True):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.D_l = D_l
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_light = input_ch_light
        self.latent_dim = latent_dim
        if skips[0] == -1:
            self.skips = []
        else:
            self.skips = skips

        self.use_viewdirs = use_viewdirs
        self.use_custom_mapping = use_custom_mapping
        self.use_sine = use_sine
        self.use_shading_nw = use_shading_nw
        self.fit = fit

        if self.use_custom_mapping:
            assert self.D == 8, "Variable depth for Siren network NOT implemented!"

            self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] +
                                             [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

            self.mapping_network = CustomMappingNetwork(z_dim=self.latent_dim, map_hidden_dim=W,
                                                        map_output_dim=(self.D + 1) * W * 2)
            if self.use_sine:
                self.pts_linears.apply(frequency_init(25))
                self.pts_linears[0].apply(first_layer_film_sine_init)

            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])
            if self.use_sine: self.views_linears.apply(frequency_init(25))

        else:
            self.pts_linears = nn.ModuleList([nn.Linear(input_ch + latent_dim, W)] +
                                             [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch + latent_dim, W) for i in range(D - 1)])
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W), nn.Linear(W, W//2)])

        if self.use_shading_nw:
            self.shading_nw = ShadingNeRF(D=D_l, W=W, input_ch=input_ch, input_ch_light=input_ch_light,
                                          input_ch_views=input_ch_views, latent_dim = latent_dim, skips=skips,
                                          feature_geom_dim = W, output_ch = output_ch, use_viewdirs = use_viewdirs)

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)

            if self.use_custom_mapping:
                if self.use_sine:
                    self.alpha_linear.apply(frequency_init(25))

                self.rgb_linear = nn.Linear(W, 3)
                if self.use_sine:
                    self.rgb_linear.apply(frequency_init(25))
            else:
                self.feature_linear = nn.Linear(W, W)
                self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, use_OLATs):
        if self.use_shading_nw:
            input_pts, input_latent, input_views, light_dirs = \
            torch.split(x, [self.input_ch, self.latent_dim, self.input_ch_views, self.input_ch_light], dim=-1)
        else:
            input_pts, input_latent, input_views = \
                torch.split(x, [self.input_ch, self.latent_dim, self.input_ch_views], dim=-1)

        # self.multires_features = torch.empty((2, input_pts.shape[0] , self.W) , device='cuda')
        if self.use_custom_mapping:
            frequencies, phase_shifts = self.mapping_network(input_latent)
            frequencies_pts =  frequencies[..., :-self.W]; phase_shifts_pts = phase_shifts[..., :-self.W]
            frequencies_view_dirs =  frequencies[..., -self.W:]; phase_shifts_view_dirs = phase_shifts[..., -self.W:]

            h = self.forward_with_frequencies_phase_shifts(input_pts, frequencies_pts, phase_shifts_pts, self.pts_linears)
        else:
            h = torch.cat([input_pts, input_latent], -1)
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, input_latent, h], -1)

                # if i==2: self.multires_features[0] = h
                # if i==6: self.multires_features[1] = h

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            if self.use_custom_mapping:
                feature = h
            else:
                feature = self.feature_linear(h)

            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                if self.use_custom_mapping:
                    frequencies_view_dirs = frequencies_view_dirs * 15 + 30
                    h = h * frequencies_view_dirs + phase_shifts_view_dirs
                    if self.use_sine:
                        h = torch.sin(h)
                    else:
                        h = F.relu(h)
                else:
                    h = F.relu(h)

            # if (self.use_shading_nw and self.fit):
            if (self.use_shading_nw and use_OLATs):
                rgb_shading = self.shading_nw(input_pts, input_latent, light_dirs, feature, input_views)
                outputs = torch.cat([rgb_shading, alpha], -1)
            else:
                rgb = self.rgb_linear(h)
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, network):
        h = input
        frequencies = frequencies * 15 + 30

        for index, layer in enumerate(network):
            start = index * self.W
            end = (index + 1) * self.W
            freq = frequencies[..., start:end]
            ps = phase_shifts[..., start:end]

            h = layer(h)
            h = freq * h + ps
            if self.use_sine:
                h = torch.sin(h)
            else:
                h = F.relu(h)
            if index in self.skips:
                h = torch.cat([input, h], -1)
        return h

    def forward_shading(self, postion, lightdirs, viewdirs, feature):
        h = torch.cat([postion, lightdirs], -1)
        for i, l in enumerate(self.shading_linears):
            h = self.shading_linears[i](h)
            h = F.relu(h)

        h = torch.cat([h, viewdirs], -1)
        for i, l in enumerate(self.shading_views_linears):
            h = self.shading_views_linears[i](h)
            h = F.relu(h)

        return self.shading_rgb_linear(h)

class ShadingNeRF(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_views=3, input_ch_light=3,
                 feature_geom_dim=8, skips=[4], latent_dim=8,
                 output_ch=4, use_viewdirs=False):
        """
        """
        super(ShadingNeRF, self).__init__()
        self.W = W
        self.D = D
        self.skips = skips
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_light = input_ch_light
        self.feature_geom_dim = feature_geom_dim
        self.use_viewdirs = use_viewdirs
        self.latent_dim = latent_dim

        self.pts_linears = nn.ModuleList([
                            nn.Linear(input_ch + input_ch_light + feature_geom_dim + latent_dim, W)] + [nn.Linear(W, W)
                            if i not in self.skips else nn.Linear(W + input_ch + input_ch_light + feature_geom_dim + latent_dim, W) for i in range(D - 1)])
        # self.pts_linears = nn.ModuleList([
        #                         nn.Linear(input_ch + input_ch_light + latent_dim, W),
        #
        #
        # ])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W), nn.Linear(W, W//2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_latent, light_dirs, feature_geom, input_views):

        h = torch.cat([input_pts, input_latent, light_dirs, feature_geom], -1)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, input_latent, light_dirs, feature_geom , h], -1)

        if self.use_viewdirs:
            feature = h
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = rgb
        else:
            outputs = self.output_linear(h)

        return outputs

# Model
class MultiResLatentNeRF(nn.Module):
    def __init__(self, D=8, D_l=3, W=256, input_ch=3, input_ch_views=3, input_ch_light=3, latent_dim=8, output_ch=4, skips=[4], use_viewdirs=False,
                 use_custom_mapping=True, use_sine=False, use_shading_nw=False, N_rays=24, fit=True):
        """
        """
        super(MultiResLatentNeRF, self).__init__()
        self.D = D
        self.D_l = D_l
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_light = input_ch_light
        self.latent_dim = latent_dim
        if skips[0] == -1:
            self.skips = []
        else:
            self.skips = skips
        self.multires_indices = [i*2 for i in range(self.D // 2)]

        assert len(skips) == 1, print("Only one skip possible")
        assert self.skips[0] in self.multires_indices, \
            print("Skip layyer should be in multi-resolution indices")

        self.use_viewdirs = use_viewdirs
        self.use_custom_mapping = use_custom_mapping
        self.use_sine = use_sine
        self.use_shading_nw = use_shading_nw
        self.fit = fit

        if self.use_custom_mapping:
            assert self.D == 8, \
                "Variable depth for Siren network NOT implemented!"
            # self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] +
            #                                  [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

            self.pts_linears = nn.ModuleList([ nn.Linear(input_ch + latent_dim, W), # 0
                                               nn.Linear(W, W),  # 1
                                               nn.Linear(W + latent_dim, W),  # 2
                                               nn.Linear(W, W),  # 3
                                               nn.Linear(W + input_ch + latent_dim, W),  # 4 + skip
                                               nn.Linear(W, W),  # 5
                                               nn.Linear(W + latent_dim, W),  # 6
                                               nn.Linear(W, W),  # 7
                                               ])

            self.mapping_network = CustomMappingNetwork(z_dim=self.latent_dim, map_hidden_dim=W,
                                                        map_output_dim=(self.D//2) * self.latent_dim)

            # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W), nn.Linear(W, W//2)])

        else:
            self.pts_linears = nn.ModuleList([nn.Linear(input_ch + latent_dim, W)] +
                                             [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch + latent_dim, W) for i in range(D - 1)])
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W), nn.Linear(W, W//2)])

        if self.use_shading_nw:
            self.shading_nw = ShadingNeRF(D=D_l, W=W, input_ch=input_ch, input_ch_light=input_ch_light,
                                          input_ch_views=input_ch_views, latent_dim = latent_dim, skips=skips,
                                          feature_geom_dim = W, output_ch = output_ch, use_viewdirs = use_viewdirs)

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, use_OLATs):
        if self.use_shading_nw:
            input_pts, input_latent, input_views, light_dirs = \
                torch.split(x, [self.input_ch, self.latent_dim, self.input_ch_views, self.input_ch_light], dim=-1)
        else:
            input_pts, input_latent, input_views = \
                torch.split(x, [self.input_ch, self.latent_dim, self.input_ch_views], dim=-1)

        # self.multires_features = torch.empty((2, input_pts.shape[0] , self.W) , device='cuda')
        if self.use_custom_mapping:
            latent_vectors = self.mapping_network(input_latent)

            for i in range(len(self.pts_linears)):
                if i in self.multires_indices:
                    start = (i // 2) * self.latent_dim
                    end = (i // 2 + 1) * self.latent_dim
                    if i == 0:
                        assert start == i, print("check latent vector indexing")
                        h = torch.cat([input_pts, latent_vectors[..., start:end]], -1) # 0
                        h = self.pts_linears[i](h)
                        h = F.relu(h)
                    elif i in self.skips:
                        h = torch.cat([input_pts, latent_vectors[..., start:end], h], -1) # skip
                        h = self.pts_linears[i](h)
                        h = F.relu(h)
                    else:
                        h = torch.cat([latent_vectors[..., start:end], h], -1) # 2, 4, 6
                        h = self.pts_linears[i](h)
                        h = F.relu(h)
                else:
                    h = self.pts_linears[i](h) # 1, 3, 5, 7
                    h = F.relu(h)

        else:
            h = torch.cat([input_pts, input_latent], -1)
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, input_latent, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            # if self.use_custom_mapping:
            if False:
                feature = h
            else:
                feature = self.feature_linear(h)

            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                # if self.use_custom_mapping:
                if False:
                    frequencies_view_dirs = frequencies_view_dirs * 15 + 30
                    h = h * frequencies_view_dirs + phase_shifts_view_dirs
                    if self.use_sine:
                        h = torch.sin(h)
                    else:
                        h = F.relu(h)
                else:
                    h = F.relu(h)

            # if (self.use_shading_nw and self.fit):
            if (self.use_shading_nw and use_OLATs):
                rgb_shading = self.shading_nw(input_pts, input_latent, light_dirs, feature, input_views)
                outputs = torch.cat([rgb_shading, alpha], -1)
            else:
                rgb = self.rgb_linear(h)
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# Ray helpers
def get_rays_old(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays(H, W, focal, c2w, i ,j):
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays2render(H, W, focal, c2w):
    H, W = int(H), int(W)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays2render_patch(H, W, focal, c2w):
    H, W = int(H), int(W)
    offset_H, offset_W = 400, 400
    crop_H, crop_W = 600, 600
    i, j = torch.meshgrid(torch.linspace(offset_W, offset_W+crop_W-1, crop_W),
                          torch.linspace(offset_H, offset_H+crop_H-1, crop_H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
#def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
#    # Get pdf
#    weights = weights + 1e-5 # prevent nans
#    pdf = weights / torch.sum(weights, -1, keepdim=True)
#    cdf = torch.cumsum(pdf, -1)
#    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
#
#    # Take uniform samples
#    if det:
#        u = torch.linspace(0., 1., steps=N_samples)
#        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#    else:
#        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
#
#    # Pytest, overwrite u with numpy's fixed random numbers
#    if pytest:
#        np.random.seed(0)
#        new_shape = list(cdf.shape[:-1]) + [N_samples]
#        if det:
#            u = np.linspace(0., 1., N_samples)
#            u = np.broadcast_to(u, new_shape)
#        else:
#            u = np.random.rand(*new_shape)
#        u = torch.Tensor(u)
#
#    # Invert CDF
#    u = u.contiguous()
#    inds = torch.searchsorted(cdf, u, side='right')
#    below = torch.max(torch.zeros_like(inds-1), inds-1)
#    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
#
#    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
#
#    denom = (cdf_g[...,1]-cdf_g[...,0])
#    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
#    t = (u-cdf_g[...,0])/denom
#    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
#
#    return samples

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    eps: float = 1e-5,
    pytest = False
):
    """
    Samples a probability density functions defined by bin edges `bins` and
    the non-negative per-bin probabilities `weights`.

    Note: This is a direct conversion of the TensorFlow function from the original
    release [1] to PyTorch.

    Args:
        bins: Tensor of shape `(..., n_bins+1)` denoting the edges of the sampling bins.
        weights: Tensor of shape `(..., n_bins)` containing non-negative numbers
            representing the probability of sampling the corresponding bin.
        N_samples: The number of samples to draw from each set of bins.
        det: If `False`, the sampling is random. `True` yields deterministic
            uniformly-spaced sampling from the inverse cumulative density function.
        eps: A constant preventing division by zero in case empty bins are present.

    Returns:
        samples: Tensor of shape `(..., N_samples)` containing `N_samples` samples
            drawn from each set probability distribution.

    Refs:
        [1] https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L183  # noqa E501
    """

    # Get pdf
    weights = weights + eps  # prevent nans
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]).contiguous()
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [N_samples], device=cdf.device, dtype=cdf.dtype
        )

    # Invert CDF
    # import pdb; pdb.set_trace()
    inds = torch.searchsorted(cdf.detach(), u.detach(), right=True)
    below = (inds - 1).clamp(0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1).view(
        *below.shape[:-1], below.shape[-1] * 2
    )

    cdf_g = torch.gather(cdf, -1, inds_g).view(*below.shape, 2)
    bins_g = torch.gather(bins, -1, inds_g).view(*below.shape, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
