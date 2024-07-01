import numpy as np
import os, imageio
from scipy import io
from scipy.spatial.transform import Rotation as R


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[], load_mask=False):
    needtoload = False
    needtoload_mask = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not (os.path.exists(imgdir)):
            needtoload = True
        if load_mask:
            maskdir = os.path.join(basedir, 'masks_{}'.format(r))
            if not (os.path.exists(maskdir)):
                needtoload_mask = True

    for r in resolutions:
        #currently ignored for masks
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True

    if not (needtoload or needtoload_mask):
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

    if load_mask:
        maskdir = os.path.join(basedir, 'masks')
        masks = [os.path.join(maskdir, f) for f in sorted(os.listdir(maskdir))]
        masks = [f for f in masks if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
        maskdir_orig = maskdir

        for r in factors + resolutions:
            if isinstance(r, int):
                name = 'masks_{}'.format(r)
                resizearg = '{}%'.format(100. / r)

            maskdir = os.path.join(basedir, name)
            if os.path.exists(maskdir):
                continue

            print('Minifying Masks', r, basedir)

            os.makedirs(maskdir)
            check_output('cp {}/* {}'.format(maskdir_orig, maskdir), shell=True)

            m_ext = masks[0].split('.')[-1]
            m_args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(m_ext)])
            print(m_args)
            os.chdir(maskdir)
            check_output(m_args, shell=True)
            os.chdir(wd)

            if m_ext != 'png':
                check_output('rm {}/*.{}'.format(maskdir, m_ext), shell=True)
                print('Removed duplicates masks')
            print('Done')


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds



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


def getRenderPoses(poses):
    poseSamples = 10
    out = None
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
    return out



