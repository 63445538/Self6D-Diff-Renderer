# test 19 NIPS DIB-Renderer
# render multi objects in batch, one in one image
import os
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import mmcv
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat
# from kaolin.graphics import DIBRenderer

from core.dr_utils.dib_renderer_x import DIBRenderer
from core.dr_utils.dr_utils import load_objs, render_dib_vc_batch, render_dib_tex_batch

HEIGHT = 480
WIDTH = 640
ZNEAR = 0.01
ZFAR = 10.0
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


def heatmap(input, min=None, max=None, to_255=False, to_rgb=False):
    """ Returns a BGR heatmap representation """
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min + 0.001))

    final = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
    if to_rgb:
        final = final[:, :, [2, 1, 0]]
    if to_255:
        return final.astype(np.uint8)
    else:
        return final.astype(np.float32) / 255.0


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    if titles is not None:
        assert len(ims) == len(titles), "{} != {}".format(len(ims), len(titles))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(ims):
                break
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1

    # plt.tight_layout()
    if show:
        plt.show()
        plt.savefig('result.png')
    else:
        if save_path is not None:
            mmcv.mkdir_or_exist(os.path.dirname(save_path))
            plt.savefig(save_path)
    return fig


# box_models_path = 'box_models'
box_models_path = 'data/lm_models'

# box_id = 'box_1x1x1'
box_id = 'ape/textured'

objs = [box_id]

obj_paths = [os.path.join('box_models', 'box_1x1x1') + '.obj']
texture_paths = [os.path.join('box_models', 'box_1x1x1') + '.jpg']

models = load_objs(obj_paths, texture_paths, height=HEIGHT, width=WIDTH)
ren = DIBRenderer(HEIGHT, WIDTH, mode="VertexColorBatch")  # TextureBatch


# pose =============================================
R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
R = np.dot(R1, R2)
quat = mat2quat(R)
t = np.array([-0.1, 0.1, 1.3], dtype=np.float32)
t2 = np.array([0.1, 0.1, 1.3], dtype=np.float32)
t3 = np.array([-0.1, -0.1, 1.3], dtype=np.float32)
t4 = np.array([0.1, -0.1, 1.3], dtype=np.float32)
t5 = np.array([0, 0.1, 1.3], dtype=np.float32)

"""
(2) render multiple objs in a batch, one obj one image
"""
tensor_args = {"device": "cuda", "dtype": torch.float32}
# Rs = [R, R.copy(), R.copy(), R.copy(), R.copy()]
Rs = [R]
# quats = [quat, quat.copy(), quat.copy(), quat.copy(), quat.copy()]
quats = [quat]
# ts = [t, t2, t3, t4, t5]
ts = [t]

Rs = torch.tensor(Rs).to(**tensor_args)
ts = torch.tensor(ts).to(**tensor_args)
# poses = [np.hstack((_R, _t.reshape(3, 1))) for _R, _t in zip(Rs, ts)]
obj_ids = np.random.choice(list(range(0, len(objs))), len(Rs))
Ks = [K for _ in Rs]
# bxhxwx3 rgb, bhw1 prob, bhw1 mask, bhw depth
box_vertices = models[0]['vertices'][0]
box_size_max, _ = torch.max(box_vertices, dim=0)
box_size_min, _ = torch.min(box_vertices, dim=0)
box_dim = box_size_max - box_size_min
nocs_map = box_vertices/box_dim + torch.tensor([0.5, 0.5, 0.5], device='cuda')
colors = nocs_map
# vertices_num, _ = box_vertices.shape
# color = torch.tensor([128, 0, 255])/255
# colors = color.repeat((1, vertices_num)).view(len(objs), -1, 3).cuda()

colors = torch.unsqueeze(colors, dim=0)
models[0]['colors'] = colors

ren_ims, ren_probs, ren_masks, ren_depths = render_dib_vc_batch(
    ren, Rs, ts, Ks, obj_ids, models, rot_type="mat", H=480, W=640, near=0.01, far=100.0, with_depth=True
)
for i in range(len(Rs)):
    cur_im = ren_ims[i].detach().cpu().numpy()
    cur_prob = ren_probs[i, :, :, 0].detach().cpu().numpy()
    cur_mask = ren_masks[i, :, :, 0].detach().cpu().numpy()
    cur_depth = ren_depths[i].detach().cpu().numpy()
    show_ims = [cur_im, cur_prob, cur_mask, heatmap(cur_depth, to_rgb=True)]
    show_titles = ["im", "prob", "mask", "depth"]
    grid_show(show_ims, show_titles, row=2, col=2)