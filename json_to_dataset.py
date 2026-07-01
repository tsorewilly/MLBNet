import argparse
import json
import os
import os.path as osp
import warnings
import io 
import PIL.Image
import yaml
import numpy as np
from labelme import utils
import base64

import io
import PIL.ImageDraw
 
 
def label_colormap(N=256):
 
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
 
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap
 
 
# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))
 
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)
 
    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled
 
    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)
 
    return lbl_viz
 
 
def draw_label(label, img=None, label_names=None, colormap=None):
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')
 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
 
    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]
 
    if colormap is None:
        colormap = label_colormap(len(label_names))
 
    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')
 
    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                          .format(value=label_value, name=label_name))
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
 
    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()
 
    plt.switch_backend(backend_org)
 
    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out
 
dataDir = "G:\Documents\Asst Prof\Publications\TMI\gw_1841\json"
count = os.listdir(r""+dataDir) #json文件
for i in range(0, len(count)):
    path = os.path.join(r""+dataDir, count[i]) #json文件

    if os.path.isfile(path) and path.endswith('json'):
        data = json.load(open(path,"rb"))
        
        if data['imageData']:
            imageData = data['imageData']
            print(imageData)
        else:
            #imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
            imagePath = os.path.join(r""+dataDir, data['imagePath']) #原图
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')


        img = utils.img_b64_to_arr(imageData)
        # seg_img = PIL.Image.fromarray(np.uint8(img))
        # print(img.shape)
        # seg_img.show()
        label_name_to_value = {'background': 0,'guidewire': 1}
        for shape in data['shapes']:
            label_name = shape['label']
            print(label_name+'---')
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        print(label_name_to_value)
        
        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            print(sorted(label_name_to_value.items(), key=lambda x: x[1]))
            label_values.append(lv)
            print(label_values)   #[0,1,2]
            label_names.append(ln)  #[background,fire,smoke]
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value) #得到标签矩阵，只需label_name_to_value和data['shapes']
        print(lbl)
        # captions = ['{}: {}'.format(lv, ln)
        #     for ln, lv in label_name_to_value.items()]
        captions = label_names
        print(captions)
        lbl_viz = draw_label(lbl, img, captions)
        out_dir = osp.basename(count[i]).replace('.', '_')
        print(out_dir)
        out_dir = osp.join(osp.dirname(count[i]), out_dir)
        print(out_dir)
        #out_dir = osp.join(out_dir)

        if not osp.exists(out_dir):
            os.mkdir(out_dir)
 
        PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))

        utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
 
        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            print(label_names)
            for lbl_name in label_names:
                f.write(lbl_name + '\n')
 
        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        print(info)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)
 
        print('Saved to: %s' % out_dir)