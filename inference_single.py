import os
import glob

import torch

import numpy as np
import cv2

from isegm.model.is_trimap_plaintvit_model_noposembed import NoPosEmbedTrimapPlainVitModel

import torchvision.transforms as T
import albumentations as A

def build_model(image_size):
    backbone_params = dict(
        img_size=image_size,
        patch_size=(14, 14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=1280,
        out_dims=[240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=torch.nn.CrossEntropyLoss(),
        align_corners=False,
        upsample='x4',
        channels=64,
    )

    model = NoPosEmbedTrimapPlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )

    return model

def load_model(checkpoint_path, image_size):
    model = build_model(image_size)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    state_dict = ckpt['state_dict']
    # inference is done on single GPU
    if list(state_dict.keys())[0].startswith('module.'): # ddp checkpoint
        state_dict = {k.replace('modlue.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    return model

def load_sample(sample_path, image_size):
    image = cv2.imread(glob.glob(os.path.join(sample_path, 'image.*'))[0],
                       cv2.IMREAD_COLOR_RGB)
    mask = cv2.imread(os.path.join(sample_path, 'mask.png'),
                       cv2.IMREAD_GRAYSCALE)
    hm, wm = mask.shape
    bboxes = [(0, 0, wm, hm, 0),]
    sample_original = dict(image=image, mask=mask, bboxes=bboxes)
    
    wi, hi = image_size
    transform_a = A.Compose([
        A.LongestMaxSize(max_size=max(wi, hi)),
        A.PadIfNeeded(min_height=hi, min_width=wi, border_mode=0),],
        bbox_params=A.BboxParams(format="coco"))
    to_tensor = T.ToTensor()
    sample_transformed = transform_a(**sample_original)

    sample_transformed['image'] = to_tensor(sample_transformed['image'])
    sample_transformed['mask'] = to_tensor(sample_transformed['mask'])

    return dict(original=sample_original, transformed=sample_transformed)

def pred_to_trimap(y_pred):
    y = torch.argmax(y_pred, dim=0).numpy()
    
    # set color according to label
    trimap = np.zeros_like(y, dtype=np.uint8)
    trimap[y == 1] = 128
    trimap[y == 2] = 255

    return trimap

def resize_trimap(trimap, image_size, sample):
    # resize trimap to input size
    trimap = cv2.resize(trimap, image_size)

    # unpad (crop)
    xt, yt, wt, ht, _ = sample['transformed']['bboxes'][0]
    xt, yt, wt, ht = map(round, [xt, yt, wt, ht])
    trimap = trimap[yt:yt+ht, xt:xt+wt]

    # resize and refine
    _, _, wo, ho, _ = sample['original']['bboxes'][0]
    trimap = cv2.resize(trimap, (wo, ho), cv2.INTER_LINEAR_EXACT)
    unknown_idx = (trimap > 0) & (trimap < 255)
    trimap[unknown_idx] = 128
    
    return trimap

def visualize_trimap(trimap, sample, alpha=0.3):
    image = sample['original']['image']

    trimap_color = np.zeros_like(image)

    trimap_color[trimap == 0] = [0, 0, 255]
    trimap_color[trimap == 128] = [255, 0, 0]
    trimap_color[trimap == 255] = [0, 255, 0]

    vis = cv2.addWeighted(image, 1 - alpha, trimap_color, alpha, 0)
    return vis

def main(sample_path, image_size=(448, 448)):
    model = load_model('weight/054.pth', image_size)
    sample = load_sample(sample_path, image_size)

    with torch.no_grad():
        image = sample['transformed']['image'].unsqueeze(0).cuda()
        mask = sample['transformed']['mask'].unsqueeze(0).cuda()

        output = model(image, mask)
        y_pred = output['instances'].cpu().squeeze(0)


    trimap_512 = pred_to_trimap(y_pred)    
    trimap_resized = resize_trimap(trimap_512, image_size, sample)
    vis = visualize_trimap(trimap_resized, sample)

    cv2.imwrite(os.path.join(sample_path, 'trimap.png'), trimap_resized)
    cv2.imwrite(os.path.join(sample_path, 'vis_trimap.png'), vis)

if __name__ == '__main__':
    main('test_data/dogs', (448, 448))