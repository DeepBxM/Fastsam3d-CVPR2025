import os

from sympy.codegen.ast import continue_

join = os.path.join
import numpy as np
from glob import glob
import torch
from segment_anything.build_sam3D import sam_model_registry3D  #
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader_v2 import Dataset_Union_ALL_Val
import time
from segment_anything.modeling.image_encoder3D import ImageEncoderViT3D  #
from thop import profile
from torchinfo import summary
# from fvcore.nn import FlopCountAnalysis
# from ptflops import get_model_complexity_info
from functools import partial
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--checkpoint_path', type=str,
                    default="/workspace/model/sam_med3d.pth")  # model check point download from provided ckpt link
parser.add_argument('-tp','--tiny_vit_checkpoint', type=str, default='/workspace/model/fastsam3d.pth', help='Path to the image encoder checkpoint') # download from provided ckpt link
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Tr')

parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)
# parser.add_argument('--load_checkpoint', action='store_true', help='If set, load the model weights from the specified checkpoint path')
parser.add_argument('-i', '--input', type=str, default='/media/wagnchogn/ssd_2t/lmy/fastsam3d_final/inputs/',
                    help='Input directory containing .npz files')
parser.add_argument('-o', '--output', type=str, default='./outputs',
                    help='Output directory for saving results')

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}


def save_preprocessed_image(image3D_tensor, img_name, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    image3D_np = image3D_tensor.cpu().numpy().squeeze()
    img_nifti = nib.Nifti1Image(image3D_np, affine=np.eye(4))
    save_path = os.path.join(save_directory, img_name.replace('.nii.gz', '_preprocessed.nii.gz'))
    nib.save(img_nifti, save_path)


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top: ori_h + top, left: ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def sam_decoder_inference(target_size, points_coords, points_labels, model, image_embeddings, mask_inputs=None,
                          multimask=False):
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )

    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
    return new_arr

def reorganize_points(points, points_label):
    # 将 'fg' 和 'bg' 转换为 1 和 0
    label_to_num = {'fg': 1, 'bg': 0}

    # 确保points_label的长度是15的倍数
    assert len(points_label) % len(points) == 0, "points_label长度必须是points长度的倍数"

    # 将points_label重新组织成多个轮次
    rounds = len(points_label) // len(points)
    label_rounds = []
    for i in range(rounds):
        start = i * len(points)
        end = start + len(points)
        label_rounds.append(points_label[start:end])

    # 初始化结果列表
    result_points = []
    result_labels = []

    # 处理每个点集
    for i, point_dict in enumerate(points):
        # 获取当前点集中的fg和bg点
        fg_points = point_dict.get('fg', [])
        bg_points = point_dict.get('bg', [])

        # 获取当前位置在每个轮次中的标签，并转换为数字
        current_labels = [label_to_num[round_labels[i]] for round_labels in label_rounds]

        # 初始化当前位置的点列表
        current_points = []

        # 跟踪fg和bg点的使用位置
        fg_index = 0
        bg_index = 0

        # 按照轮次顺序添加点
        for label in current_labels:
            if label == 1:  # fg
                if fg_index < len(fg_points):
                    current_points.append(fg_points[fg_index])
                    fg_index += 1
                else:
                    raise ValueError(f"位置{i}的fg点数量不足")
            else:  # bg (label == 0)
                if bg_index < len(bg_points):
                    current_points.append(bg_points[bg_index])
                    bg_index += 1
                else:
                    raise ValueError(f"位置{i}的bg点数量不足")

        # 确保所有点都被使用
        if fg_index < len(fg_points):
            raise ValueError(f"位置{i}的fg点有剩余未使用的点")
        if bg_index < len(bg_points):
            raise ValueError(f"位置{i}的bg点有剩余未使用的点")

        result_points.append(current_points)
        result_labels.append(current_labels)

    return result_points, result_labels



def finetune_model_predict3D(tiny_vit, img3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10,
                             prev_masks=None,points=None,points_label=None,prev_pred =None,original_size=None):
    img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    click_points,click_labels = reorganize_points(points,points_label)
    now_size = img3D.size()[-3:]

    with torch.no_grad():
        image_embedding = tiny_vit(img3D.to(device))
    image_embedding = image_embedding[-1]  # (1, 384, 16, 16, 16)

    pred_list = []

    final_seg = np.zeros(original_size, dtype=np.uint8)  # [img_size, img_size, img_size]

    for class_idx in range(len(click_points)):
        all_datas = click_points[class_idx]
        all_labels = click_labels[class_idx]

        if prev_pred is None:
            prev_masks = torch.zeros_like(img3D).to(device)

        # prev_masks = F.interpolate(prev_masks.float(), size=now_size, mode='trilinear', align_corners=False)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4))

        for point_idx in range(len(all_datas)):
            batch_data = [[all_datas[point_idx]]]
            batch_labels = [[all_labels[point_idx]]]
            points_input = torch.tensor(batch_data, device=device)  # [batch_size, N, 3]
            labels_input = torch.tensor(batch_labels, device=device)  # [batch_size, N]

            with torch.no_grad():

                # low_res_masks = low_res_masks.repeat(batch_size, 1, 1, 1, 1)
                # 只添加batch维度
                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,  #
                    masks=low_res_masks.to(device),
                )

                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                    multimask_output=False,
                )

        prev_masks = F.interpolate(low_res_masks,
                                   size=original_size,  # 使用原始图像尺寸，而不是 img3D.shape[-3:]
                                   mode='trilinear',
                                   align_corners=False)

        medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
        # convert prob to mask

        medsam_seg_prob = medsam_seg_prob.cpu().numpy()
        medsam_seg_prob = np.squeeze(medsam_seg_prob, axis=(0, 1))
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


        final_seg[medsam_seg == 1] = class_idx + 1

    return pred_list, final_seg



if __name__ == "__main__":
    st = time.time()
    '''
    all_dataset_paths = [
        "/root/autodl-tmp/fastsam3d/data/Microscopy_SELMA3D_patchvolume_label_113_description_1/Microscopy_SELMA3D_patchvolume",
        "/root/autodl-tmp/fastsam3d/data/Microscopy_SELMA3D_patchvolume_label_101_description_1/Microscopy_SELMA3D_patchvolume",
    ]
    '''
    all_dataset_paths =args.input

    os.makedirs(args.output, exist_ok=True)

    input_files = os.listdir(args.input)
    if not input_files:
        print("No file found in input directory")
        exit()
    filename = input_files[0]  # 只取第一个文件
    input_path = os.path.join(args.input, filename)
    output_path = os.path.join(args.output, filename)
    points_infos = np.load(input_path)
    import sys
    if 'clicks' not in points_infos:
        imgs = points_infos['imgs']

        zero_mask = np.zeros_like(imgs)  # 假设最后一维是通道维
        np.savez_compressed(
            output_path,
            segs=zero_mask, )

        print(f"No clicks found, saved zero mask to {output_path}")
        sys.exit(0)  # 直接退出程序

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size, args.crop_size, args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        path=all_dataset_paths,
        mode="Val",
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)

    # change checkpoint here
    tiny_vit_checkpoint_path = args.tiny_vit_checkpoint  # Load image encoder weight here, download form the checkpoint link
    tiny_vit = ImageEncoderViT3D(
        depth=6,
        embed_dim=768,
        img_size=128,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=0,
        out_chans=384,
        skip_layer=2,
    )

    model_dict = torch.load(tiny_vit_checkpoint_path, map_location=args.device)
    state_dict = model_dict['model_state_dict']
    tiny_vit.load_state_dict(state_dict)
    print(f"Loaded weights from {args.checkpoint_path}")

    tiny_vit = tiny_vit.to(device)
    all_iou_list = []
    all_dice_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()
    encoder_times = []
    decoder_times = []
    average_decoder_times = []
    memory_befores = []
    memory_decoders = []
    FLOPSS = []


    for batch_data in tqdm(test_dataloader):
        image3D,  img_name = batch_data
        points_path = img_name[0]
        points_infos = np.load(points_path, allow_pickle=True)
        sz = image3D.size()
        if (sz[2] < args.crop_size or sz[3] < args.crop_size or sz[4] < args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)


        points = points_infos['clicks']
        points_label = points_infos['clicks_order']
        prev_pred = points_infos['prev_pred']

        #prev_masks_pred = points_infos.get('prev_masks_pred')
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        imgs = points_infos['imgs']
        original_size = imgs.shape[-3:]
        seg_mask_list, final_seg = finetune_model_predict3D(
            tiny_vit,
            image3D,  sam_model_tune, device=device,
            click_method=args.point_method, num_clicks=args.num_clicks,
            prev_masks=None,points = points,points_label = points_label,original_size=original_size)

        if final_seg is not None:
            np.savez_compressed(
                output_path,
                segs=final_seg,
            )
        else:
            np.savez_compressed(
                output_path,
                segs=None
            )
