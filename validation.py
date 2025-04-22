import os

from monai.losses import DiceCELoss
from thop import profile

from segment_anything import sam_model_registry

join = os.path.join
import numpy as np
from glob import glob
import torch
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
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
from utils.data_loader import Dataset_Union_ALL_Val
import time

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='data_10per/val_with_organ (copy)')
parser.add_argument('-vp', '--vis_path', type=str, default='results/val_tea')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('-sn', '--save_name', type=str, default='results/val_tea/sam_med3d.py')

parser.add_argument('--image_size', type=int, default=128)  #
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_original')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Tr')
parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

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


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None, device='cuda'):
    # device = "cuda"
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
    return low_res_masks, prev_masks


def get_points(click_type, prev_masks, gt3D, click_points, click_labels, device):
    batch_points, batch_labels = click_methods[click_type](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_multi = torch.cat(click_points, dim=1).to(device)
    labels_multi = torch.cat(click_labels, dim=1).to(device)

    points_input = points_multi
    labels_input = labels_multi

    return points_input, labels_input, click_points, click_labels


def interaction(sam_model, image_embedding, gt3D, num_clicks):
    click_type = 'random'
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    img_size = 128
    device = "cuda"
    return_loss = 0
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(img_size // 4, img_size // 4, img_size // 4))

    click_points = []
    click_labels = []

    for num_click in range(num_clicks):
        random_insert = np.random.randint(2, 9)
        points_input, labels_input, click_points, click_labels = get_points(click_type, prev_masks, gt3D, click_points,
                                                                            click_labels, device)

        if num_click == random_insert or num_click == num_clicks - 1:
            low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
        else:
            low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                      points=[points_input, labels_input])
        loss = seg_loss(prev_masks, gt3D)
        return_loss += loss
    return prev_masks, return_loss


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


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
        torch.cuda.reset_max_memory_allocated(model.device)
        low_res_masks, iou_predictions, t = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
        memory = torch.cuda.max_memory_allocated(model.device)

    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions, t, sparse_embeddings, dense_embeddings


def sam_decoder_inference_n(target_size, points_coords, points_labels, model, image_embeddings, sparse_embeddings,
                            dense_embeddings, mask_inputs=None, multimask=False):
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated(model.device)
        low_res_masks, iou_predictions, t = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
        memory = torch.cuda.max_memory_allocated(model.device)
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions, t


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
    return new_arr


def random_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels


def finetune_model_predict2D(img3D, gt3D, sam_model_tune, target_size=256, click_method='random', device='cuda',
                             num_clicks=1, prev_masks=None):
    pred_list = []
    iou_list = []
    dice_list = []

    slice_mask_list = defaultdict(list)
    k = 0
    img3D = torch.repeat_interleave(img3D, repeats=3, dim=1)  # 1 channel -> 3 channel (align to RGB)
    sparse_embeddings = []
    dense_embeddings = []
    click_points = []
    click_labels = []
    for slice_idx in tqdm(range(img3D.size(-1)), desc="transverse slices", leave=False):
        img2D, gt2D = repixel_value(img3D[..., slice_idx]), gt3D[..., slice_idx]

        if (gt2D == 0).all():
            empty_result = torch.zeros(list(gt3D.size()[:-1]) + [1]).to(device)
            for iter in range(num_clicks):
                slice_mask_list[iter].append(empty_result)
            continue

        img2D = F.interpolate(img2D, (target_size, target_size), mode="bilinear", align_corners=False)
        gt2D = F.interpolate(gt2D.float(), (target_size, target_size), mode="nearest").int()

        img2D, gt2D = img2D.to(device), gt2D.to(device)
        img2D = (img2D - img2D.mean()) / img2D.std()

        with torch.no_grad():
            image_embeddings, _ = sam_model_tune.image_encoder(img2D.float())

        points_co, points_la = torch.zeros(1, 0, 2).to(device), torch.zeros(1, 0).to(device)
        low_res_masks = None
        gt_semantic_seg = gt2D[0, 0].to(device)
        true_masks = (gt_semantic_seg > 0)
        if k == 0:
            k = 1
            for iter in range(num_clicks):
                if (low_res_masks == None):
                    pred_masks = torch.zeros_like(true_masks).to(device)
                else:
                    pred_masks = (prev_masks[0, 0] > 0.0).to(device)
                fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
                fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
                mask_to_sample = torch.logical_or(fn_masks, fp_masks)
                new_points_co, _ = random_point_sampling(mask_to_sample.cpu(), get_point=1)
                new_points_la = torch.Tensor([1]).to(torch.int64) if (
                true_masks[new_points_co[0, 1].int(), new_points_co[0, 0].int()]) else torch.Tensor([0]).to(torch.int64)
                new_points_co, new_points_la = new_points_co[None].to(device), new_points_la[None].to(device)
                points_co = torch.cat([points_co, new_points_co], dim=1)
                points_la = torch.cat([points_la, new_points_la], dim=1)
                prev_masks, low_res_masks, iou_predictions, _, sparse, dense, = sam_decoder_inference(
                    target_size, points_co, points_la, sam_model_tune, image_embeddings,
                    mask_inputs=low_res_masks, multimask=True)
                sparse_embeddings.append(sparse)
                dense_embeddings.append(dense)
                click_points.append(new_points_co)
                click_labels.append(new_points_la)

                slice_mask, _ = postprocess_masks(low_res_masks, target_size, (gt3D.size(2), gt3D.size(3)))
                slice_mask_list[iter].append(slice_mask[..., None])  # append (B, C, H, W, 1)
        else:
            for iter in range(num_clicks):
                if (low_res_masks == None):
                    pred_masks = torch.zeros_like(true_masks).to(device)
                else:
                    pred_masks = (prev_masks[0, 0] > 0.0).to(device)
                fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
                fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
                mask_to_sample = torch.logical_or(fn_masks, fp_masks)
                new_points_co, _ = random_point_sampling(mask_to_sample.cpu(), get_point=1)
                new_points_la = torch.Tensor([1]).to(torch.int64) if (
                true_masks[new_points_co[0, 1].int(), new_points_co[0, 0].int()]) else torch.Tensor([0]).to(torch.int64)
                new_points_co, new_points_la = new_points_co[None].to(device), new_points_la[None].to(device)
                points_co = torch.cat([points_co, new_points_co], dim=1)
                points_la = torch.cat([points_la, new_points_la], dim=1)
                prev_masks, low_res_masks, iou_predictions, _ = sam_decoder_inference_n(
                    target_size, points_co, points_la, sam_model_tune, image_embeddings, sparse_embeddings[iter],
                    dense_embeddings[iter],
                    mask_inputs=low_res_masks, multimask=True)
                click_points.append(new_points_co)
                click_labels.append(new_points_la)

                slice_mask, _ = postprocess_masks(low_res_masks, target_size, (gt3D.size(2), gt3D.size(3)))
                slice_mask_list[iter].append(slice_mask[..., None])  # append (B, C, H, W, 1)

    for iter in range(num_clicks):
        medsam_seg = torch.cat(slice_mask_list[iter], dim=-1).cpu().numpy().squeeze()
        medsam_seg = medsam_seg > sam_model_tune.mask_threshold
        medsam_seg = medsam_seg.astype(np.uint8)

        pred_list.append(medsam_seg)
        iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
        dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    return pred_list, click_points, click_labels, iou_list, dice_list, 0, 0


def finetune_model_predict3D(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10,
                             prev_masks=None):
    torch.cuda.reset_max_memory_allocated(device)
    encoder_time = 0  #
    decoder_time = []
    img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    click_points = []
    click_labels = []
    FLOPS = np.zeros(num_clicks)

    pred_list = []
    iou_list = []
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(),
                                  size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4))
    start_time = time.time()

    with torch.no_grad():
        image_embedding= sam_model_tune.image_encoder(img3D.to(device))  # (1, 384, 16, 16, 16)
    image_embedding = image_embedding[-1]
    memory_before = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_max_memory_allocated(device)
    for num_click in range(num_clicks):
        #
        with torch.no_grad():
            if (num_click > 1):
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,  #
                masks=low_res_masks.to(device),
            )
            FLOPS[num_click] += \
            profile(sam_model_tune.prompt_encoder, ([points_input, labels_input], None, low_res_masks.to(device),))[0]
            start_time = time.time()
            torch.cuda.reset_max_memory_allocated(device)
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)

                multimask_output=False,
            )
            FLOPS[num_click] += profile(sam_model_tune.mask_decoder, (
            image_embedding, sam_model_tune.prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings,
            False,))[0]
            print('flops' + str(profile(sam_model_tune.mask_decoder, (
            image_embedding, sam_model_tune.prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings,
            False,))[0]))
            memory_decoder = torch.cuda.max_memory_allocated(device)  #
            print('memorydecoder' + str(memory_decoder))
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
    # print(np.average(FLOPS))
    return pred_list, click_points, click_labels, iou_list, dice_list, encoder_time, decoder_time, memory_before, memory_decoder, FLOPS


if __name__ == "__main__":
    st = time.time()
    all_dataset_paths = [
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_101_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_113_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_55_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Inferior_vena_cava/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Inferior_vena_cava/CT_AMOS",
        "data_10per/val_with_organ (copy)/Inferior_vena_cava/MR_AMOS",
        "data_10per/val_with_organ (copy)/Inferior_vena_cava/MR_totalseg",
        "data_10per/val_with_organ (copy)/Surrounding_non-enhancing_FLAIR_hyperintensity/MR_BraTS-T2w",
        "data_10per/val_with_organ (copy)/Vertebrae_S1/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/MR_Heart_la_label_1_description_1/MR_Heart_la",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_53_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_1_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_HCC_label_1_description_1/CT_AbdTumor_HCC",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_77_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_26_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Mandible/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Brachiocephalic_trunk/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_31_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_112_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_T2/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Right_anterior_segment_of_the_eyeball/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_7_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_96_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_127_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_15_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Esophagus/CT_AMOS",
        "data_10per/val_with_organ (copy)/Esophagus/MR_AMOS",
        "data_10per/val_with_organ (copy)/Esophagus/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Esophagus/MR_totalseg",
        "data_10per/val_with_organ (copy)/Esophagus/CT_TCIA-LCTSC",
        "data_10per/val_with_organ (copy)/Surrounding_non-enhancing_FLAIR_hyperintensit/MR_BraTS-T1n",
        "data_10per/val_with_organ (copy)/Surrounding_non-enhancing_FLAIR_hyperintensit/MR_BraTS-T1c",
        "data_10per/val_with_organ (copy)/Surrounding_non-enhancing_FLAIR_hyperintensit/MR_BraTS-T2f",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_5_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_24_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_21_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_37_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Bladder/CT_AMOS",
        "data_10per/val_with_organ (copy)/Bladder/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_54_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_C7/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_17_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Lung_upper_lobe_left/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_126_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_64_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_122_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_GTVpNCCT_segrap_label_1_description_1/CT_GTVpNCCT_segrap",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_21_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_2_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_93_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_83_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/GTVp_and_GTVn_tumor/MR_Head_HNTSMRG24",
        "data_10per/val_with_organ (copy)/Larynx-supraglottic/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_106_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_94_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_105_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_79_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Adrenocortical_carcinoma/CT_AbdTumor_Adrenal",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_137_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_25_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_109_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_45_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_11_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_133_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_anterior_segment_of_the_eyeball/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Lips/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_16_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_17_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_81_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_47_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_iliopsoas/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Liver_tumors/CT_AbdTumor_liver",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_135_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_13_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_67_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_kidney/MR_CHAOS-T1",
        "data_10per/val_with_organ (copy)/Right_kidney/MR_CHAOS-T2",
        "data_10per/val_with_organ (copy)/Right_kidney/CT_AMOS",
        "data_10per/val_with_organ (copy)/Right_kidney/MR_AMOS",
        "data_10per/val_with_organ (copy)/Right_kidney/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Right_kidney/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_kidney/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Enhancing_tissue/MR_BraTS-T1n",
        "data_10per/val_with_organ (copy)/Enhancing_tissue/MR_BraTS-T1c",
        "data_10per/val_with_organ (copy)/Enhancing_tissue/MR_BraTS-T2f",
        "data_10per/val_with_organ (copy)/Enhancing_tissue/MR_BraTS-T2w",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_97_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_T11/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_renal_structure_identified_via_abdominal_magnetic_resonance/MR_AMOS",
        "data_10per/val_with_organ (copy)/Pancreas_tumors/CT_AbdTumor_pancreas",
        "data_10per/val_with_organ (copy)/Sacrum/MR_totalseg",
        "data_10per/val_with_organ (copy)/Sacrum/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Vertebrae_T5/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Thyroid/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_76_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_61_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_49_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_4_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Left_clavicula/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_clavicula/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_clavicula/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_clavicula/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_scapula/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_scapula/MR_totalseg",
        "data_10per/val_with_organ (copy)/Brainstem/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_19_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_L1/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Brain/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_5_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/White_matter_hyperintensities/MR_WMH_T1",
        "data_10per/val_with_organ (copy)/White_matter_hyperintensities/MR_WMH_FLAIR",
        "data_10per/val_with_organ (copy)/Left_autochthon/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_autochthon/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_66_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_65_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Lung_upper_lobe_right/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_115_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Gallbladder/CT_AMOS",
        "data_10per/val_with_organ (copy)/Gallbladder/MR_AMOS",
        "data_10per/val_with_organ (copy)/Gallbladder/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Gallbladder/MR_totalseg",
        "data_10per/val_with_organ (copy)/Gallbladder/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_35_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_3_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Portal_vein_and_splenic_vein/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Portal_vein_and_splenic_vein/MR_totalseg",
        "data_10per/val_with_organ (copy)/Lung_middle_lobe_right/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_118_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_44_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Skull/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_125_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_78_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_128_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Superior_vena_cava/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Left_optic_nerve/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Prostate/uterus",
        "data_10per/val_with_organ (copy)/Prostate/MR_ProstateT2",
        "data_10per/val_with_organ (copy)/Prostate/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Prostate/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_84_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Thyroid_gland/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_110_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Arytenoids_delineation/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_50_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_91_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_48_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Pituitary_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Vertebrae_T8/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Right_humerus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_humerus/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_36_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Spleen/MR_CHAOS-T1",
        "data_10per/val_with_organ (copy)/Spleen/MR_CHAOS-T2",
        "data_10per/val_with_organ (copy)/Spleen/CT_AMOS",
        "data_10per/val_with_organ (copy)/Spleen/MR_AMOS",
        "data_10per/val_with_organ (copy)/Spleen/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Spleen/MR_totalseg",
        "data_10per/val_with_organ (copy)/Spleen/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Vertebrae_L5/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_adrenal_gland/CT_AMOS",
        "data_10per/val_with_organ (copy)/Left_adrenal_gland/MR_AMOS",
        "data_10per/val_with_organ (copy)/Left_adrenal_gland/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Left_adrenal_gland/MR_totalseg",
        "data_10per/val_with_organ (copy)/Left_adrenal_gland/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_73_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_7_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Right_common_carotid_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Larynx-glottis/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Left_parotid_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_72_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_3_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Airway_Tree/CT_AirwayTree",
        "data_10per/val_with_organ (copy)/Right_subclavian_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_63_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_T6/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_hip/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_hip/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_46_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_femur/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_femur/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_4_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Lung_lesions/CT_LungLesion",
        "data_10per/val_with_organ (copy)/Intervertebral_discs/MR_Spider_IVD",
        "data_10per/val_with_organ (copy)/Intervertebral_discs/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_59_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Cervical_esophagus/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_case_label_3_description_1/CT_AbdTumor_case",
        "data_10per/val_with_organ (copy)/Left_posterior_segment_of_the_eyeball/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Prostate_lesion/MR_QIN-PROSTATE",
        "data_10per/val_with_organ (copy)/Spinal_canal/MR_Spider_Spine",
        "data_10per/val_with_organ (copy)/Right_submandibular_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Myocardium/MR_heart-ACDC",
        "data_10per/val_with_organ (copy)/Myocardium/US_Cardiac",
        "data_10per/val_with_organ (copy)/Vertebrae_C5/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_Ventricle/US_Cardiac",
        "data_10per/val_with_organ (copy)/Vertebrae_T1/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_gluteus_maximus/MR_totalseg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_case_label_1_description_1/CT_AbdTumor_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_95_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_27_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_62_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_82_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_humerus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_humerus/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_autochthon/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Non-enhancing_tumor_core/MR_BraTS-T1n",
        "data_10per/val_with_organ (copy)/Non-enhancing_tumor_core/MR_BraTS-T1c",
        "data_10per/val_with_organ (copy)/Non-enhancing_tumor_core/MR_BraTS-T2f",
        "data_10per/val_with_organ (copy)/Non-enhancing_tumor_core/MR_BraTS-T2w",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_39_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_T7/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Resection_cavity/MR_BraTS-T1n",
        "data_10per/val_with_organ (copy)/Resection_cavity/MR_BraTS-T1c",
        "data_10per/val_with_organ (copy)/Resection_cavity/MR_BraTS-T2f",
        "data_10per/val_with_organ (copy)/Resection_cavity/MR_BraTS-T2w",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_58_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Lymph_node/CT_LNQ_LymphNode",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_124_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_86_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_25_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_adrenal_gland/CT_AMOS",
        "data_10per/val_with_organ (copy)/Right_adrenal_gland/MR_AMOS",
        "data_10per/val_with_organ (copy)/Right_adrenal_gland/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Right_adrenal_gland/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_adrenal_gland/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_120_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_134_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_71_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_iliac_vena/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Right_iliac_vena/MR_totalseg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_5_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_57_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_12_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Left_subclavian_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Vertebrae_T9/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_7_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_136_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_19_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Right_gluteus_maximus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_gluteus_maximus/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_optic_nerve/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_9_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Vertebrae_C3/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_13_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Vertebrae/MR_Spider_Vertebrae",
        "data_10per/val_with_organ (copy)/Vertebrae/MR_totalseg",
        "data_10per/val_with_organ (copy)/Stroke_lesion/MR_ISLES2022_DWI",
        "data_10per/val_with_organ (copy)/Stroke_lesion/MR_ISLES2022_ADC",
        "data_10per/val_with_organ (copy)/Vertebrae_C4/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Transition_zone/MR_ProstateADC",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_10_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_27_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_33_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_case_label_2_description_1/CT_AbdTumor_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_22_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_92_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_60_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_119_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_117_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_26_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Oral_cavity_delineation/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_34_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Pancreas/CT_AMOS",
        "data_10per/val_with_organ (copy)/Pancreas/MR_AMOS",
        "data_10per/val_with_organ (copy)/Pancreas/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Pancreas/MR_totalseg",
        "data_10per/val_with_organ (copy)/Pancreas/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Left_atrial_appendage/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_1_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_80_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_132_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_52_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_L4/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Pulmonary_vein/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_129_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_parotid_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Right_brachiocephalic_vein/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Vertebrae_T3/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_87_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_lung/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_lung/CT_TCIA-LCTSC",
        "data_10per/val_with_organ (copy)/Right_lung/CT_LungMasks",
        "data_10per/val_with_organ (copy)/Extra-meatal_region_of_vestibular_schwannoma/MR_crossmoda",
        "data_10per/val_with_organ (copy)/Vertebrae_T12/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_70_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_130_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_T4/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_8_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_89_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Trachea/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_28_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_10_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Liver/MR_CHAOS-T1",
        "data_10per/val_with_organ (copy)/Liver/MR_CHAOS-T2",
        "data_10per/val_with_organ (copy)/Liver/CT_AMOS",
        "data_10per/val_with_organ (copy)/Liver/MR_AMOS",
        "data_10per/val_with_organ (copy)/Liver/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Liver/MR_totalseg",
        "data_10per/val_with_organ (copy)/Liver/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Spinal_cord/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Spinal_cord/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Spinal_cord/MR_totalseg",
        "data_10per/val_with_organ (copy)/Spinal_cord/CT_TCIA-LCTSC",
        "data_10per/val_with_organ (copy)/Gastrocnemius_Lateralis/US_Low-limb-Leg",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_11_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Left_gluteus_Maximus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Intra-meatal_region_of_vestibular_schwannoma/MR_crossmoda",
        "data_10per/val_with_organ (copy)/Lesion/PET_autoPET",
        "data_10per/val_with_organ (copy)/Aorta/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Aorta/CT_AMOS",
        "data_10per/val_with_organ (copy)/Aorta/MR_AMOS",
        "data_10per/val_with_organ (copy)/Aorta/MR_totalseg",
        "data_10per/val_with_organ (copy)/Aorta/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_56_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_brachiocephalic_vein/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Right_hip/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_hip/MR_totalseg",
        "data_10per/val_with_organ (copy)/CT_GTVpEnhance_segrap_label_1_description_1/CT_GTVpEnhance_segrap",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_38_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_111_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_2_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Vertebrae_T10/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Vertebrae_L2/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_16_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_74_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_29_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Stomach/CT_AMOS",
        "data_10per/val_with_organ (copy)/Stomach/MR_AMOS",
        "data_10per/val_with_organ (copy)/Stomach/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Stomach/MR_totalseg",
        "data_10per/val_with_organ (copy)/Stomach/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_42_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_41_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_12_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_68_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_107_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_iliac_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Right_iliac_artery/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_30_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_108_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Soleus/US_Low-limb-Leg",
        "data_10per/val_with_organ (copy)/COVID-19_infection/CT_COVID",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_69_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_femur/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_femur/MR_totalseg",
        "data_10per/val_with_organ (copy)/Left_lung/MR_totalseg",
        "data_10per/val_with_organ (copy)/Left_lung/CT_TCIA-LCTSC",
        "data_10per/val_with_organ (copy)/Left_lung/CT_LungMasks",
        "data_10per/val_with_organ (copy)/Small_bowel/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Small_bowel/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_cochlea/MR_crossmoda",
        "data_10per/val_with_organ (copy)/Right_cochlea/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_6_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_121_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Right_gluteus_medius/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_gluteus_medius/MR_totalseg",
        "data_10per/val_with_organ (copy)/Postcava/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_75_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Buccal_mucosa/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Colon_cancer_primaries/CT_AbdTumor_colon",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_98_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_23_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_18_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_103_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_43_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_C1/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_6_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_123_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_22_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Right_gluteus_minimus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Right_gluteus_minimus/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_104_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_1_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_4_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Left_gluteus_medius/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_gluteus_medius/MR_totalseg",
        "data_10per/val_with_organ (copy)/Right_ventricle_cavity/MR_heart-ACDC",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_14_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Vertebrae_C2/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_20_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Left_common_carotid_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_14_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_114_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_51_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Gastrocnemius_Medialis/US_Low-limb-Leg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_116_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_Atrium/US_Cardiac",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_40_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_3_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Lung_lower_lobe_left/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_6_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Vertebrae_L3/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Colon/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Colon/MR_totalseg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_case_label_4_description_1/CT_AbdTumor_case",
        "data_10per/val_with_organ (copy)/Left_kidney/MR_CHAOS-T1",
        "data_10per/val_with_organ (copy)/Left_kidney/MR_CHAOS-T2",
        "data_10per/val_with_organ (copy)/Left_kidney/CT_AMOS",
        "data_10per/val_with_organ (copy)/Left_kidney/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Left_kidney/MR_totalseg",
        "data_10per/val_with_organ (copy)/Left_kidney/CT_AbdomenAtlas",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_100_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_hepaticvessel_label_1_description_1/CT_AbdTumor_hepaticvessel",
        "data_10per/val_with_organ (copy)/Left_lacrimal_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_9_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_18_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Lung_lower_lobe_right/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Optic_chiasm/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_23_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Urinary_bladder/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Urinary_bladder/MR_totalseg",
        "data_10per/val_with_organ (copy)/Vertebrae_C6/CT_totalseg-vertebrae",
        "data_10per/val_with_organ (copy)/Left_iliac_artery/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Left_iliac_artery/MR_totalseg",
        "data_10per/val_with_organ (copy)/Cricopharyngeal_inlet/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Aortic_vessel_trees/CT_Aorta",
        "data_10per/val_with_organ (copy)/Left_scapula/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_scapula/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_131_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_99_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_8_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Left_kidney_cyst/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_24_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Duodenum/CT_AMOS",
        "data_10per/val_with_organ (copy)/Duodenum/MR_AMOS",
        "data_10per/val_with_organ (copy)/Duodenum/CT_TotalSeg_organs",
        "data_10per/val_with_organ (copy)/Duodenum/MR_totalseg",
        "data_10per/val_with_organ (copy)/Left_iliac_vena/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Left_iliac_vena/MR_totalseg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_9_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_iliopsoas/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_88_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_102_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_28_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/MR_HeadHaN_case_label_15_description_1/MR_HeadHaN_case",
        "data_10per/val_with_organ (copy)/Left_carotid_artery/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_8_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_90_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Left_gluteus_minimus/CT_TotalSeg_muscles",
        "data_10per/val_with_organ (copy)/Left_gluteus_minimus/MR_totalseg",
        "data_10per/val_with_organ (copy)/CT_AbdTumor_PETCT_label_2_description_1/CT_AbdTumor_PETCT",
        "data_10per/val_with_organ (copy)/Left_submandibular_gland/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_85_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Heart/CT_TotalSeg_cardiac",
        "data_10per/val_with_organ (copy)/Heart/MR_totalseg",
        "data_10per/val_with_organ (copy)/Heart/CT_TCIA-LCTSC",
        "data_10per/val_with_organ (copy)/Right_carotid_artery/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Left_cochlea/CT_HaN-Seg",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_20_description_1/Microscopy_SELMA3D_patchvolume",
        "data_10per/val_with_organ (copy)/Microscopy_SELMA3D_patchvolume_label_32_description_1/Microscopy_SELMA3D_patchvolume"

    ]
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size, args.crop_size, args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths,
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

    if (args.dim == 3):
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model_tune.load_state_dict(state_dict)
    elif (args.dim == 2):
        args.sam_checkpoint = args.checkpoint_path
        sam_model_tune = sam_model_registry[args.model_type](args.checkpoint_path).to(device)

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)
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
    w = []
    for batch_data in tqdm(test_dataloader):
        for i in range(0, 1):
            image3D, gt3D, img_name = batch_data
            image3D = image3D.float()
            image3D = image3D.to(device)
            sz = image3D.size()
            if (sz[2] < args.crop_size or sz[3] < args.crop_size or sz[4] < args.crop_size):
                print("[ERROR] wrong size", sz, "for", img_name)
            modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
            dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
            vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
            click_suffix = f"_pred{args.num_clicks - 1}.nii.gz"
            pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", click_suffix))

            sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
            model_dict = torch.load('./ckpt/sam_med3d.pth', map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model.load_state_dict(state_dict)
            image_embedding= sam_model.image_encoder(image3D)
            start_time = time.time()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"self.interaction excution time:{elapsed_time} seconds")
            if (1 == 0):
                iou_list, dice_list = [], []
                for iter in range(args.num_clicks):
                    curr_pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz",
                                                                                                  f"_pred{iter}.nii.gz"))
                    medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                    iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
                    dice_list.append(
                        round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
            else:
                norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
                if (args.dim == 3):
                    seg_mask_list, points, labels, iou_list, dice_list, t, decoder_time, memory_before, memory_decoder, FLOPS = finetune_model_predict3D(
                        image3D, gt3D, sam_model_tune, device=device,
                        click_method=args.point_method, num_clicks=args.num_clicks,
                        prev_masks=None)
                elif (args.dim == 2):
                    seg_mask_list, points, labels, iou_list, dice_list, t, decoder_time, = finetune_model_predict2D(
                        image3D, gt3D, sam_model_tune, device=device, target_size=args.image_size,
                        click_method=args.point_method, num_clicks=args.num_clicks,
                        prev_masks=None)
                os.makedirs(vis_root, exist_ok=True)
                points = [p.cpu().numpy() for p in points]
                labels = [l.cpu().numpy() for l in labels]
                pt_info = dict(points=points, labels=labels)
                print("save to",
                      os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz")))
                pt_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl"))
                pickle.dump(pt_info, open(pt_path, "wb"))
                for idx, pred3D in enumerate(seg_mask_list):
                    out = sitk.GetImageFromArray(pred3D)
                    sitk.WriteImage(out, os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz",
                                                                                                      f"_pred{idx}.nii.gz")))
            per_iou = max(iou_list)
            all_iou_list.append(per_iou)
            all_dice_list.append(max(dice_list))
            print(dice_list)
            out_dice[img_name] = max(dice_list)
            cur_dice_dict = OrderedDict()
            encoder_times.append(t)
            decoder_times.append(decoder_time)
            for i, dice in enumerate(dice_list):
                cur_dice_dict[f'{i}'] = dice
            out_dice_all[img_name[0]] = cur_dice_dict
    print('Mean IoU : ', sum(all_iou_list) / len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list) / len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if (args.split_num > 1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print(np.mean(encoder_times))
    print(np.mean(decoder_times))
    print("Done")
    eo = time.time() - st
    print(eo)