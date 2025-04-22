import os
import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from segment_anything.modeling.image_encoder3D import ImageEncoderViT3D
from segment_anything.build_sam3D import sam_model_registry3D
from functools import partial
import time
import pickle
import nibabel as nib
# 手动输入点的函数
def manual_point_input():
    print("请输入三维点的坐标 (x, y, z)，用空格分隔：")
    point_input = input()
    x, y, z = map(int, point_input.split())

    # (1, 1, 3)
    return torch.tensor([[[x, y, z]]], dtype=torch.float), torch.tensor([[1]], dtype=torch.int)

def finetune_model_predict3D(tiny_vit, img3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10,
                             prev_masks=None):
    # 预处理图像
    img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    # 初始化prev_masks为全零掩膜，如果没有传入
    if prev_masks is None:
        prev_masks = torch.zeros_like(img3D).to(device)  # 创建与输入图像同大小的全零掩膜

    # 用于存储点击点和标签
    click_points = []
    click_labels = []

    # 用于存储预测结果
    pred_list = []
    iou_list = []
    dice_list = []

    for num_click in range(num_clicks):
        with torch.no_grad():
            if num_click == 0:
                # 第一次点击时使用手动输入的点
                new_points_co, new_points_la = manual_point_input()
            else:
                # 后续点击点使用随机选择的点
                new_points_co, new_points_la = random_point_sampling(prev_masks.to(device), get_point=1)
            manual_point_input = [tensor([[[ 24, 112,  94]]])]
            # 将输入的点移动到设备上
            new_points_co = new_points_co.to(device)
            new_points_la = new_points_la.to(device)

            # 存储当前点击的点和标签
            click_points.append(new_points_co)
            click_labels.append(new_points_la)

            points_input = new_points_co
            labels_input = new_points_la

            # 通过SAM模型的prompt encoder获取稀疏和稠密的嵌入
            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,  # 这里没有使用框
                masks=prev_masks.to(device),  # 使用之前的分割掩模
            )

            # 使用mask decoder进行掩模预测
            # 先通过 image_encoder 提取嵌入与位置编码
            image_embeddings = sam_model_tune.image_encoder(img3D.to(device))

            # image_embeddings = sam_model_tune.image_encoder(img2D.float())


            image_pe = sam_model_tune.prompt_encoder.get_dense_pe()

            # 再调用 mask decoder
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # 更新prev_masks
            prev_masks = F.interpolate(low_res_masks, size=img3D.shape[-3:], mode='trilinear', align_corners=False)

            # 转换为二值掩模
            pred_masks = torch.sigmoid(prev_masks)  # 概率转掩模
            pred_masks = pred_masks.cpu().numpy().squeeze()
            pred_masks = (pred_masks > 0.5).astype(np.uint8)  # 设置阈值为0.5，转为二值图

            # 存储预测结果
            pred_list.append(pred_masks)

            # 由于没有真实标签，这里IoU和Dice计算部分可以省略或设置为0
            iou_list.append(0)  # 没有gt3D，暂时没有IoU计算
            dice_list.append(0)  # 没有gt3D，暂时没有Dice计算

    return pred_list, click_points, click_labels, iou_list, dice_list


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
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],dtype=torch.int)

        return torch.as_tensor([fg_coords.tolist()], dtype=torch.float).unsqueeze(0), \
               torch.as_tensor([[labels]], dtype=torch.int)


        #return coords, labels

def norm_transform(img):
    """正则化图像"""
    return (img - img.mean()) / img.std()

# 加载图像数据
def load_image(image_path):
    img = nib.load(image_path)
    img_data = img.get_fdata()
    img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, D, H, W)

    # Resize 到 (1, 1, 128, 128, 128)
    img_data = F.interpolate(img_data, size=(128, 128, 128), mode='trilinear', align_corners=False)

    return img_data  # shape: (1, 1, 128, 128, 128)

# 其他必要的函数

if __name__ == "__main__":
    # 代码初始化部分
    checkpoint_path = "/media/wagnchogn/ssd_2t/lmy/FastSAM3D_copy/work_dir/distillation/sam_model_loss_best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)  # 加载SAM模型
    tiny_vit = ImageEncoderViT3D(
        depth=6,
        embed_dim=768,
        img_size=128,
        mlp_ratio=4,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        out_chans=384,
    ).to(device)

    # 假设提供的图像文件路径
    image_path = "/media/wagnchogn/ssd_2t/lmy/FastSAM3D_copy/data_10per/val_with_organ (copy)/Adrenocortical_carcinoma/CT_AbdTumor_Adrenal/imagesTr/CT_AbdTumor_Adrenal_Ki67_Seg_003.nii.gz"  # 替换为待分割图像的路径
    image3D = load_image(image_path).to(device)

    # 在不使用真实标签的情况下进行预测
    seg_mask_list, click_points, click_labels, iou_list, dice_list = finetune_model_predict3D(
        tiny_vit, image3D, sam_model_tune, device=device, click_method='random', num_clicks=5, prev_masks=None)

    # 输出分割结果
    print("分割结果:", seg_mask_list)
    print("点击点:", click_points)
    print("点击标签:", click_labels)
    print("IoU:", iou_list)
    print("Dice系数:", dice_list)

