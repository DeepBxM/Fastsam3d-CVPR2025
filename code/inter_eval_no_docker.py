import os
os.environ['CUPY_GPU_MEMORY_LIMIT'] = str(8 * 1024 * 1024 * 1024)  # 8GB in bytes

join = os.path.join
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import subprocess
import shutil
from scipy.ndimage import distance_transform_edt
import cc3d
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from scipy import integrate
from tqdm import tqdm

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)

def sample_coord(edt):
    # Find all coordinates with max EDT value
    np.random.seed(42)

    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)

    # Uniformly choose one of them
    chosen_index = max_coords[np.random.choice(len(max_coords))]

    center = tuple(chosen_index)
    return center

# Compute the EDT with same shape as the image
def compute_edt(error_component):
    # Get bounding box of the largest error component to limit computation
    coords = np.argwhere(error_component)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_shape = max_coords - min_coords

    # Compute padding (25% of crop size in each dimension)
    padding = (crop_shape * 0.25).astype(int)

    # Define new padded shape
    padded_shape = crop_shape + 2 * padding

    # Create new empty array with padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)

    # Fill center region with actual cropped data
    center_crop[
    padding[0]:padding[0] + crop_shape[0],
    padding[1]:padding[1] + crop_shape[1],
    padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
        ]

    # Compute EDT on the padded array
    if torch.cuda.is_available():  # GPU available
        try:
            import cupy as cp
            from cucim.core.operations import morphology

            # 清理 GPU 内存
            cp.get_default_memory_pool().free_all_blocks()

            error_mask_cp = cp.array(center_crop)
            edt_cp = morphology.distance_transform_edt(error_mask_cp)
            edt = cp.asnumpy(edt_cp)

            # 清理 GPU 变量
            del error_mask_cp
            del edt_cp
            cp.get_default_memory_pool().free_all_blocks()

        except (cp.cuda.memory.OutOfMemoryError, Exception) as e:
            print(f"GPU 内存不足或出现错误 ({str(e)})，自动切换到 CPU 处理...")
            edt = distance_transform_edt(center_crop)
    else:  # CPU available only
        edt = distance_transform_edt(center_crop)

    # Crop out the center (remove padding)
    dist_cropped = edt[
                   padding[0]:padding[0] + crop_shape[0],
                   padding[1]:padding[1] + crop_shape[1],
                   padding[2]:padding[2] + crop_shape[2]
                   ]

    # Create full-sized EDT result array and splat back
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
    min_coords[0]:max_coords[0],
    min_coords[1]:max_coords[1],
    min_coords[2]:max_coords[2]
    ] = dist_cropped

    dist_transformed = dist_full

    return dist_transformed


# [前面的辅助函数保持不变]

def main():
    parser = argparse.ArgumentParser('Segmentation iterative refinement with clicks evaluation')
    parser.add_argument('-i', '--test_img_path', default='./data/imgs', type=str, help='testing data path')
    parser.add_argument('-o', '--save_path', default='./data/output', type=str, help='segmentation output path')
    parser.add_argument('-val_gts', '--validation_gts_path', default='./data/validation_gts', type=str,
                        help='path to validation set GT files')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose output")

    args = parser.parse_args()

    test_img_path = args.test_img_path
    save_path = args.save_path
    validation_gts_path = args.validation_gts_path
    verbose = args.verbose

    # 创建临时目录
    input_temp = './inputs'
    output_temp = './outputs'
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(input_temp):
        shutil.rmtree(input_temp)
    if os.path.exists(output_temp):
        shutil.rmtree(output_temp)
    os.makedirs(input_temp)
    os.makedirs(output_temp)

    # 设置评估指标
    metric = OrderedDict()
    metric['CaseName'] = []
    metric['TotalRunningTime'] = []
    for i in range(6):
        metric[f'RunningTime_{i + 1}'] = []
        metric[f'DSC_{i + 1}'] = []
        metric[f'NSD_{i + 1}'] = []
    metric['DSC_AUC'] = []
    metric['NSD_AUC'] = []
    metric['DSC_Final'] = []
    metric['NSD_Final'] = []
    metric['num_class'] = []
    metric['runtime_upperbound'] = []
    n_clicks = 5

    test_cases = sorted(os.listdir(test_img_path))
    for case in tqdm(test_cases):
        real_running_time = 0
        dscs = []
        nsds = []
        all_segs = []
        no_bbox = False
        if os.path.exists(os.path.join(save_path,case)):
            print(f'exists files')
            #continue
        # 复制输入图像
        shutil.copy(join(test_img_path, case), input_temp)
        if validation_gts_path is None:
            gts = np.load(join(input_temp, case),allow_pickle=True)['gts']
        else:
            gts = np.load(join(validation_gts_path, case),allow_pickle=True)['gts']

        num_classes = len(np.unique(gts)) - 1
        metric['num_class'].append(num_classes)
        metric['runtime_upperbound'].append(num_classes * 90)

        clicks_cls = [{'fg': [], 'bg': []} for _ in np.unique(gts)[1:]]
        clicks_order = []
        if "boxes" in np.load(join(input_temp, case),allow_pickle=True).keys():
            boxes = np.load(join(input_temp, case),allow_pickle=True)['boxes']
        for it in range(n_clicks + 1):
            if it == 0:
                if "boxes" not in np.load(join(input_temp, case),allow_pickle=True).keys():
                    if verbose:
                        print(f'This sample does not use a Bounding Box for the initial iteration {it}')
                    no_bbox = True
                    metric["RunningTime_1"].append(0)
                    metric["DSC_1"].append(0)
                    metric["NSD_1"].append(0)
                    dscs.append(0)
                    nsds.append(0)
                    continue
                if verbose:
                    print(f'Using Bounding Box for iteration {it}')
            else:
                if verbose:
                    print(f'Using Clicks for iteration {it}')
                if os.path.isfile(join(output_temp, case)):
                    segs = np.load(join(output_temp, case),allow_pickle=True)['segs'].astype(np.uint8)
                else:
                    segs = np.zeros_like(gts).astype(np.uint8)
                all_segs.append(segs.astype(np.uint8))

                for ind, cls in enumerate(sorted(np.unique(gts)[1:])):
                    if cls == 0:
                        continue  # skip background

                    segs_cls = (segs == cls).astype(np.uint8)
                    gts_cls = (gts == cls).astype(np.uint8)

                    # Compute error mask
                    error_mask = (segs_cls != gts_cls).astype(np.uint8)
                    if np.sum(error_mask) > 0:
                        errors = cc3d.connected_components(error_mask, connectivity=26)  # 26 for 3D connectivity

                        # Calculate the sizes of connected error components
                        component_sizes = np.bincount(errors.flat)

                        # Ignore non-error regions
                        component_sizes[0] = 0

                        # Find the largest error component
                        largest_component_error = np.argmax(component_sizes)

                        # Find the voxel coordinates of the largest error component
                        largest_component = (errors == largest_component_error)

                        edt = compute_edt(largest_component)
                        center = sample_coord(edt)

                        if gts_cls[center] == 0:  # oversegmentation -> place background click
                            assert segs_cls[center] == 1
                            clicks_cls[ind]['bg'].append(list(center))
                            clicks_order.append('bg')
                        else:  # undersegmentation -> place foreground click
                            assert segs_cls[center] == 0
                            clicks_cls[ind]['fg'].append(list(center))
                            clicks_order.append('fg')

                        assert largest_component[center]  # click within error

                        if verbose:
                            print(f"Class {cls}: Largest error component center is at {center}")
                    else:
                        if verbose:
                            print(
                                f"Class {cls}: No error connected components found. Prediction is perfect! No clicks were added.")

                # 更新模型输入
                input_img = np.load(join(input_temp, case),allow_pickle=True)
                if validation_gts_path is None:
                    if no_bbox:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            gts=input_img['gts'],
                            spacing=input_img['spacing'],
                            clicks=clicks_cls,
                            clicks_order=clicks_order,
                            prev_pred=segs,
                        )
                    else:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            gts=input_img['gts'],
                            spacing=input_img['spacing'],
                            clicks=clicks_cls,
                            clicks_order=clicks_order,
                            prev_pred=segs,
                            boxes=boxes,
                        )
                else:
                    if no_bbox:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            spacing=input_img['spacing'],
                            clicks=clicks_cls,
                            clicks_order=clicks_order,
                            prev_pred=segs,
                        )
                    else:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            spacing=input_img['spacing'],
                            clicks=clicks_cls,
                            clicks_order=clicks_order,
                            prev_pred=segs,
                            boxes=boxes,
                        )

            # 运行预测脚本
            # 运行预测脚本
            start_time = time.time()
            try:
                if torch.cuda.is_available():
                    # GPU 版本
                    cmd = f"CUDA_VISIBLE_DEVICES=0 bash predict.sh -i {input_temp} -o {output_temp}"
                else:
                    # CPU 版本
                    cmd = f"bash predict.sh --input {input_temp} --output {output_temp}"

                if verbose:
                    print(f"Running command: {cmd}")

                # 使用subprocess运行命令
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    print(f"Error running prediction: {stderr.decode()}")
                    if verbose:
                        print(f"Output: {stdout.decode()}")

                infer_time = time.time() - start_time
                real_running_time += infer_time
                print(f"{case} finished! Inference time: {infer_time}")
                metric[f"RunningTime_{it + 1}"].append(infer_time)

            except Exception as e:
                print(f"Error during prediction: {e}")
                infer_time = 0
                metric[f"RunningTime_{it + 1}"].append(0)

            # 加载预测结果
            if not os.path.isfile(join(output_temp, case)):
                print(f"[WARNING] Failed prediction for iteration {it}! Setting prediction to zeros...")
                segs = np.zeros_like(gts).astype(np.uint8)
            else:
                segs = np.load(join(output_temp, case),allow_pickle=True)['segs']
            all_segs.append(segs.astype(np.uint8))

            # 计算评估指标
            dsc = compute_multi_class_dsc(gts, segs)
            if dsc > 0.2:
                nsd = compute_multi_class_nsd(gts, segs, np.load(join(input_temp, case),allow_pickle=True)['spacing'])
            else:
                nsd = 0.0

            dscs.append(dsc)
            nsds.append(nsd)
            metric[f'DSC_{it + 1}'].append(dsc)
            metric[f'NSD_{it + 1}'].append(nsd)
            print('Dice', dsc, 'NSD', nsd)

            # 保存结果
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                shutil.copy(join(output_temp, case), join(save_path, case))
                segs = np.load(join(save_path, case),allow_pickle=True)['segs']
                np.savez_compressed(
                    join(save_path, case),
                    segs=segs,
                    all_segs=all_segs,
                )
            except Exception as e:
                print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                print("Final prediction could not be copied!")

        # 计算并保存最终指标
        if real_running_time > 90 * (len(np.unique(gts)) - 1):
            print(
                "[WARNING] Your model seems to take more than 90 seconds per class during inference! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
            time_warning = True
        # Compute interactive metrics
        dsc_auc = integrate.cumulative_trapezoid(np.array(dscs[-n_clicks:]), np.arange(n_clicks))[
            -1]  # AUC is only over the point prompts since the bbox prompt is optional
        nsd_auc = integrate.cumulative_trapezoid(np.array(nsds[-n_clicks:]), np.arange(n_clicks))[-1]
        dsc_final = dscs[-1]
        nsd_final = nsds[-1]
        metric['CaseName'].append(case)
        metric['TotalRunningTime'].append(real_running_time)
        metric['DSC_AUC'].append(dsc_auc)
        metric['NSD_AUC'].append(nsd_auc)
        metric['DSC_Final'].append(dsc_final)
        metric['NSD_Final'].append(nsd_final)
        os.remove(join(input_temp, case))

        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(join(save_path,  '_metrics.csv'), index=False)
    torch.cuda.empty_cache()
    shutil.rmtree(input_temp)
    shutil.rmtree(output_temp)
if __name__ == '__main__':
    main()