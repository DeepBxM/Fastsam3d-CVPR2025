import torch
from tqdm import tqdm
import os
join = os.path.join
from glob import glob
from segment_anything.build_sam3D import sam_model_registry3D
from torch.utils.data import DataLoader
import torchio as tio
from utils.data_loader import Dataset_Union_ALL_Val
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Store labels for augmented data.')
parser.add_argument('--data_train_path', type=str, default='./data_10per/train', help='Path to the teacher encoder to generate logits')
parser.add_argument('--label_path_base', type=str, default='./data/augumentation/label', help='Base path for saving labels')
parser.add_argument('--train_path_base', type=str, default='./data/augumentation/images', help='Base path for saving training images')
parser.add_argument('--checkpoint_path', type=str, default='./ckpt/sam_med3d.pth', help='Path to the model checkpoint')
parser.add_argument('--crop_size', nargs=3, type=int, default=[128, 128, 128], help='Crop size for the images')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size for data loading')
parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
args = parser.parse_args()

def save_model_weights(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def store_label(model, data: DataLoader, args):
    i = 0
    for batch_data in tqdm(data):


        image3D, _, _ = batch_data
        print(f"image name: {_}")
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        image3D = norm_transform(image3D.squeeze(dim=1))
        image3D = image3D.unsqueeze(dim=1)
        train_path = f"{args.train_path_base}/images{i}.pt"
        label_path = f"{args.label_path_base}/label{i}.pt"
        i += 1
        image3D = image3D.float().to(args.device)
        output = model(image3D)
        for j in range(len(output)):
            output[j] = output[j].cpu().squeeze(dim=0)
        image3D = image3D.cpu().squeeze(dim=0)
        torch.save(image3D, train_path)
        torch.save(output, label_path)
        print(f"Batch {i}: Saved image to {train_path} and label to {label_path}")

if __name__ == "__main__":    
    np.random.seed(2023)
    torch.manual_seed(2023)
    all_dataset_paths =  [
"data_10per/train/oral_cavity_delineation/CT_HaN-Seg",
"data_10per/train/intra-meatal_region_of_vestibular_schwannoma/MR_T1c_crossMoDA_Tumor_Cochlea",
"data_10per/train/left_gluteus_maximus/MR_TotalSeg",
"data_10per/train/left_gluteus_maximus/CT_TotalSeg_muscles",
"data_10per/train/left_iliac_vena/MR_TotalSeg",
"data_10per/train/left_iliac_vena/CT_TotalSeg_cardiac",
"data_10per/train/prostate_lesion/MR_QIN-PROSTATE-Lesion",
"data_10per/train/right_adrenal_gland/MR_TotalSeg",
"data_10per/train/right_adrenal_gland/CT_AMOS",
"data_10per/train/right_adrenal_gland/MR_AMOS",
"data_10per/train/right_adrenal_gland/CT_TotalSeg_organs",
"data_10per/train/right_adrenal_gland/CT_AbdomenAtlas",
"data_10per/train/right_iliac_artery/MR_TotalSeg",
"data_10per/train/right_iliac_artery/CT_TotalSeg_cardiac",
"data_10per/train/right_ventricle_cavity/MR_Heart_ACDC",
"data_10per/train/spleen/MR_CHAOS-T1",
"data_10per/train/spleen/MR_TotalSeg",
"data_10per/train/spleen/MR_CHAOS-T2",
"data_10per/train/spleen/CT_AMOS",
"data_10per/train/spleen/MR_AMOS",
"data_10per/train/spleen/CT_TotalSeg_organs",
"data_10per/train/spleen/CT_AbdomenAtlas",
"data_10per/train/spleen/CT_AbdomenCT1K",
"data_10per/train/arytenoids_delineation/CT_HaN-Seg",
"data_10per/train/lung_lower_lobe_right/CT_TotalSeg_organs",
"data_10per/train/cervical_cancer_tumor/MR_CervicalCancer",
"data_10per/train/stroke_lesion/MR_ISLES_DWI",
"data_10per/train/stroke_lesion/MR_ISLES_ADC",
"data_10per/train/lips/CT_HaN-Seg",
"data_10per/train/left_iliopsoas/MR_TotalSeg",
"data_10per/train/left_iliopsoas/CT_TotalSeg_muscles",
"data_10per/train/right_femur/MR_TotalSeg",
"data_10per/train/right_femur/CT_TotalSeg_muscles",
"data_10per/train/right_clavicula/MR_TotalSeg",
"data_10per/train/right_clavicula/CT_TotalSeg_muscles",
"data_10per/train/inferior_vena_cava/MR_TotalSeg",
"data_10per/train/inferior_vena_cava/CT_TotalSeg_cardiac",
"data_10per/train/inferior_vena_cava/CT_AMOS",
"data_10per/train/inferior_vena_cava/MR_AMOS",
"data_10per/train/right_kidney_cyst/CT_TotalSeg_organs",
"data_10per/train/esophagus/MR_TotalSeg",
"data_10per/train/esophagus/CT_AMOS",
"data_10per/train/esophagus/MR_AMOS",
"data_10per/train/esophagus/CT_TotalSeg_organs",
"data_10per/train/esophagus/CT_ThoracicOrgans-TCIA-LCTSC",
"data_10per/train/esophagus/CT_AbdomenAtlas",
"data_10per/train/liver/MR_CHAOS-T1",
"data_10per/train/liver/MR_TotalSeg",
"data_10per/train/liver/MR_CHAOS-T2",
"data_10per/train/liver/CT_AMOS",
"data_10per/train/liver/MR_AMOS",
"data_10per/train/liver/CT_TotalSeg_organs",
"data_10per/train/liver/CT_AbdomenAtlas",
"data_10per/train/liver/CT_AbdomenCT1K",
"data_10per/train/bladder/CT_AMOS",
"data_10per/train/bladder/MR_AMOS",
"data_10per/train/bladder/CT_AbdomenAtlas",
"data_10per/train/pancreas_tumors/CT_PancreasTumor",
"data_10per/train/left_iliac_artery/MR_TotalSeg",
"data_10per/train/left_iliac_artery/CT_TotalSeg_cardiac",
"data_10per/train/right_hip/MR_TotalSeg",
"data_10per/train/right_hip/CT_TotalSeg_muscles",
"data_10per/train/left_kidney/MR_CHAOS-T1",
"data_10per/train/left_kidney/MR_TotalSeg",
"data_10per/train/left_kidney/MR_CHAOS-T2",
"data_10per/train/left_kidney/CT_AMOS",
"data_10per/train/left_kidney/CT_TotalSeg_organs",
"data_10per/train/left_kidney/CT_AbdomenAtlas",
"data_10per/train/left_kidney/CT_AbdomenCT1K",
"data_10per/train/vertebrae_t10/CT_TotalSeg-vertebrae",
"data_10per/train/left_gluteus_minimus/MR_TotalSeg",
"data_10per/train/left_gluteus_minimus/CT_TotalSeg_muscles",
"data_10per/train/left_brachiocephalic_vein/CT_TotalSeg_cardiac",
"data_10per/train/duodenum/MR_TotalSeg",
"data_10per/train/duodenum/CT_AMOS",
"data_10per/train/duodenum/MR_AMOS",
"data_10per/train/duodenum/CT_TotalSeg_organs",
"data_10per/train/right_gluteus_maximus/MR_TotalSeg",
"data_10per/train/right_gluteus_maximus/CT_TotalSeg_muscles",
"data_10per/train/trachea/CT_TotalSeg_organs",
"data_10per/train/left_clavicula/MR_TotalSeg",
"data_10per/train/left_clavicula/CT_TotalSeg_muscles",
"data_10per/train/right_brachiocephalic_vein/CT_TotalSeg_cardiac",
"data_10per/train/left_gluteus_medius/MR_TotalSeg",
"data_10per/train/left_gluteus_medius/CT_TotalSeg_muscles",
"data_10per/train/spinal_cord/MR_TotalSeg",
"data_10per/train/spinal_cord/CT_TotalSeg_muscles",
"data_10per/train/spinal_cord/CT_ThoracicOrgans-TCIA-LCTSC",
"data_10per/train/spinal_cord/CT_HaN-Seg",
"data_10per/train/gallbladder/MR_TotalSeg",
"data_10per/train/gallbladder/CT_AMOS",
"data_10per/train/gallbladder/MR_AMOS",
"data_10per/train/gallbladder/CT_TotalSeg_organs",
"data_10per/train/gallbladder/CT_AbdomenAtlas",
"data_10per/train/pulmonary_vein/CT_TotalSeg_cardiac",
"data_10per/train/lung_middle_lobe_right/CT_TotalSeg_organs",
"data_10per/train/vertebrae_l4/CT_TotalSeg-vertebrae",
"data_10per/train/left_hip/MR_TotalSeg",
"data_10per/train/left_hip/CT_TotalSeg_muscles",
"data_10per/train/right_submandibular_gland/CT_HaN-Seg",
"data_10per/train/right_optic_nerve/CT_HaN-Seg",
"data_10per/train/right_lung/MR_TotalSeg",
"data_10per/train/right_lung/CT_Lungs",
"data_10per/train/right_lung/CT_ThoracicOrgans-TCIA-LCTSC",
"data_10per/train/vertebrae_t7/CT_TotalSeg-vertebrae",
"data_10per/train/right_cochlea/CT_HaN-Seg",
"data_10per/train/right_cochlea/MR_T1c_crossMoDA_Tumor_Cochlea",
"data_10per/train/intervertebral_discs/MR_TotalSeg",
"data_10per/train/left_adrenal_gland/MR_TotalSeg",
"data_10per/train/left_adrenal_gland/CT_AMOS",
"data_10per/train/left_adrenal_gland/MR_AMOS",
"data_10per/train/left_adrenal_gland/CT_TotalSeg_organs",
"data_10per/train/left_adrenal_gland/CT_AbdomenAtlas",
"data_10per/train/right_gluteus_minimus/MR_TotalSeg",
"data_10per/train/right_gluteus_minimus/CT_TotalSeg_muscles",
"data_10per/train/postcava/CT_AbdomenAtlas",
"data_10per/train/right_subclavian_artery/CT_TotalSeg_cardiac",
"data_10per/train/vertebrae_c1/CT_TotalSeg-vertebrae",
"data_10per/train/larynx-supraglottic/CT_HaN-Seg",
"data_10per/train/white_matter_hyperintensities/MR_WMH_T1",
"data_10per/train/white_matter_hyperintensities/MR_WMH_FLAIR",
"data_10per/train/vertebrae_l5/CT_TotalSeg-vertebrae",
"data_10per/train/thyroid/CT_HaN-Seg",
"data_10per/train/heart/MR_TotalSeg",
"data_10per/train/heart/CT_TotalSeg_cardiac",
"data_10per/train/heart/CT_ThoracicOrgans-TCIA-LCTSC",
"data_10per/train/liver_tumors/CT_LiverTumor",
"data_10per/train/right_posterior_segment_of_the_eyeball/CT_HaN-Seg",
"data_10per/train/vertebrae_c3/CT_TotalSeg-vertebrae",
"data_10per/train/mandible/CT_HaN-Seg",
"data_10per/train/right_autochthon/MR_TotalSeg",
"data_10per/train/right_autochthon/CT_TotalSeg_muscles",
"data_10per/train/prostate/MR_TotalSeg",
"data_10per/train/prostate/CT_AMOS",
"data_10per/train/prostate/MR_ProstateT2",
"data_10per/train/prostate/CT_TotalSeg_organs",
"data_10per/train/superior_vena_cava/CT_TotalSeg_cardiac",
"data_10per/train/left_anterior_segment_of_the_eyeball/CT_HaN-Seg",
"data_10per/train/left_lung/MR_TotalSeg",
"data_10per/train/left_lung/CT_Lungs",
"data_10per/train/left_lung/CT_ThoracicOrgans-TCIA-LCTSC",
"data_10per/train/buccal_mucosa/CT_HaN-Seg",
"data_10per/train/colon/MR_TotalSeg",
"data_10per/train/colon/CT_TotalSeg_organs",
"data_10per/train/right_gluteus_medius/MR_TotalSeg",
"data_10per/train/right_gluteus_medius/CT_TotalSeg_muscles",
"data_10per/train/surrounding_non-enhancing_flair_hyperintensit/MR_BraTS-T1n",
"data_10per/train/surrounding_non-enhancing_flair_hyperintensit/MR_BraTS-T1c",
"data_10per/train/surrounding_non-enhancing_flair_hyperintensit/MR_BraTS-T2f",
"data_10per/train/colon_cancer_primaries/CT_ColonTumor",
"data_10per/train/right_carotid_artery/CT_HaN-Seg",
"data_10per/train/vertebrae_t1/CT_TotalSeg-vertebrae",
"data_10per/train/left_optic_nerve/CT_HaN-Seg",
"data_10per/train/lymph_node/CT_LymphNode",
"data_10per/train/covid-19_infection/CT_COVID19-Infection",
"data_10per/train/small_bowel/MR_TotalSeg",
"data_10per/train/small_bowel/CT_TotalSeg_organs",
"data_10per/train/urinary_bladder/MR_TotalSeg",
"data_10per/train/urinary_bladder/CT_TotalSeg_organs",
"data_10per/train/vertebrae_c4/CT_TotalSeg-vertebrae",
"data_10per/train/lung_lesions/CT_LungLesion",
"data_10per/train/vertebrae_l1/CT_TotalSeg-vertebrae",
"data_10per/train/sacrum/MR_TotalSeg",
"data_10per/train/sacrum/CT_TotalSeg-vertebrae",
"data_10per/train/enhancing_tissue/MR_BraTS-T1n",
"data_10per/train/enhancing_tissue/MR_BraTS-T1c",
"data_10per/train/enhancing_tissue/MR_BraTS-T2f",
"data_10per/train/enhancing_tissue/MR_BraTS-T2w",
"data_10per/train/vertebrae_t12/CT_TotalSeg-vertebrae",
"data_10per/train/left_femur/MR_TotalSeg",
"data_10per/train/left_femur/CT_TotalSeg_muscles",
"data_10per/train/stomach/MR_TotalSeg",
"data_10per/train/stomach/CT_AMOS",
"data_10per/train/stomach/MR_AMOS",
"data_10per/train/stomach/CT_TotalSeg_organs",
"data_10per/train/stomach/CT_AbdomenAtlas",
"data_10per/train/right_scapula/MR_TotalSeg",
"data_10per/train/right_scapula/CT_TotalSeg_muscles",
"data_10per/train/lung_upper_lobe_left/CT_TotalSeg_organs",
"data_10per/train/thyroid_gland/CT_TotalSeg_organs",
"data_10per/train/aorta/MR_TotalSeg",
"data_10per/train/aorta/CT_TotalSeg_cardiac",
"data_10per/train/aorta/CT_AMOS",
"data_10per/train/aorta/MR_AMOS",
"data_10per/train/aorta/CT_AbdomenAtlas",
"data_10per/train/right_parotid_gland/CT_HaN-Seg",
"data_10per/train/vertebrae_l3/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_t9/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_c5/CT_TotalSeg-vertebrae",
"data_10per/train/pancreas/MR_TotalSeg",
"data_10per/train/pancreas/CT_AMOS",
"data_10per/train/pancreas/MR_AMOS",
"data_10per/train/pancreas/CT_TotalSeg_organs",
"data_10per/train/pancreas/CT_AbdomenAtlas",
"data_10per/train/pancreas/CT_AbdomenCT1K",
"data_10per/train/myocardium/MR_Heart_ACDC",
"data_10per/train/pituitary_gland/CT_HaN-Seg",
"data_10per/train/transition_zone/MR_ProstateADC",
"data_10per/train/vertebrae_t11/CT_TotalSeg-vertebrae",
"data_10per/train/left_carotid_artery/CT_HaN-Seg",
"data_10per/train/larynx-glottis/CT_HaN-Seg",
"data_10per/train/left_ventricle_cavity/MR_Heart_ACDC",
"data_10per/train/vertebrae/MR_TotalSeg",
"data_10per/train/vertebrae_c6/CT_TotalSeg-vertebrae",
"data_10per/train/optic_chiasm/CT_HaN-Seg",
"data_10per/train/left_common_carotid_artery/CT_TotalSeg_cardiac",
"data_10per/train/vertebrae_l2/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_t5/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_s1/CT_TotalSeg-vertebrae",
"data_10per/train/left_renal_structure_identified_via_abdominal_magnetic_resonance/MR_AMOS",
"data_10per/train/vertebrae_t3/CT_TotalSeg-vertebrae",
"data_10per/train/cervical_esophagus/CT_HaN-Seg",
"data_10per/train/left_kidney_cyst/CT_TotalSeg_organs",
"data_10per/train/left_autochthon/MR_TotalSeg",
"data_10per/train/left_autochthon/CT_TotalSeg_muscles",
"data_10per/train/left_atrial_appendage/CT_TotalSeg_cardiac",
"data_10per/train/right_kidney/MR_CHAOS-T1",
"data_10per/train/right_kidney/MR_TotalSeg",
"data_10per/train/right_kidney/MR_CHAOS-T2",
"data_10per/train/right_kidney/CT_AMOS",
"data_10per/train/right_kidney/MR_AMOS",
"data_10per/train/right_kidney/CT_TotalSeg_organs",
"data_10per/train/right_kidney/CT_AbdomenAtlas",
"data_10per/train/right_kidney/CT_AbdomenCT1K",
"data_10per/train/cricopharyngeal_inlet/CT_HaN-Seg",
"data_10per/train/vertebrae_t8/CT_TotalSeg-vertebrae",
"data_10per/train/skull/CT_TotalSeg_muscles",
"data_10per/train/right_iliopsoas/MR_TotalSeg",
"data_10per/train/right_iliopsoas/CT_TotalSeg_muscles",
"data_10per/train/brainstem/CT_HaN-Seg",
"data_10per/train/vertebrae_c2/CT_TotalSeg-vertebrae",
"data_10per/train/right_common_carotid_artery/CT_TotalSeg_cardiac",
"data_10per/train/lung_upper_lobe_right/CT_TotalSeg_organs",
"data_10per/train/lesion/CT_WholeBodyTumor",
"data_10per/train/resection_cavity/MR_BraTS-T1n",
"data_10per/train/resection_cavity/MR_BraTS-T1c",
"data_10per/train/resection_cavity/MR_BraTS-T2f",
"data_10per/train/resection_cavity/MR_BraTS-T2w",
"data_10per/train/left_atrium/MR_LeftAtrium",
"data_10per/train/vertebrae_t6/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_c7/CT_TotalSeg-vertebrae",
"data_10per/train/right_humerus/MR_TotalSeg",
"data_10per/train/right_humerus/CT_TotalSeg_muscles",
"data_10per/train/right_iliac_vena/MR_TotalSeg",
"data_10per/train/right_iliac_vena/CT_TotalSeg_cardiac",
"data_10per/train/right_anterior_segment_of_the_eyeball/CT_HaN-Seg",
"data_10per/train/left_cochlea/CT_HaN-Seg",
"data_10per/train/left_cochlea/MR_T1c_crossMoDA_Tumor_Cochlea",
"data_10per/train/left_parotid_gland/CT_HaN-Seg",
"data_10per/train/vertebrae_t2/CT_TotalSeg-vertebrae",
"data_10per/train/vertebrae_t4/CT_TotalSeg-vertebrae",
"data_10per/train/surrounding_non-enhancing_flair_hyperintensity/MR_BraTS-T2w",
"data_10per/train/left_scapula/MR_TotalSeg",
"data_10per/train/left_scapula/CT_TotalSeg_muscles",
"data_10per/train/head-neck_cancer/CT_SegRap_HeadNeckTumor",
"data_10per/train/left_lacrimal_gland/CT_HaN-Seg",
"data_10per/train/lung_lower_lobe_left/CT_TotalSeg_organs",
"data_10per/train/left_posterior_segment_of_the_eyeball/CT_HaN-Seg",
"data_10per/train/brachiocephalic_trunk/CT_TotalSeg_cardiac",
"data_10per/train/left_submandibular_gland/CT_HaN-Seg",
"data_10per/train/left_humerus/MR_TotalSeg",
"data_10per/train/left_humerus/CT_TotalSeg_muscles",
"data_10per/train/portal_vein_and_splenic_vein/MR_TotalSeg",
"data_10per/train/portal_vein_and_splenic_vein/CT_TotalSeg_cardiac",
"data_10per/train/extra-meatal_region_of_vestibular_schwannoma/MR_T1c_crossMoDA_Tumor_Cochlea",
"data_10per/train/left_subclavian_artery/CT_TotalSeg_cardiac",
"data_10per/train/non-enhancing_tumor_core/MR_BraTS-T1n",
"data_10per/train/non-enhancing_tumor_core/MR_BraTS-T1c",
"data_10per/train/non-enhancing_tumor_core/MR_BraTS-T2f",
"data_10per/train/non-enhancing_tumor_core/MR_BraTS-T2w",
"data_10per/train/brain/MR_TotalSeg",
"data_10per/train/brain/CT_TotalSeg_muscles"
]

    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(128,128,128)),
    ]
    
    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type='Ts', 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=1,
        split_idx=0,
        pcc=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    device = args.device
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    model_dict = torch.load(args.checkpoint_path, map_location=device)
    state_dict = model_dict['model_state_dict']
    sam_model_tune.load_state_dict(state_dict)

    store_label(sam_model_tune.image_encoder, test_dataloader, args)
