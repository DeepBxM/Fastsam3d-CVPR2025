import torch
import torchio as tio
import SimpleITK as sitk
sitk_image = sitk.Image((224, 224), sitk.sitkVectorFloat32, 3)
i2 = tio.ScalarImage.from_sitk(sitk_image)

data = '/media/wagnchogn/ssd_2t/lmy/FastSAM3D_copy/data_10per/val_with_organ (copy)/Adrenocortical_carcinoma/CT_AbdTumor_Adrenal/imagesTr/CT_AbdTumor_Adrenal_Ki67_Seg_004.nii.gz'
sitk_image = sitk.ReadImage(data)
sitk_image = sitk.GetArrayFromImage(sitk_image)
subject = tio.Subject(
    image=tio.ScalarImage.from_sitk(sitk_image),
)
