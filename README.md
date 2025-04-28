# FastSAM3D-CVPR 2025 3D Medical Image Segmentation Challenge

FastSAM3D is specifically developed for the CVPR 2025 3D Medical Image Segmentation Challenge. It is designed to provide fast, efficient, and accurate 3D segmentation, leveraging lightweight architectures and optimized inference strategies tailored for the challenge’s unique dataset and requirements.

---

##  System Requirements

- **Python** ≥ 3.9  
- **CUDA** = 12.1  
- **GPU Requirements**: GPUs supporting FLASH Attention, such as A100, RTX 3090/4090, H100 (Ampere, Ada, Hopper architectures)

---

##  Installation

```
pip install -r requirements.txt
```
---

## Steps

### 1 Prepare Your Training Data

Organize all NPZ files containing gts.npy from the validation set into the 'labels' folder. The  `classify.py` script can convert NPZ files to NII.GZ format and categorize them based on  `CVPR25.json`.

>    ```console
>    python classify.py
>    ```

Categorize all .nii.gz files into subfolders based on the top-level keys in the corresponding JSON files, creating the subfolders as needed.

>    ```console
>    python categorize.py
>    ```

Convert the imgs.npy array from image.npz to NIfTI format image.nii.gz, then move it to respective organ-specific subfolders' imagesTr/ directories and standardize filenames by removing _image/_label suffixes.

>    ```console
>    python reallocate.py
>    ```

The target file structures should be like the following:

> ```
> data/medical_preprocessed
>       ├── adrenal
>       │ ├── ct_WORD
>       │ │ ├── imagesTr
>       │ │ │ ├── word_0025.nii.gz
>       │ │ │ ├── ...
>       │ │ ├── labelsTr
>       │ │ │ ├── word_0025.nii.gz
>       │ │ │ ├── ...
>       ├── ...
> ```

---

### 2 Modify `utils/data_paths.py` based on your own data. 
> ```
> img_datas = [
> "data/train/adrenal/ct_WORD",
> "data/train/liver/ct_WORD",
> ...
> ]
> ```

---

### 3 Train the Teacher Model and Prepare Labels(logits)

To train the teacher model and prepare labels for guided distillation to the student model, run the command below. Ensure your data and checkpoint are placed in the designated locations within the shell script.

>    ```console
>    python preparelabel.py
>    ```

---

### 4 Distill the Model

To perform distillation, run the command below. The distilled checkpoint will be saved in  `work_dir`. Ensure your data and checkpoint paths are correctly specified in the shell script.

>    ```console
>    python distillation.py
>    ```

---

### 5 Validate the Teacher Model

To validate the teacher model, run the command below. Ensure your data and the teacher model checkpoint (linked below) are correctly placed in the shell script.

>    ```console
>    python validation.py
>    ```

---

### 6 Validate FastSAM3D model, or your distilled student model

To validate the distilled student model, run the command below. Ensure your data, teacher model, and FastSAM3D checkpoint (linked below) are properly placed in the shell script.

>    ```console
>    python validation_student.py
>    ```

---
Below are the links to the docker_file and results for FastSAM3D-CVPR2025 :


| Model                | Download Link |
|----------------------|---------------|
| Docker_file          | [Download](https://drive.google.com/file/d/1RNqeVXSjUFmTo__qahKY-OMJGubu0fE3/view?usp=drive_link ) |
| Results              | [Download](https://drive.google.com/file/d/1rFgAb-lEt9fQ5ikXsULY4XeqQi6piJwL/view?usp=drive_link) |

---


```

## Citation

```
@misc{shen2024fastsam3d,
      title={FastSAM3D: An Efficient Segment Anything Model for 3D Volumetric Medical Images}, 
      author={Yiqing Shen and Jingxing Li and Xinyuan Shao and Blanca Inigo Romillo and Ankush Jindal and David Dreizin and Mathias Unberath},
      year={2024},
      eprint={2403.09827},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

