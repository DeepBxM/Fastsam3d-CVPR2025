import os
import numpy as np
import nibabel as nib
import json
import shutil


def save_to_nifti(data, save_path, is_label=False):
    """
    Save a numpy array to a .nii.gz file.
    
    Parameters:
    - data: numpy array, image or label data
    - save_path: output path
    - is_label: whether the data is label (affects data type)
    """
    if is_label:
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32) if data.dtype == np.float64 else data

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)

    if is_label:
        nifti_img.header.set_data_dtype(np.uint8)

    nib.save(nifti_img, save_path)
    print(f"✅ Saved: {save_path}")


def convert_folder_npz_to_nifti(input_dir, output_dir):
    """
    Convert all .npz files in a folder (with 'gts' arrays) to .nii.gz files.
    
    Parameters:
    - input_dir: directory containing .npz files
    - output_dir: directory to save .nii.gz files
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.npz'):
            npz_path = os.path.join(input_dir, filename)
            print(f"🔄 Processing: {npz_path}")

            data = np.load(npz_path)
            if 'gts' not in data:
                print(f"⚠️ Warning: 'gts' not found in {filename}, skipping...")
                continue

            gts = data['gts']
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_label.nii.gz"
            output_path = os.path.join(output_dir, output_filename)

            save_to_nifti(gts, output_path, is_label=True)

    print("✅ All files converted!")


def organize_nii_files(json_path, labels_dir, output_dir):
    """
    Organize .nii.gz label files into subdirectories based on JSON keys.
    
    Parameters:
    - json_path: path to JSON file containing top-level keys
    - labels_dir: directory containing .nii.gz files
    - output_dir: root directory to organize output
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON file {json_path} does not exist!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    top_keys = set(data.keys())
    print(f"📋 Top-level keys in JSON: {top_keys}")

    if not os.path.exists(labels_dir):
        print(f"❌ Error: Labels directory {labels_dir} does not exist!")
        return

    nii_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
    print(f"📂 Found {len(nii_files)} .nii.gz files: {nii_files}")

    moved_files = 0
    unmatched_files = []

    for nii_file in nii_files:
        matched = False
        for key in top_keys:
            if key in nii_file:
                target_folder = os.path.join(output_dir, key)
                os.makedirs(target_folder, exist_ok=True)

                src_path = os.path.join(labels_dir, nii_file)
                dst_path = os.path.join(target_folder, nii_file)

                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    moved_files += 1
                    print(f"Moved {nii_file} -> {target_folder}")
                else:
                    print(f"Warning: File {nii_file} does not exist in labels directory!")

                matched = True
                break

        if not matched:
            unmatched_files.append(nii_file)

    print(f"\nTask completed. {moved_files} files moved.")
    print(f"Unmatched files (still in labels directory): {unmatched_files}")


# ==== User Settings (modify as needed) ====
input_directory = './input_npz'               # Directory with input .npz files
intermediate_label_dir = './labelsTr'           # Intermediate output for converted labels
json_path = './CVPR25.json'            # Path to JSON file with top-level keys
output_directory = './validation'             # Final organized output directory

# ==== Execution ====
convert_folder_npz_to_nifti(input_directory, intermediate_label_dir)
organize_nii_files(json_path, intermediate_label_dir, output_directory)
