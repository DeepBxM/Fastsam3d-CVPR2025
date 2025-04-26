import os
import json
import shutil
import numpy as np
import nibabel as nib


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_labels_from_nii(nii_path):
    """ Read nii.gz file and extract unique label values (excluding background 0) """
    img = nib.load(nii_path)
    data = img.get_fdata()
    unique_labels = np.unique(data).astype(int)  # Get unique label values
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background 0
    return unique_labels


def get_organ_name(label, json_data, dataset_name):
    """ Get the organ name corresponding to the label from JSON """
    dataset_info = json_data.get(dataset_name, {})
    return dataset_info.get(str(label), [None])[0]  # Get the first organ name


def organize_dataset(dataset_folder, json_path, output_folder):
    """
    Iterate over all subdirectories in dataset_folder, extract labels from label.nii.gz files, and reorganize data structure.
    dataset_folder: Root directory of the original dataset (including CT_AbdomenAtlas, CT_AMOS, MR_AMOS, etc.)
    json_path: Path to the JSON file
    output_folder: Destination output folder
    """
    json_data = load_json(json_path)

    # Iterate over all subdirectories in dataset_folder (e.g., CT_AbdomenAtlas, CT_AMOS, MR_AMOS)
    for dataset_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        # Iterate over nii.gz files in the subfolder
        for file_name in os.listdir(dataset_path):
            if not file_name.endswith('.nii.gz'):
                continue

            nii_path = os.path.join(dataset_path, file_name)
            unique_labels = extract_labels_from_nii(nii_path)

            for label in unique_labels:
                organ_name = get_organ_name(label, json_data, dataset_name)
                if organ_name is None:
                    continue

                organ_name = organ_name.replace(" ", "_")  # Replace spaces with underscores
                organ_folder = os.path.join(output_folder, "val_1", organ_name, dataset_name, "labelsTr")

                # Create the directory and copy the file
                os.makedirs(organ_folder, exist_ok=True)
                shutil.copy(nii_path, os.path.join(organ_folder, file_name))
                print(f"Moved {file_name} to {organ_folder}")


# Run the code
dataset_folder = "./validation"  # Your dataset folder, including CT_AbdomenAtlas, CT_AMOS, MR_AMOS
json_path = "./CVPR25.json"  # Path to the JSON file
output_folder = "./val_new"  # Destination output folder

organize_dataset(dataset_folder, json_path, output_folder)
