import os
import re
import json
import nibabel as nib
import numpy as np
from collections import defaultdict


def delete_non_pred4_files(root_dir):
    """
    Recursively delete all files ending with _predX.nii.gz (X ≠ 4),
    keeping only those ending with _pred4.nii.gz.
    """
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".nii.gz"):
                if re.search(r'_pred\d+\.nii\.gz$', filename):
                    if not filename.endswith("_pred4.nii.gz"):
                        file_path = os.path.join(root, filename)
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    else:
                        print(f"Kept: {os.path.join(root, filename)}")


class NiftiToNPZConverter:
    def __init__(self, json_path: str, input_root: str, output_dir: str):
        self.json_path = json_path
        self.input_root = input_root
        self.output_dir = output_dir
        self.dataset_mappings = self._load_mappings()
        self.sample_registry = defaultdict(lambda: {'dataset': None, 'organs': {}})

    def _load_mappings(self) -> dict:
        with open(self.json_path) as f:
            data = json.load(f)

        mappings = {}
        for dataset, organs in data.items():
            if dataset == "instance_label":
                continue
            mapping = {}
            for key, values in organs.items():
                if key.isdigit():
                    primary_name = values[0].lower().replace(' ', '_')
                    mapping[primary_name] = int(key)
            mappings[dataset] = mapping
        return mappings

    def _parse_sample_id(self, filename: str) -> str:
        base = re.sub(r'_pred\d+\.nii\.gz$', '', filename)
        return base.replace('_', '-')

    def _discover_samples(self):
        for root, dirs, files in os.walk(self.input_root):
            for file in files:
                if not file.endswith('.nii.gz'):
                    continue
                path_parts = root.split(os.sep)
                if len(path_parts) < 2:
                    continue
                organ_dir = path_parts[-2]
                dataset = path_parts[-1]
                if dataset not in self.dataset_mappings:
                    continue
                organ_map = self.dataset_mappings[dataset]
                standardized_organ = organ_dir.lower().replace(' ', '_')
                organ_id = organ_map.get(standardized_organ)
                if organ_id is None:
                    continue
                sample_id = self._parse_sample_id(file)
                self.sample_registry[sample_id]['dataset'] = dataset
                self.sample_registry[sample_id]['organs'][organ_id] = os.path.join(root, file)

    def _merge_labels(self, sample_id: str) -> dict:
        meta = self.sample_registry[sample_id]
        first_path = next(iter(meta['organs'].values()))
        ref_img = nib.load(first_path)
        spacing = ref_img.header.get_zooms()[:3]
        merged_label = np.zeros(ref_img.shape, dtype=np.uint16)
        for organ_id in sorted(meta['organs'].keys()):
            organ_data = nib.load(meta['organs'][organ_id]).get_fdata()
            merged_label = np.where(organ_data > 0, organ_id, merged_label)
        return {'label': merged_label, 'spacing': np.array(spacing, dtype=np.float32)}

    def _save_npz(self, sample_id: str, data: dict):
        output_path = os.path.join(self.output_dir, f"{sample_id}.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **data)
        print(f"Saved: {output_path}")

    def process(self):
        print("Scanning input directory...")
        self._discover_samples()
        print(f"Found {len(self.sample_registry)} samples to process")
        for i, sample_id in enumerate(self.sample_registry, 1):
            print(f"Processing [{i}/{len(self.sample_registry)}] {sample_id}")
            try:
                merged_data = self._merge_labels(sample_id)
                self._save_npz(sample_id, merged_data)
            except Exception as e:
                print(f"Failed to process {sample_id}: {str(e)}")
        print(f"All done! Output directory: {self.output_dir}")


if __name__ == "__main__":
    input_dir = "./val_stu_5"  # Set to your input root directory
    delete_non_pred4_files(input_dir)

    converter = NiftiToNPZConverter(
        json_path="./label_structure.json",  # Update with your actual JSON path
        input_root=input_dir,
        output_dir="./final/save_path"
    )
    converter.process()
