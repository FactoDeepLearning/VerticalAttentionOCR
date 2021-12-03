
#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective
from basic.transforms import SignFlipping, DPIAdjusting, Dilation, Erosion, ElasticDistortion, RandomTransform
from basic.utils import LM_str_to_ind
import os
import numpy as np
import pickle
from PIL import Image
import cv2
import copy
import torch


class DatasetManager:

    def __init__(self, params):
        self.params = params
        self.img_padding_value = params["config"]["padding_value"]
        self.dataset_class = params["dataset_class"]
        self.tokens = {
            "pad": params["config"]["padding_token"],
        }

        self.train_dataset = None
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_loader = None
        self.valid_loaders = dict()
        self.test_loaders = dict()

        self.train_sampler = None
        self.valid_samplers = dict()
        self.test_samplers = dict()

        self.charset = self.get_merged_charsets()
        self.load_datasets()

        # token configuration
        if params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = len(self.charset)
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 1
            params["config"]["padding_token"] = self.tokens["pad"]
        elif params["config"]["charset_mode"] == "attention":
            self.tokens["end"] = len(self.charset)
            self.tokens["start"] = len(self.charset) + 1
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 2
            if params["config"]["pad_label_with_end_token"]:
                params["config"]["padding_token"] = self.tokens["end"]
            else:
                params["config"]["padding_token"] = self.tokens["pad"]
        self.update_charset()
        self.my_collate_function = OCRCollateFunction(self.params["config"])

        self.load_ddp_samplers()
        self.load_dataloaders()

    def get_merged_charsets(self):
        datasets = self.params["datasets"]
        charset = set()
        for key in datasets.keys():
            with open(os.path.join(datasets[key], "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                charset = charset.union(set(info["charset"]))
        if "\n" in charset:
            charset.remove("\n")
        if "¬" in charset:
            charset.remove("¬")
        if "" in charset:
            charset.remove("")
        return sorted(list(charset))

    def update_dataset(self, dataset):
        dataset.charset = self.charset
        dataset.tokens = self.tokens
        dataset.convert_labels()

    def load_datasets(self):
        self.train_dataset = self.dataset_class(self.params, "train", self.params["train"]["name"], self.get_paths_and_sets("train", self.params["train"]["datasets"]))
        self.params["config"]["mean"], self.params["config"]["std"] = self.train_dataset.compute_std_mean()

        for custom_name in self.params["valid"].keys():
            self.valid_datasets[custom_name] = self.dataset_class(self.params, "valid", custom_name, self.get_paths_and_sets("valid", self.params["valid"][custom_name]))

    def update_charset(self):
        self.train_dataset.charset = self.charset
        self.train_dataset.tokens = self.tokens
        self.train_dataset.convert_labels()
        for key in self.valid_datasets:
            self.update_dataset(self.valid_datasets[key])

    def load_ddp_samplers(self):
        if self.params["use_ddp"]:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=True)
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = DistributedSampler(self.valid_datasets[custom_name], num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=False)
        else:
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = None

    def load_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.params["batch_size"],
                                       shuffle=True if not self.train_sampler else False,
                                       sampler=self.train_sampler,
                                       num_workers=self.params["num_gpu"], pin_memory=True, drop_last=False,
                                       collate_fn=self.my_collate_function)

        for key in self.valid_datasets.keys():
            self.valid_loaders[key] = DataLoader(self.valid_datasets[key], batch_size=self.params["batch_size"],
                                                 sampler=self.valid_samplers[key],
                                                 shuffle=False, num_workers=self.params["num_gpu"], pin_memory=True,
                                                 drop_last=False, collate_fn=self.my_collate_function)

    def generate_test_loader(self, custom_name, sets_list):
        if custom_name in self.test_loaders.keys():
            return
        paths_and_sets = list()
        for set_info in sets_list:
            paths_and_sets.append({
                "path": self.params["datasets"][set_info[0]],
                "set_name": set_info[1]
            })
        self.test_datasets[custom_name] = self.dataset_class(self.params, "test", custom_name, paths_and_sets)
        if self.dataset_class is OCRDataset:
            self.update_dataset(self.test_datasets[custom_name])
        self.test_samplers[custom_name] = DistributedSampler(self.test_datasets[custom_name],
                                                             num_replicas=self.params["num_gpu"],
                                                             rank=self.params["ddp_rank"], shuffle=False) \
            if self.params["use_ddp"] else None
        self.test_loaders[custom_name] = DataLoader(self.test_datasets[custom_name], batch_size=self.params["batch_size"],
                                                    sampler=self.test_samplers[custom_name],
                                                    shuffle=False, num_workers=self.params["num_gpu"], pin_memory=True,
                                                    drop_last=False, collate_fn=self.my_collate_function)

    def get_paths_and_sets(self, set_name, dataset_names):
        paths_and_sets = list()
        for dataset_name in dataset_names:
            path = self.params["datasets"][dataset_name]
            paths_and_sets.append({
                "path": path,
                "set_name": set_name
            })
        return paths_and_sets


class GenericDataset(Dataset):

    def __init__(self, params, set_name, custom_name, paths_and_sets, from_segmentation=False):
        self.name = custom_name
        self.set_name = set_name
        self.params = params
        self.mean = params["config"]["mean"] if "mean" in params["config"].keys() else None
        self.std = params["config"]["std"] if "std" in params["config"].keys() else None
        if from_segmentation:
            self.samples = self.load_segmented_samples(paths_and_sets)
        else:
            self.samples = self.load_samples(paths_and_sets)

        self.apply_preprocessing(params["config"]["preprocessings"])

        self.padding_value = params["config"]["padding_value"]
        if self.padding_value == "mean":
            if self.mean is None:
                _, _ = self.compute_std_mean()
            self.padding_value = self.mean
            self.params["config"]["padding_value"] = self.padding_value

        self.curriculum_config = None

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_samples(paths_and_sets):
        """
        Load images and labels
        """
        samples = list()
        for path_and_set in paths_and_sets:
            path = path_and_set["path"]
            set_name = path_and_set["set_name"]
            with open(os.path.join(path, "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                gt = info["ground_truth"][set_name]
                for filename in gt.keys():
                    name = os.path.join(os.path.basename(path), set_name, filename)
                    if type(gt[filename]) is str:
                        label = gt[filename]
                    else:
                        label = gt[filename]["text"]
                    with Image.open(os.path.join(path, set_name, filename)) as pil_img:
                        img = np.array(pil_img)
                        ## grayscale images
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=2)
                    samples.append({
                        "name": name,
                        "label": label.replace("¬", ""),
                        "img": img,
                        "unchanged_label": label,
                    })
                    if type(gt[filename]) is dict and "lines" in gt[filename].keys():
                        samples[-1]["raw_line_seg_label"] = gt[filename]["lines"]
        return samples

    def apply_preprocessing(self, preprocessings):
        for i in range(len(self.samples)):
            self.samples[i]["resize_ratio"] = [1, 1]
            for preprocessing in preprocessings:

                if preprocessing["type"] == "dpi":
                    ratio = preprocessing["target"] / preprocessing["source"]
                    temp_img = self.samples[i]["img"]
                    h, w, c = temp_img.shape
                    temp_img = cv2.resize(temp_img, (int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))
                    if len(temp_img.shape) == 2:
                        temp_img = np.expand_dims(temp_img, axis=2)
                    self.samples[i]["img"] = temp_img

                    self.samples[i]["resize_ratio"] = [ratio, ratio]

                if preprocessing["type"] == "to_grayscaled":
                    temp_img = self.samples[i]["img"]
                    h, w, c = temp_img.shape
                    if c == 3:
                        self.samples[i]["img"] = np.expand_dims(0.2125*temp_img[:, :, 0] + 0.7154*temp_img[:, :, 1] + 0.0721*temp_img[:, :, 2], axis=2).astype(np.uint8)

                if preprocessing["type"] == "to_RGB":
                    temp_img = self.samples[i]["img"]
                    h, w, c = temp_img.shape
                    if c == 1:
                        self.samples[i]["img"] = np.concatenate([temp_img, temp_img, temp_img], axis=2)

                if preprocessing["type"] == "resize":
                    pad_val, keep_ratio = preprocessing["padding_value"], preprocessing["keep_ratio"]
                    max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]
                    temp_img = self.samples[i]["img"]
                    h, w, c = temp_img.shape

                    ratio_h = max_h / h if max_h else 1
                    ratio_w = max_w / w if max_w else 1
                    if keep_ratio:
                        ratio_h = ratio_w = min(ratio_w, ratio_h)
                    new_h = min(max_h,  int(h * ratio_h))
                    new_w = min(max_w,  int(w * ratio_w))
                    temp_img = cv2.resize(temp_img, (new_w, new_h))
                    if len(temp_img.shape) == 2:
                        temp_img = np.expand_dims(temp_img, axis=2)

                    self.samples[i]["img"] = temp_img
                    self.samples[i]["resize_ratio"] = [ratio_h, ratio_w]

    def compute_std_mean(self):
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        _, _, c = self.samples[0]["img"].shape
        sum = np.zeros((c,))
        nb_pixels = 0
        for i in range(len(self.samples)):
            img = self.samples[i]["img"]
            sum += np.sum(img, axis=(0, 1))
            nb_pixels += np.prod(img.shape[:2])
        mean = sum / nb_pixels
        diff = np.zeros((c,))
        for i in range(len(self.samples)):
            img = self.samples[i]["img"]
            diff += [np.sum((img[:, :, k] - mean[k]) ** 2) for k in range(c)]
        std = np.sqrt(diff / nb_pixels)
        self.mean = mean
        self.std = std
        return mean, std

    def apply_data_augmentation(self, img):
        aug = self.params["config"]["augmentation"]
        if aug and self.set_name == "train":
            # Convert to PIL Image
            img = img[:, :, 0] if img.shape[2] == 1 else img
            img = Image.fromarray(img)
            # Apply data augmentation
            if "dpi" in aug.keys() and np.random.rand() < aug["dpi"]["proba"]:
                factor = np.random.uniform(aug["dpi"]["min_factor"], aug["dpi"]["max_factor"])
                img = DPIAdjusting(factor)(img)
            if "perspective" in aug.keys() and np.random.rand() < aug["perspective"]["proba"]:
                scale = np.random.uniform(aug["perspective"]["min_factor"], aug["perspective"]["max_factor"])
                img = RandomPerspective(distortion_scale=scale, p=1, interpolation=Image.BILINEAR, fill=255)(img)
            elif "elastic_distortion" in aug.keys() and np.random.rand() < aug["elastic_distortion"]["proba"]:
                magnitude = np.random.randint(1, aug["elastic_distortion"]["max_magnitude"] + 1)
                kernel = np.random.randint(1, aug["elastic_distortion"]["max_kernel"] + 1)
                magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
                img = ElasticDistortion(grid=(kernel, kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1, 1))(
                    img)
            elif "random_transform" in aug.keys() and np.random.rand() < aug["random_transform"]["proba"]:
                img = RandomTransform(aug["random_transform"]["max_val"])(img)
            if "dilation_erosion" in aug.keys() and np.random.rand() < aug["dilation_erosion"]["proba"]:
                kernel_h = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                             aug["dilation_erosion"]["max_kernel"] + 1)
                kernel_w = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                             aug["dilation_erosion"]["max_kernel"] + 1)
                if np.random.randint(2) == 0:
                    img = Erosion((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
                else:
                    img = Dilation((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
            if "contrast" in aug.keys() and np.random.rand() < aug["contrast"]["proba"]:
                factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
                img = adjust_contrast(img, factor)
            if "brightness" in aug.keys() and np.random.rand() < aug["brightness"]["proba"]:
                factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
                img = adjust_brightness(img, factor)
            if "sign_flipping" in aug.keys() and np.random.rand() < aug["sign_flipping"]["proba"]:
                img = SignFlipping()(img)
            # convert to numpy array
            img = np.array(img)
            img = np.expand_dims(img, axis=2) if len(img.shape) == 2 else img
        return img


class OCRDataset(GenericDataset):

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.charset = None
        self.tokens = None
        self.reduce_dims_factor = np.array([params["config"]["height_divisor"], params["config"]["width_divisor"], 1])

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        # Curriculum learning
        if self.curriculum_config:
            if self.curriculum_config["mode"] == "char":
                sample["label"] = sample["label"][:self.curriculum_config["nb_chars"]]
                sample["token_label"] = sample["token_label"][:self.curriculum_config["nb_chars"]]
                sample["label_len"] = len(sample["label"])

        # Data augmentation
        sample["img"] = self.apply_data_augmentation(sample["img"])

        # Normalization if requested
        if "normalize" in self.params["config"]["constraints"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.floor(sample["img_shape"] / self.reduce_dims_factor).astype(int)

        # Padding to handle CTC requirements
        if self.set_name == "train":
            max_label_len = 0
            height = 1
            if "CTC_line" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
            if "CTC_va" in self.params["config"]["constraints"]:
                max_label_len = max(sample["line_label_len"])
            if "CTC_pg" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                height = max(sample["img_reduced_shape"][0], 1)
            if 2 * max_label_len + 1 > sample["img_reduced_shape"][1]*height:
                sample["img"] = pad_image_width_right(sample["img"], int(np.ceil((2 * max_label_len + 1) / height) * self.reduce_dims_factor[1]), self.padding_value)
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = np.floor(sample["img_shape"] / self.reduce_dims_factor).astype(int)
            sample["img_reduced_shape"] = [max(1, t) for t in sample["img_reduced_shape"]]

        # Padding constraints to handle model needs
        if "padding" in self.params["config"]["constraints"]:
            if sample["img_shape"][0] < self.params["config"]["padding"]["min_height"]:
                sample["img"] = pad_image_height_bottom(sample["img"], self.params["config"]["padding"]["min_height"], self.padding_value)
            if sample["img_shape"][1] < self.params["config"]["padding"]["min_width"]:
                sample["img"] = pad_image_width_right(sample["img"], self.params["config"]["padding"]["min_width"], self.padding_value)

        return sample

    def get_charset(self):
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Label str to token at character level
        """
        for i in range(len(self.samples)):
            label = self.samples[i]["label"]
            line_labels = label.split("\n")
            full_label = label.replace("\n", " ").replace("  ", " ")

            self.samples[i]["label"] = full_label
            self.samples[i]["token_label"] = LM_str_to_ind(self.charset, full_label)
            if "add_eot" in self.params["config"]["constraints"]:
                self.samples[i]["token_label"].append(self.tokens["end"])
            self.samples[i]["label_len"] = len(self.samples[i]["token_label"])

            self.samples[i]["line_label"] = line_labels
            self.samples[i]["token_line_label"] = [LM_str_to_ind(self.charset, l) for l in line_labels]
            self.samples[i]["line_label_len"] = [len(l) for l in line_labels]
            self.samples[i]["nb_lines"] = len(line_labels)


class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for OCR task
    """

    def __init__(self, config):
        self.img_padding_value = float(config["padding_value"])
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        ids = [int(batch_data[i]["name"].split("/")[-1].split("_")[-1].split(".")[0]) for i in range(len(batch_data))]

        labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
        labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
        labels = torch.tensor(labels).long()
        labels_len = [batch_data[i]["label_len"] for i in range(len(batch_data))]

        raw_labels = [batch_data[i]["label"] for i in range(len(batch_data))]
        unchanged_labels = [batch_data[i]["unchanged_label"] for i in range(len(batch_data))]
        nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]

        line_raw= [batch_data[i]["line_label"] for i in range(len(batch_data))]
        line_token = [batch_data[i]["token_line_label"] for i in range(len(batch_data))]
        pad_line_token = list()
        line_len = [batch_data[i]["line_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_lines)):
            current_lines = [line_token[j][i] if i < nb_lines[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_line_token.append(torch.tensor(pad_sequences_1D(current_lines, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_lines[j]:
                    line_len[j].append(0)
        line_len = [i for i in zip(*line_len)]

        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs = pad_images(imgs, padding_value=self.img_padding_value)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)
        formatted_batch_data = {
            "names": names,
            "ids": ids,
            "nb_lines": nb_lines,
            "labels": labels,
            "raw_labels": raw_labels,
            "unchanged_labels": unchanged_labels,
            "labels_len": labels_len,
            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,
            "line_raw": line_raw,
            "line_labels": pad_line_token,
            "line_labels_len": line_len
        }

        return formatted_batch_data


def pad_sequences_1D(data, padding_value):
    """
    Pad data with padding_value to get same length
    """
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)).astype(np.int32) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data


def pad_images(data, padding_value):
    """
    data: list of numpy array
    """
    x_lengths = [x.shape[0] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)
    padded_data = np.ones((len(data), longest_x, longest_y, data[0].shape[2])) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        padded_data[i, :x_len, :y_len, ...] = data[i][:x_len, :y_len, ...]
    return padded_data


def pad_image_width_right(img, new_width, padding_value):
    """
    Pad img to right side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_right = np.ones((h, pad_width, c)) * padding_value
    img = np.concatenate([img, pad_right], axis=1)
    return img


def pad_image_height_bottom(img, new_height, padding_value):
    """
    Pad img to bottom side with padding value to reach new_height as height
    """
    h, w, c = img.shape
    pad_height = max((new_height - h), 0)
    pad_bottom = np.ones((pad_height, w, c)) * padding_value
    img = np.concatenate([img, pad_bottom], axis=0)
    return img
