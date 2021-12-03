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

import sys
sys.path.insert(0, '../../..')
from torch.optim import Adam
from OCR.document_OCR.v_attention.trainer_pg_va import Manager
from OCR.document_OCR.v_attention.models_pg_va import VerticalAttention, LineDecoderCTC
from basic.models import FCN_Encoder
from basic.generic_dataset_manager import OCRDataset
import torch
import torch.multiprocessing as mp


def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = Manager(params)
    # Model trains until max_time_training or max_nb_epochs is reached
    model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()


    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "diff_len", "time", "worst_cer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


if __name__ == "__main__":

    dataset_name = "IAM"  # ["RIMES", "IAM", "READ_2016"]

    params = {
        "dataset_params": {
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_paragraph".format(dataset_name),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [dataset_name, ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [dataset_name, ],
            },
            "dataset_class": OCRDataset,
            "config": {
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": None,  # Label padding value (None: default value is chosen)
                "charset_mode": "CTC",  # add blank label
                "constraints": ["padding", "CTC_va"],  # Padding for models constraints and CTC requirements
                "padding": {
                    "min_height": 480,  # to handle model requirements (AdaptivePooling)
                    "min_width": 800,  # to handle model requirements (AdaptivePooling)
                },
                "preprocessings": [
                    {
                        "type": "dpi",  # modify image resolution
                        "source": 300,  # from 300 dpi
                        "target": 150,  # to 150 dpi
                    },
                    {
                        "type": "to_RGB",
                        # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                # Augmentation techniques to use at training time
                "augmentation": {
                    "dpi": {
                        "proba": 0.2,
                        "min_factor": 0.75,
                        "max_factor": 1,
                    },
                    "perspective": {
                        "proba": 0.2,
                        "min_factor": 0,
                        "max_factor": 0.3,
                    },
                    "elastic_distortion": {
                        "proba": 0.2,
                        "max_magnitude": 20,
                        "max_kernel": 3,
                    },
                    "random_transform": {
                        "proba": 0.2,
                        "max_val": 125,
                    },
                    "dilation_erosion": {
                        "proba": 0.2,
                        "min_kernel": 1,
                        "max_kernel": 3,
                        "iterations": 1,
                    },
                    "brightness": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "contrast": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "sign_flipping": {
                        "proba": 0.2,
                    },
                },
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "attention": VerticalAttention,
                "decoder": LineDecoderCTC,
            },
            "transfer_learning": None,
            # "transfer_learning": {
            #     # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            #     "encoder": ["encoder", "../../line_OCR/ctc/outputs/iam/checkpoints/best_XX.pt", True, True],
            #     "decoder": ["decoder", "../../line_OCR/ctc/outputs/iam/checkpoints/best_XX.pt", True, True],
            #
            # },
            "input_channels": 3,  # 3 for RGB images, 1 for grayscale images

            # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
            "dropout": 0.5,  # dropout for encoder module
            "dec_dropout": 0.5,  # dropout for decoder module
            "att_dropout": 0,  # dropout for attention module

            "features_size": 256,  # encoder output features maps
            "att_fc_size": 256,  # number of channels for attention sum computation

            "use_location": True,  # use previous attention weights in attention module
            "use_coverage_vector": True,  # use coverage vector in attention module
            "coverage_mode": "clamp",  # mode to use for the coverage vector

            "emb_max_features_width": 250,  # maximum feature width (for use_abs_position)
            "emb_max_features_height": 100,  # maximum feature height (for use_abs_position)

            "use_hidden": True,  # use decoder hidden state in attention (and thus LSTM in decoder)
            "hidden_size": 256,  # number of cells for LSTM decoder hidden state
            "nb_layers_decoder": 1,  # number of layers for LSTM decoder

            "min_height_feat": 15,  # min height for attention module (AdaptivePooling)
            "min_width_feat": 100,  # min width for attention module (AdaptivePooling)
        },

        "training_params": {
            "output_folder": "van_iam_paragraph_learned_stop",  # folder names for logs and weigths
            "max_nb_epochs": 5000,  # max number of epochs for the training
            "max_training_time": 3600 * (24 + 23),  # max training time limit (in seconds)
            "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "batch_size": 8,  # mini-batch size per GPU
            "use_ddp": False,  # Use DistributedDataParallel
            "ddp_port": "10000",  # Port for Distributed Data Parallel communications
            "use_apex": True,  # Enable mix-precision with apex package
            "nb_gpu": torch.cuda.device_count(),
            "optimizer": {
                "class": Adam,
                "args": {
                    "lr": 0.0001,
                    "amsgrad": False,
                }
            },
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["cer", "wer", "diff_len"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
            "max_pred_lines": 30,  # Maximum number of line predictions at evaluation time
            "stop_mode": "learned",  # ["fixed", "early", "learned"]

        },

    }

    if params["training_params"]["stop_mode"] == "learned":
        params["training_params"]["train_metrics"].append("loss_ce")
    params["model_params"]["stop_mode"] = params["training_params"]["stop_mode"]

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)
