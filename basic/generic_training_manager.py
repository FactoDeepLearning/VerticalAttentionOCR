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

import torch
import os
try:
    import apex
    from apex.parallel import DistributedDataParallel as aDDP
    is_installed_apex = True
except ImportError:
    is_installed_apex = False
    print("Apex not installed")
import copy
import json
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, Linear, InstanceNorm2d
from torch.nn.init import zeros_, ones_, kaiming_uniform_
from tqdm import tqdm
from time import time
from torch.nn.parallel import DistributedDataParallel as tDDP
from basic.generic_dataset_manager import DatasetManager


class GenericTrainingManager:

    def __init__(self, params):
        self.type = None
        self.is_master = False
        self.params = params
        self.models = {}
        self.begin_time = None
        self.dataset = None
        self.paths = None
        self.latest_epoch = -1
        self.latest_batch = 0
        self.total_batch = 0
        self.latest_train_metrics = dict()
        self.latest_valid_metrics = dict()
        self.curriculum_info = dict()
        self.curriculum_info["latest_valid_metrics"] = dict()

        self.optimizer = None
        self.lr_scheduler = None
        self.best = None
        self.writer = None

        self.init_hardware_config()
        self.init_apex_config()
        self.init_paths()
        self.load_dataset()
        self.load_model()

    def init_paths(self):
        ## Create output folders
        output_path = os.path.join("outputs", self.params["training_params"]["output_folder"])
        os.makedirs(output_path, exist_ok=True)
        checkpoints_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        self.paths = {
            "results": results_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

    def load_dataset(self):
        self.params["dataset_params"]["use_ddp"] = self.params["training_params"]["use_ddp"]
        self.params["dataset_params"]["batch_size"] = self.params["training_params"]["batch_size"]
        self.params["dataset_params"]["num_gpu"] = self.params["training_params"]["nb_gpu"]
        self.dataset = DatasetManager(self.params["dataset_params"])
        if self.dataset.charset:
            self.params["model_params"]["vocab_size"] = len(self.dataset.charset)

    def init_apex_config(self):
        if not is_installed_apex:
            if self.params["training_params"]["use_apex"]:
                print("Warning: Apex not used ! (not installed)")
            self.params["training_params"]["use_apex"] = False
        self.apex_config = {
            "level": "O2",
        }
        self.params["dataset_params"]["use_apex"] = self.params["training_params"]["use_apex"]

    def init_hardware_config(self):
        # Debug mode
        if self.params["training_params"]["force_cpu"]:
            self.params["training_params"]["use_ddp"] = False
            self.params["training_params"]["use_apex"] = False
        # Manage Distributed Data Parallel & GPU usage
        self.manual_seed = 1111 if "manual_seed" not in self.params["training_params"].keys() else \
        self.params["training_params"]["manual_seed"]
        self.ddp_config = {
            "master": self.params["training_params"]["use_ddp"] and self.params["training_params"]["ddp_rank"] == 0,
            "address": "localhost" if "ddp_addr" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_addr"],
            "port": "11111" if "ddp_port" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_port"],
            "backend": "nccl" if "ddp_backend" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_backend"],
            "rank": self.params["training_params"]["ddp_rank"],
        }
        self.is_master = self.ddp_config["master"] or not self.params["training_params"]["use_ddp"]
        if self.params["training_params"]["force_cpu"]:
            self.device = "cpu"
        else:
            if self.params["training_params"]["use_ddp"]:
                self.device = torch.device(self.ddp_config["rank"])
                self.params["dataset_params"]["ddp_rank"] = self.ddp_config["rank"]
                self.launch_ddp()
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Print GPU info
        # global
        if (self.params["training_params"]["use_ddp"] and self.ddp_config["master"]) or not self.params["training_params"]["use_ddp"]:
            print("##################")
            print("Available GPUS: {}".format(self.params["training_params"]["nb_gpu"]))
            for i in range(self.params["training_params"]["nb_gpu"]):
                print("Rank {}: {} {}".format(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i)))
            print("##################")
        # local
        print("Local GPU:")
        if self.device != "cpu":
            print("Rank {}: {} {}".format(self.params["training_params"]["ddp_rank"], torch.cuda.get_device_name(), torch.cuda.get_device_properties(self.device)))
        else:
            print("WORKING ON CPU !\n")
        print("##################")

    def load_model(self, reset_optimizer=False):
        def to_DDP(model, use_apex, rank):
            if use_apex:
                return aDDP(model)
            else:
                return tDDP(model, device_ids=[rank])

        # Instanciate Model
        for model_name in self.params["model_params"]["models"].keys():
            self.models[model_name] = self.params["model_params"]["models"][model_name](self.params["model_params"])
            self.models[model_name].to(self.device)  # To GPU or CPU

        # Instanciate optimizer
        self.reset_optimizer()
        if "lr_scheduler" in self.params["training_params"] and self.params["training_params"]["lr_scheduler"]:
            self.lr_scheduler = self.params["training_params"]["lr_scheduler"]["type"](self.optimizer, gamma=self.params["training_params"]["lr_scheduler"]["gamma"])

        # Load previous weights
        checkpoint = None
        if self.params["training_params"]["load_epoch"] in ("best", "last"):
            for filename in os.listdir(self.paths["checkpoints"]):
                # Continue training
                if self.params["training_params"]["load_epoch"] in filename:
                    checkpoint_path = os.path.join(self.paths["checkpoints"], filename)
                    checkpoint = torch.load(checkpoint_path)
                    self.load_save_info(checkpoint)
                    self.latest_epoch = checkpoint["epoch"]
                    self.best = checkpoint["best"]
                    # Make model and optimizer compatible with apex if used
                    if self.params["training_params"]["use_apex"]:
                        models = [self.models[model_name] for model_name in self.models.keys()]
                        models, self.optimizer = apex.amp.initialize(models, self.optimizer, opt_level=self.apex_config["level"])
                        for i, model_name in enumerate(self.models.keys()):
                            self.models[model_name] = models[i]
                    # Make model compatible with Distributed Data Parallel if used
                    if self.params["training_params"]["use_ddp"]:
                        for model_name in self.models.keys():
                            self.models[model_name] = to_DDP(self.models[model_name], self.params["training_params"]["use_apex"], self.ddp_config["rank"])
                    # Load model weights from past training
                    for model_name in self.models.keys():
                        self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(model_name)])
                    # Load optimizer state from past training
                    if not reset_optimizer:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    # Load optimizer scheduler config from past training if used
                    if "lr_scheduler" in self.params["training_params"] and self.params["training_params"]["lr_scheduler"] and "lr_scheduler_state_dict" in checkpoint.keys():
                        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    # Load apex state from past training if used
                    if self.params["training_params"]["use_apex"]:
                        apex.amp.load_state_dict(checkpoint["apex_state_dict"])
                    break

        # Print the number of trained epoch so far with the model
        if self.is_master:
            print("LOADED EPOCH: {}\n".format(self.latest_epoch), flush=True)

        # New training
        if not checkpoint:
            # Weights initialization
            for model_name in self.models.keys():
                self.models[model_name].apply(self.weights_init)
            # Handle transfer learning instructions
            if self.params["model_params"]["transfer_learning"]:
                # Iterates over models
                for model_name in self.params["model_params"]["transfer_learning"].keys():
                    state_dict_name, path, learnable, strict = self.params["model_params"]["transfer_learning"][model_name]
                    # Loading pretrained weights file
                    checkpoint = torch.load(path)
                    try:
                        # Load pretrained weights for model
                        self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(state_dict_name)], strict=strict)
                        print("transfered weights for {}".format(state_dict_name), flush=True)
                    except RuntimeError as e:
                        print(e, flush=True)
                        # if error, try to load each parts of the model (useful if only few layers are different)
                        for key in checkpoint["{}_state_dict".format(state_dict_name)].keys():
                            try:
                                self.models[model_name].load_state_dict({key: checkpoint["{}_state_dict".format(state_dict_name)][key]}, strict=False)
                            except RuntimeError as e:
                                print(e, flush=True)
                    # Set parameters no trainable
                    if not learnable:
                        self.set_model_learnable(self.models[model_name], False)

            # Make model and optimizer compatible with apex if used
            if self.params["training_params"]["use_apex"]:
                models = [self.models[model_name] for model_name in self.models.keys()]
                models, self.optimizer = apex.amp.initialize(models, optimizers=self.optimizer,
                                                             opt_level=self.apex_config["level"],
                                                             keep_batchnorm_fp32=True,
                                                             loss_scale="dynamic")
                for i, model_name in enumerate(self.models.keys()):
                    self.models[model_name] = models[i]

            # make the model compatible with Distributed Data Parallel if used
            if self.params["training_params"]["use_ddp"]:
                for model_name in self.models.keys():
                    self.models[model_name] = to_DDP(self.models[model_name], self.params["training_params"]["use_apex"], self.ddp_config["rank"])
            return

    @staticmethod
    def set_model_learnable(model, learnable=True):
        for p in list(model.parameters()):
            p.requires_grad = learnable

    def save_model(self, epoch, name, keep_weights=False):
        """
        Save model weights
        """
        if not self.is_master:
            return
        to_del = []
        for filename in os.listdir(self.paths["checkpoints"]):
            if name in filename:
                to_del.append(os.path.join(self.paths["checkpoints"], filename))
        path = os.path.join(self.paths["checkpoints"], "{}_{}.pt".format(name, epoch))
        content = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            "apex_state_dict": apex.amp.state_dict() if self.params["training_params"]["use_apex"] else None,
            'best': self.best,
        }
        if self.lr_scheduler:
            content["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        content = self.add_save_info(content)
        for model_name in self.models.keys():
            content["{}_state_dict".format(model_name)] = self.models[model_name].state_dict()
        torch.save(content, path)
        if not keep_weights:
            for path_to_del in to_del:
                if path_to_del != path:
                    os.remove(path_to_del)

    def reset_optimizer(self):
        """
        Reset optimizer learning rate
        """
        parameters = list()
        for model_name in self.models.keys():
            parameters += list(self.models[model_name].parameters())
        self.optimizer = self.params["training_params"]["optimizer"]["class"]\
            (parameters, **self.params["training_params"]["optimizer"]["args"])


    @staticmethod
    def weights_init(m):
        """
        Weights initialization for model training from scratch
        """
        if isinstance(m, Conv2d) or isinstance(m, Linear):
            if m.weight is not None:
                kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, InstanceNorm2d):
            if m.weight is not None:
                ones_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)


    def save_params(self):
        """
        Output text file containing a summary of all hyperparameters chosen for the training
        """
        def compute_nb_params(module):
            return sum([np.prod(p.size()) for p in list(module.parameters())])

        def class_to_str_dict(my_dict):
            for key in my_dict.keys():
                if callable(my_dict[key]):
                    my_dict[key] = my_dict[key].__name__
                elif isinstance(my_dict[key], np.ndarray):
                    my_dict[key] = my_dict[key].tolist()
                elif isinstance(my_dict[key], dict):
                    my_dict[key] = class_to_str_dict(my_dict[key])
            return my_dict

        path = os.path.join(self.paths["results"], "params")
        if os.path.isfile(path):
            return
        params = copy.deepcopy(self.params)
        params = class_to_str_dict(params)
        total_params = 0
        for model_name in self.models.keys():
            current_params = compute_nb_params(self.models[model_name])
            params["model_params"]["models"][model_name] = [params["model_params"]["models"][model_name], "{:,}".format(current_params)]
            total_params += current_params
        params["model_params"]["total_params"] = "{:,}".format(total_params)

        params["hardware"] = dict()
        if self.device != "cpu":
            for i in range(self.params["training_params"]["nb_gpu"]):
                params["hardware"][str(i)] = "{} {}".format(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
        else:
            params["hardware"]["0"] = "CPU"
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    @staticmethod
    def init_metrics(metrics_name):
        """
        Initialization of the metrics specified in metrics_name
        """
        metrics = {
            "nb_samples": 0,
            "weights": 0,
            "names": list(),
            "ids": list(),
        }
        for metric_name in metrics_name:
            if metric_name == "cer":
                metrics["nb_chars"] = 0
                metrics[metric_name] = list()
                continue
            elif metric_name == "wer":
                metrics["nb_words"] = 0
            elif metric_name == "pred":
                metrics[metric_name] = list()
                continue
            elif metric_name == "probas":
                metrics[metric_name] = list()
                continue
            elif metric_name == "diff_len":
                metrics[metric_name] = None
                continue
            metrics[metric_name] = 0
        return metrics

    @staticmethod
    def update_metrics(metrics, batch_metrics):
        """
        Add batch metrics to the metrics
        """
        for key in batch_metrics.keys():
            if key in ["diff_len", ]:
                if metrics[key] is None:
                    metrics[key] = batch_metrics[key]
                else:
                    metrics[key] = np.concatenate([metrics[key], batch_metrics[key]], axis=0)
            elif key in ["pred", ]:
                if len(metrics[key]) == 0:
                    metrics[key] = batch_metrics[key]
                else:
                    for i in range(len(metrics[key])):
                        metrics[key][i] += batch_metrics[key][i]
            else:
                metrics[key] += batch_metrics[key]
        return metrics

    def get_display_values(self, metrics, metrics_name, num_batch):
        """
        format metrics values for shell display purposes
        """
        display_values = {}
        for metric_name in metrics_name:
            if metric_name in ["cer", "cer_force_len", ]:
                edit = metrics[metric_name] if metric_name == "cer_force_len" else np.sum(metrics[metric_name])
                display_values[metric_name] = round(edit / metrics["nb_chars"], 4)
            elif metric_name == "wer":
                display_values[metric_name] = round(metrics[metric_name] / metrics["nb_words"], 4)
            elif metric_name in ["f_measure", "precision", "recall", "IoU", "mAP", "pp_f_measure", "pp_precision", "pp_recall", "pp_IoU", "pp_mAP"]:
                display_values[metric_name] = round(metrics[metric_name] / metrics["weights"], 4)
            elif metric_name in ["diff_len", ]:
                display_values[metric_name] = np.round(np.mean(np.abs(metrics[metric_name])), 3)
            elif metric_name in ["time", "pred", "probas", "nb_max_len", "worst_cer", ]:
                continue
            elif metric_name in ["loss", "loss_ctc", "loss_ce", "loss_ce_end"]:
                display_values[metric_name] = round(metrics[metric_name] / self.latest_batch, 4)
            else:
                display_values[metric_name] = round(metrics[metric_name] / metrics["nb_samples"], 4)
        return display_values

    def backward_loss(self, loss, retain_graph=False):
        """
        Custom loss backward depending on the use of apex package
        """
        if self.params["training_params"]["use_apex"]:
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def train(self):
        # init tensorboard file and output param summary file
        if self.is_master:
            self.writer = SummaryWriter(self.paths["results"])
            self.save_params()
        # init variables
        self.begin_time = time()
        focus_metric_name = self.params["training_params"]["focus_metric"]
        nb_epochs = self.params["training_params"]["max_nb_epochs"]
        interval_save_weights = self.params["training_params"]["interval_save_weights"]
        metrics_name = self.params["training_params"]["train_metrics"]
        display_values = None
        # init curriculum learning
        if "curriculum_learning" in self.params["training_params"].keys() and self.params["training_params"]["curriculum_learning"]:
            self.init_curriculum()
        # perform epochs
        for num_epoch in range(self.latest_epoch+1, nb_epochs):
            # Check maximum training time stop condition
            if self.params["training_params"]["max_training_time"] and time() - self.begin_time > self.params["training_params"]["max_training_time"]:
                break
            # set models trainable
            for model_name in self.models.keys():
                self.models[model_name].train()
            self.latest_epoch = num_epoch
            # init epoch metrics values
            metrics = self.init_metrics(metrics_name)
            t = tqdm(self.dataset.train_loader)
            t.set_description("EPOCH {}/{}".format(num_epoch, nb_epochs))
            # iterates over mini-batch data
            for ind_batch, batch_data in enumerate(t):
                self.latest_batch = ind_batch + 1
                self.total_batch += 1
                # train on batch data and compute metrics
                batch_metrics = self.train_batch(batch_data, metrics_name)
                batch_metrics["names"] = batch_data["names"]
                batch_metrics["ids"] = batch_data["ids"]
                # Merge metrics if Distributed Data Parallel is used
                if self.params["training_params"]["use_ddp"]:
                    batch_metrics = self.merge_ddp_metrics(batch_metrics)
                # Update learning rate via scheduler if one is used
                if self.lr_scheduler and ind_batch % self.params["training_params"]["lr_scheduler"]["step_interval"] == 0:
                    self.lr_scheduler.step()
                # Add batch metrics values to epoch metrics values
                metrics = self.update_metrics(metrics, batch_metrics)
                display_values = self.get_display_values(metrics, metrics_name, ind_batch)
                t.set_postfix(values=str(display_values))
            # log metrics in tensorboard file
            if self.is_master:
                for key in display_values.keys():
                    self.writer.add_scalar('{}_{}'.format(self.params["dataset_params"]["train"]["name"], key), display_values[key], num_epoch)
            self.latest_train_metrics = display_values

            # Handle curriculum learning update
            if self.dataset.train_dataset.curriculum_config:
                self.check_and_update_curriculum()

            # evaluate and compute metrics for valid sets
            if self.params["training_params"]["eval_on_valid"] and num_epoch % self.params["training_params"]["eval_on_valid_interval"] == 0:
                for valid_set_name in self.dataset.valid_loaders.keys():
                    # evaluate set and compute metrics
                    eval_values = self.evaluate(valid_set_name)
                    self.latest_valid_metrics = eval_values
                    # log valid metrics in tensorboard file
                    if self.is_master:
                        for key in eval_values.keys():
                            self.writer.add_scalar('{}_{}'.format(valid_set_name, key), eval_values[key], num_epoch)
                        if valid_set_name == self.params["training_params"]["set_name_focus_metric"] and (self.best is None or \
                                (eval_values[focus_metric_name] < self.best and self.params["training_params"]["expected_metric_value"] == "low") or\
                                (eval_values[focus_metric_name] > self.best and self.params["training_params"]["expected_metric_value"] == "high")):
                            self.save_model(epoch=num_epoch, name="best")
                            self.best = eval_values[focus_metric_name]

            ## save model weights
            if self.is_master:
                self.save_model(epoch=num_epoch, name="last")
                if interval_save_weights and num_epoch % interval_save_weights == 0:
                    self.save_model(epoch=num_epoch, name="weigths", keep_weights=True)
                self.writer.flush()

    def evaluate(self, set_name, **kwargs):
        loader = self.dataset.valid_loaders[set_name]
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()
        metrics_name = self.params["training_params"]["eval_metrics"]
        display_values = None
        # initialize epoch metrics
        metrics = self.init_metrics(metrics_name)
        t = tqdm(loader)
        t.set_description("Evaluation E{}".format(self.latest_epoch))
        with torch.no_grad():
            # iterate over batch data
            for ind_batch, batch_data in enumerate(t):
                self.latest_batch = ind_batch + 1
                # eval batch data and compute metrics
                batch_metrics = self.evaluate_batch(batch_data, metrics_name)
                batch_metrics["names"] = batch_data["names"]
                batch_metrics["ids"] = batch_data["ids"]
                # merge metrics values if Distributed Data Parallel is used
                if self.params["training_params"]["use_ddp"]:
                    batch_metrics = self.merge_ddp_metrics(batch_metrics)
                # add batch metrics to epoch metrics
                metrics = self.update_metrics(metrics, batch_metrics)
                display_values = self.get_display_values(metrics, metrics_name, ind_batch)
                t.set_postfix(values=str(display_values))
        return display_values

    def predict(self, custom_name, sets_list, metrics_name, output=False):
        metrics_name = metrics_name.copy()
        self.dataset.generate_test_loader(custom_name, sets_list)
        loader = self.dataset.test_loaders[custom_name]
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()
        pred_time_metric = False
        if "time" in metrics_name:
            metrics_name.remove("time")
            pred_time_metric = True
        # initialize epoch metrics
        metrics = self.init_metrics(metrics_name)
        t = tqdm(loader)
        t.set_description("Prediction")
        begin_time = time()
        with torch.no_grad():
            for ind_batch, batch_data in enumerate(t):
                # iterates over batch data
                self.latest_batch = ind_batch + 1
                # eval batch data and compute metrics
                batch_metrics = self.evaluate_batch(batch_data, metrics_name)
                batch_metrics["names"] = batch_data["names"]
                batch_metrics["ids"] = batch_data["ids"]
                # merge batch metrics if Distributed Data Parallel is used
                if self.params["training_params"]["use_ddp"]:
                    batch_metrics = self.merge_ddp_metrics(batch_metrics)
                # add batch metrics to epoch metrics
                metrics = self.update_metrics(metrics, batch_metrics)
                display_values = self.get_display_values(metrics, metrics_name, ind_batch)
                t.set_postfix(values=str(display_values))
        pred_time = time() - begin_time
        # add time metric values if requested
        if pred_time_metric:
            metrics["total_time"] = np.round(pred_time, 3)
            metrics["sample_time"] = np.round(pred_time / len(self.dataset.test_datasets[custom_name]), 4)
        # output metrics values if requested
        if output:
            for name in ["probas", ]:
                if name in metrics.keys():
                    path = os.path.join(self.paths["results"], "{}_{}_{}.txt".format(name, custom_name, self.latest_epoch))
                    info = "\n".join(metrics[name])
                    with open(path, "w") as f:
                        f.write(info)
                    del metrics[name]
            self.output(metrics, custom_name)

    def launch_ddp(self):
        """
        Initialize Distributed Data Parallel system
        """
        mp.set_start_method('fork', force=True)
        os.environ['MASTER_ADDR'] = self.ddp_config["address"]
        os.environ['MASTER_PORT'] = str(self.ddp_config["port"])
        dist.init_process_group(self.ddp_config["backend"], rank=self.ddp_config["rank"], world_size=self.params["training_params"]["nb_gpu"])
        torch.cuda.set_device(self.ddp_config["rank"])
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed(self.manual_seed)

    def merge_ddp_metrics(self, metrics):
        """
        Merge metrics when Distributed Data Parallel is used
        """
        for metric_name in metrics.keys():
            if metric_name in ["wer", "cer_force_len", "wer_force_len", "nb_samples", "nb_words", "nb_chars", "nb_max_len",
                               "f_measure", "precision", "recall", "IoU", "mAP", "pp_f_measure", "pp_precision", "pp_recall", "pp_IoU", "pp_mAP"]:
                metrics[metric_name] = self.sum_ddp_metric(metrics[metric_name])
            elif metric_name in ["loss", "loss_ce", "loss_ctc", "loss_ce_end"]:
                metrics[metric_name] = self.sum_ddp_metric(metrics[metric_name], average=True)
            elif metric_name in ["diff_len", "cer", "ids"]:
                metrics[metric_name] = self.cat_ddp_metric(metrics[metric_name])
        return metrics

    def sum_ddp_metric(self, metric, average=False):
        """
        Sum metrics for Distributed Data Parallel
        """
        sum = torch.tensor(metric).to(self.device)
        dist.all_reduce(sum, op=dist.ReduceOp.SUM)
        if average:
            sum.true_divide(dist.get_world_size())
        return sum.item()

    def cat_ddp_metric(self, metric):
        """
        Concatenate metrics for Distributed Data Parallel
        """
        tensor = torch.tensor(metric).unsqueeze(0).to(self.device)
        res = [torch.zeros(tensor.size()).long().to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(res, tensor)
        return list(torch.cat(res, dim=0).flatten().cpu().numpy())

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def train_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def evaluate_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def init_curriculum(self):
        raise NotImplementedError

    def update_curriculum(self):
        raise NotImplementedError

    def output_pred(self, pred, set_name):
        raise NotImplementedError

    def add_checkpoint_info(self, load_mode="last", **kwargs):
        for filename in os.listdir(self.paths["checkpoints"]):
            if load_mode in filename:
                checkpoint_path = os.path.join(self.paths["checkpoints"], filename)
                checkpoint = torch.load(checkpoint_path)
                for key in kwargs.keys():
                    checkpoint[key] = kwargs[key]
                torch.save(checkpoint, checkpoint_path)
            return
        self.save_model(self.latest_epoch, "last")

    def output(self, metrics, set_name):
        """
        Output metrics in text file
        """
        path = os.path.join(self.paths["results"], "predict_{}_{}.txt".format(set_name, self.latest_epoch))
        if "pred" in metrics.keys():
            temp_time = time()
            self.output_pred(metrics["pred"], set_name)
            output_time = time() - temp_time
            if "total_time" in metrics.keys():
                metrics["total_output_time"] = np.round(output_time, 3)
                metrics["sample_output_time"] = np.round(output_time / metrics["nb_samples"], 4)

            del metrics["pred"]

        with open(path, "w") as f:
            for metric_name in metrics.keys():
                if metric_name in ["cer", "cer_force_len"]:
                    edit = metrics[metric_name] if metric_name == "cer_force_len" else np.sum(metrics[metric_name])
                    value = round(edit / metrics["nb_chars"], 4)
                elif metric_name in ["wer", ]:
                    value = round(metrics[metric_name] / metrics["nb_words"], 4)
                elif metric_name in ["loss_ce", ]:
                    value = round(metrics[metric_name] / metrics["nb_samples"], 4)
                elif metric_name in ["f_measure", "precision", "recall", "IoU", "mAP",
                                   "pp_f_measure", "pp_precision", "pp_recall", "pp_IoU", "pp_mAP", ]:
                    value = round(metrics[metric_name] / metrics["weights"], 4)
                elif metric_name in ["total_time", "sample_time", "total_output_time", "sample_output_time"]:
                    value = metrics[metric_name]
                elif metric_name in ["nb_samples", "nb_words", "nb_chars", "nb_max_len"]:
                    value = metrics[metric_name]
                elif metric_name in ["diff_len", ]:
                    f.write("{}: {}\n".format(metric_name, sorted(list(metrics[metric_name]))))
                    f.write("{}-mean_abs: {}\n".format(metric_name, np.mean(np.abs(metrics[metric_name]))))
                    continue
                elif metric_name in ["worst_cer", ]:
                    m = metric_name.split("_")[-1]
                    value = [[c, id] for c, id in zip(metrics[m], metrics["ids"])]
                    value = sorted(value, key=lambda x: x[0], reverse=True)
                    value = value[:50]
                else:
                    continue
                f.write("{}: {}\n".format(metric_name, value))


    def load_save_info(self, info_dict):
        """
        Load curriculum info from saved model info
        """
        if "curriculum_config" in info_dict.keys():
            self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        """
        Add curriculum info to model info to be saved
        """
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def check_and_update_curriculum(self):
        """
        Check curriculum update conditions and update if they are satisfied
        """

        if not self.dataset.train_dataset.curriculum_config:
            return
        curr_metrics = self.dataset.train_dataset.curriculum_config["log_metrics"]
        # Log curriculum metrics in tensorboard file
        if self.is_master:
            for curr_metric in curr_metrics:
                if curr_metric in self.dataset.train_dataset.curriculum_config:
                    self.writer.add_scalar('curriculum_{}'.format(curr_metric), self.dataset.train_dataset.curriculum_config[curr_metric], self.latest_epoch)
        update = True

        # Check if update condition is satified
        for cond in self.dataset.train_dataset.curriculum_config["update_conditions"]:
            metrics = None
            if "set_name" in cond.keys():
                if cond["set_name"] == "train":
                    metrics = self.latest_train_metrics
                elif cond["set_name"] == "valid":
                    metrics = self.latest_valid_metrics
                elif cond["set_name"] == "valid_curriculum":
                    metrics = self.curriculum_info["latest_valid_metrics"]
            if cond["type"] == "no_improvment":
                key = "best_{}_{}".format(cond["metric_name"], cond["set_name"])
                if self.curriculum_info[key]["value"] is None or \
                        (cond["low_or_high"] == "low" and metrics[cond["metric_name"]] < self.curriculum_info[key]["value"]) or \
                        (cond["low_or_high"] == "high" and metrics[cond["metric_name"]] > self.curriculum_info[key]["value"]):
                    self.curriculum_info[key]["value"] = metrics[cond["metric_name"]]
                    self.curriculum_info[key]["epoch"] = self.latest_epoch
                    update = False
                elif cond["nb_epochs"] > self.latest_epoch - self.curriculum_info[key]["epoch"]:
                    update = False
            elif cond["type"] == "limit":
                if (cond["low_or_high"] == "low" and metrics[cond["metric_name"]] < cond["threshold"]) or \
                        (cond["low_or_high"] == "high" and metrics[cond["metric_name"]] > cond["threshold"]):
                    pass
                else:
                    update = False
            elif cond["type"] == "nb_epochs":
                key = "cond_nb_epochs"
                if cond["start_epoch"] > self.latest_epoch or (self.latest_epoch - self.curriculum_info[key]["epoch"]) < cond["nb_epochs"]:
                    update = False
                else:
                    self.curriculum_info[key]["epoch"] = self.latest_epoch
                    if "nb_epochs_update" in cond:
                        mode, param = cond["nb_epochs_update"]
                        if mode == "div":
                            cond["nb_epochs"] = cond["nb_epochs"] // param

        if update:
            # Update curriculum config
            self.update_curriculum()
            self.reset_curr_update_conditions()
            # Reset optimizer learning rate on curriculum update if requested
            if self.dataset.train_dataset.curriculum_config and \
                    "reset_optimizer_on_update" in self.dataset.train_dataset.curriculum_config.keys() and \
                    self.dataset.train_dataset.curriculum_config["reset_optimizer_on_update"]:
                self.save_model(self.latest_epoch, "last")
                self.load_model(reset_optimizer=True)

    def reset_curr_update_conditions(self):
        if not self.dataset.train_dataset.curriculum_config:
            return
        for cond in self.dataset.train_dataset.curriculum_config["update_conditions"]:
            if cond["type"] == "no_improvment":
                key = "best_{}_{}".format(cond["metric_name"], cond["set_name"])
                self.curriculum_info[key] = {
                    "value": None,
                    "epoch": -1,
                }
            elif cond["type"] == "limit":
                key = "limit_{}_{}".format(cond["metric_name"], cond["set_name"])
                self.curriculum_info[key] = {
                    "reached": False
                }
            elif cond["type"] == "nb_epochs":
                key = "cond_nb_epochs"
                self.curriculum_info[key] = {
                    "epoch": self.latest_epoch
                }
