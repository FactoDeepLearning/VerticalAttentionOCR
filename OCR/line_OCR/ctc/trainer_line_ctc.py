from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
import editdistance
import torch
from torch.nn import CTCLoss


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        self.optimizer.zero_grad()
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        self.backward_loss(loss)
        self.optimizer.step()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")

        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, x, y, x_len, y_len, loss=None, metric_names=list()):
        batch_size = y.shape[0]
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss / metrics["nb_chars"]
        metrics["nb_samples"] = len(x)
        return metrics
