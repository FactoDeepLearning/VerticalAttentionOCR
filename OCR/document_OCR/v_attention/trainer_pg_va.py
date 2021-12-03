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

from basic.generic_training_manager import GenericTrainingManager
from torch.nn import CrossEntropyLoss, CTCLoss
import torch
from basic.utils import edit_wer_from_list, nb_chars_from_list, nb_words_from_list, LM_ind_to_str
import numpy as np
import editdistance
import re


class Manager(GenericTrainingManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def get_init_hidden(self, batch_size):
        num_layers = self.params["model_params"]["nb_layers_decoder"]
        hidden_size = self.params["model_params"]["hidden_size"]
        return torch.zeros((num_layers, batch_size, hidden_size), device=self.device), torch.zeros((num_layers, batch_size, hidden_size), device=self.device)

    def train_batch(self, batch_data, metric_names):
        loss_ctc_func = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        loss_ce_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])
        global_loss = 0
        total_loss_ctc = 0
        total_loss_ce = 0
        self.optimizer.zero_grad()

        x = batch_data["imgs"].to(self.device)
        y = [l.to(self.device) for l in batch_data["line_labels"]]
        y_len = batch_data["line_labels_len"]
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        batch_size = y[0].size()[0]

        mode = self.params["training_params"]["stop_mode"]
        max_nb_lines = self.params["training_params"]["max_pred_lines"] if mode == "fixed" else len(y)
        num_iter = max_nb_lines if mode == "fixed" else max_nb_lines+1
        for i in range(len(y), num_iter):
            y.append(torch.ones((batch_size, 1), dtype=torch.long , device=self.device)*self.dataset.tokens["pad"])
            y_len.append([0 for _ in range(batch_size)])

        status = "init"
        features = self.models["encoder"](x)
        batch_size, c, h, w = features.size()
        attention_weights = torch.zeros((batch_size, h), dtype=torch.float, device=self.device)
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None

        line_preds = [list() for _ in range(batch_size)]
        for i in range(num_iter):
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            status = "inprogress"
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            
            if mode in ["fixed", "early"] or i < max_nb_lines:
                probs, hidden = self.models["decoder"](context_vector, hidden)
                loss_ctc = loss_ctc_func(probs.permute(2, 0, 1), y[i], x_reduced_len, y_len[i])
                total_loss_ctc += loss_ctc.item()
                global_loss += loss_ctc
            
            if mode == "learned":
                gt_decision = torch.ones((batch_size, ), device=self.device, dtype=torch.long)
                for j in range(batch_size):
                    if y_len[i][j] == 0:
                        if i > 0 and y_len[i-1][j] == 0:
                            gt_decision[j] = self.dataset.tokens["pad"]
                        else:
                            gt_decision[j] = 0
                loss_ce = loss_ce_func(decision, gt_decision)
                total_loss_ce += loss_ce.item()
                global_loss += loss_ce

            line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if y_len[i][j] > 0 else None for j, lp in enumerate(probs)]
            for i, lp in enumerate(line_pred):
                if lp is not None:
                    line_preds[i].append(lp)

        self.backward_loss(global_loss)
        self.optimizer.step()

        metrics = self.compute_metrics(line_preds, batch_data["raw_labels"], metric_names, from_line=True)
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = total_loss_ctc / metrics["nb_chars"]
        if "loss_ce" in metric_names:
            metrics["loss_ce"] = total_loss_ce
        return metrics

    def evaluate_batch(self, batch_data, metric_names):

        def append_preds(pg_preds, line_preds):
            for i, lp in enumerate(line_preds):
                if lp is not None:
                    pg_preds[i].append(lp)
            return pg_preds

        x = batch_data["imgs"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        status = "init"
        mode = self.params["training_params"]["stop_mode"]
        max_nb_lines = self.params["training_params"]["max_pred_lines"]
        features = self.models["encoder"](x)
        batch_size, c, h, w = features.size()
        attention_weights = torch.zeros((batch_size, h), device=self.device, dtype=torch.float)
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None
        preds = [list() for _ in range(batch_size)]
        end_pred = [None for _ in range(batch_size)]

        for i in range(max_nb_lines):
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            probs, hidden = self.models["decoder"](context_vector, hidden)
            status = "inprogress"

            line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(probs)]
            if mode == "learned":
                decision = [torch.argmax(d, dim=0) for d in decision]
                for k, d in enumerate(decision):
                    if d == 0 and end_pred[k] is None:
                        end_pred[k] = i

            if mode in ["learned", "early"]:
                for k, p in enumerate(line_pred):
                    if end_pred[k] is None and np.all(p == self.dataset.tokens["blank"]):
                        end_pred[k] = i
            line_pred = [l if end_pred[j] is None else None for j, l in enumerate(line_pred)]
            preds = append_preds(preds, line_pred)
            if np.all([end_pred[k] is not None for k in range(batch_size)]):
                break

        metrics = self.compute_metrics(preds, batch_data["raw_labels"], metric_names, from_line=True)
        if "diff_len" in metric_names:
            end_pred = [end_pred[k] if end_pred[k] is not None else i for k in range(len(end_pred))]
            diff_len = np.array(end_pred)-np.array(batch_data["nb_lines"])
            metrics["diff_len"] = diff_len
        return metrics

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def compute_metrics(self, ind_x, str_y,  metric_names=list(), from_line=False):
        if from_line:
            str_x = list()
            for lines_token in ind_x:
                list_str = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in lines_token]
                str_x.append(re.sub("( )+", ' ', " ".join(list_str).strip(" ")))
        else:
            str_x = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in ind_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u, v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
        metrics["nb_samples"] = len(str_x)
        return metrics
