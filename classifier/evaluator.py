import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler, freeze_parameters
import pdb
import time
from torchmetrics.classification import MulticlassCalibrationError
from utils.display_results import get_measures, print_measures

class Evaluator():
    def __init__(self, args):
        self.args = args
        self.task_to_original_acc = {}
        self.time_step_to_acc = {}
        self.time_step_to_taw_acc = {}
        self.task_to_ood_metrics = {}
        self.time_step_to_future_task_acc = {}
        self.time_step_to_test_id_to_module_id = {}
        
        self.use_fc_head = getattr(args, 'use_fc_head', False)  # Lấy từ args, mặc định False
        self.old_means = None

    def flush_task_to_accs(self):
        self.task_to_original_acc = {}

    def compute_backward_transfer(self, curr_task_accs):
        bwt_scores = []
        for t in range(len(curr_task_accs) -1):
            acc = curr_task_accs[t]
            bwt = acc - self.task_to_original_acc[t]
            bwt_scores.append(bwt)
        print(f"Average BWT score: {np.mean(bwt_scores)}")


    def mask_classes(self, outputs: torch.Tensor, k: int) -> None:
        """
        Given the output tensor, the dataset at hand and the current task,
        masks the former by setting the responses for the other tasks at -inf.
        It is used to obtain the results for the task-il setting.
        :param outputs: the output tensor
        :param dataset: the continual dataset
        :param k: the task index
        """
        outputs[:, 0:k * self.args.class_per_task] = -float('inf')
        outputs[:, (k + 1) * self.args.class_per_task:
                self.args.num_task * self.args.class_per_task] = -float('inf')

    @staticmethod
    def compute_confidence_score(preds, metric="energy"):
        conf = None
        if metric == "energy":
            if preds.dim() == 3:
                preds = preds.mean(1)
            conf = torch.logsumexp(preds, dim=-1)
        elif metric == "softmax":
            if preds.dim() == 3:
                preds = preds.mean(1)
            conf, _ = torch.max(preds.softmax(dim=-1), dim=1)
        elif metric == "variance":
            conf = -torch.var(preds, dim=1).sum(1)
        elif metric == "variance_softmax":
            conf = -torch.var(preds.softmax(dim=-1), dim=1).sum(1)
        elif metric == "variance_max_prob":
            preds, _ = torch.max(preds.softmax(dim=-1), dim=-1)
            conf = -torch.var(preds, dim=1)#.sum(1)
        else:
            raise NotImplementedError(f"Confidence metric: '{metric}' is not defined!")
        return conf 
    
    def compute_ood_scores(self, id_preds, ood_test_loader, num_test=None, test_class=None):
        ood_preds = []
        total_count, acc_count = 0, 0
        for i, (x, y, idx) in tqdm(enumerate(ood_test_loader), total=len(ood_test_loader), desc=f"Running OOD inference:"):
            pred_y_, _ = self.inference(x.cuda(device=self.args.default_gpu), y, num_test=num_test, test_class=test_class)
            if pred_y_.dim() == 3:
                pred_y_ = pred_y_.permute(1, 0, 2)
            ood_preds.append(pred_y_.clone().cpu())
            pred_y = pred_y_.mean(0) if pred_y_.dim() == 3 else pred_y_
            if self.use_fc_head:
                pred_y = pred_y  # FC head đã áp dụng trong inference, không cần softmax lại
            else:
                pred_y = pred_y.softmax(dim=-1)
            _, top_labels = pred_y.topk(1, dim=-1)
            top_labels = top_labels.squeeze(-1) if top_labels.dim() > 1 else top_labels
            y_device = y.cuda(device=self.args.default_gpu)
            correct = (top_labels == y_device).sum().cpu().numpy()
            acc_count += correct
            total_count += y.shape[0]

        ood_preds = torch.cat(ood_preds, 0)
        acc = acc_count*1.0/total_count
        acc = acc.item()
        self.time_step_to_future_task_acc[self.args.sess] = acc
        print(f"Future tasks avg acc: {np.mean(list(self.time_step_to_future_task_acc.values()))} {acc}")
        print(f"Total ID examples: {id_preds.shape[0]}, Total OOD examples: {ood_preds.shape[0]}")
        for metric in ["energy"]:#, "softmax", "variance", "variance_softmax", "variance_max_prob"]:
            if "variance" in metric and id_preds.dim() == 2:
                continue
            confidence_id = self.compute_confidence_score(id_preds, metric=metric)
            confidence_ood = self.compute_confidence_score(ood_preds, metric=metric)
            measures = get_measures(confidence_id, confidence_ood)
            print_measures(measures[0], measures[1], measures[2], metric)
            if metric == "softmax":
                self.task_to_ood_metrics[self.args.sess] = [measures[0], measures[1], measures[2]]
                if self.args.sess > 0:
                    all_vals = list(self.task_to_ood_metrics.values())
                    means_ = np.mean(all_vals, 0)
                    print(f"Average {metric}: FPR95: {means_[2]}  || AUROC: {means_[0]} || AUPR: {means_[1]}")
    
    def map_class_id_to_module_id(self, class_id):
        module_id = torch.div(class_id, self.args.class_per_task, rounding_mode='trunc')
        return module_id
    
    @torch.no_grad()
    def _accuracy(self, loaders, num_test=None, test_class=None, only_eval=False, ood_test_loader=None):
        visual_feats, textual_feats, indices, labels = [],[], [], []
        # pdb.set_trace()
        accs, accs_mask_classes = [], []
        calibration_errors = []
        task_to_module_accuracy = {}
        self.calibration_evaluator = MulticlassCalibrationError(num_classes=len(self.current_class_names)) if self.args.compute_ece else None 
        id_preds = []
        inference_times = []
        if self.args.sess >= 0:
        #     return 0
        # else:
            for k, loader in enumerate(loaders):
                total_count=0
                acc_count =0
                correct_mask_classes = 0
                task_calibration_errors=[]
                selected_module_ids = []
                for i, (x, y, idx) in tqdm(enumerate(loader), total=len(loader), desc=f"Task {k} inference:"):
                    start_time = time.time()
                    pred_y_, feats = self.inference(x.cuda(device=self.args.default_gpu), y, num_test=num_test, test_class=test_class)
                    inference_times.append(time.time() - start_time)
                    
                    # Xử lý shape của predictions
                    if pred_y_.dim() == 3:
                        # Shape: [num_forward_times, batch_size, num_classes]
                        batch_size = pred_y_.shape[1]
                        pred_y = pred_y_.mean(0)  # [batch_size, num_classes]
                    elif pred_y_.dim() == 2:
                        # Shape: [batch_size, num_classes]
                        batch_size = pred_y_.shape[0]
                        pred_y = pred_y_
                    else:
                        raise ValueError(f"Unexpected pred_y_ shape: {pred_y_.shape}")

                    # Đảm bảo batch size khớp với labels
                    if batch_size != y.shape[0]:
                        min_batch_size = min(batch_size, y.shape[0])
                        pred_y = pred_y[:min_batch_size]
                        y = y[:min_batch_size]
                        batch_size = min_batch_size

                    # Apply softmax/FC head
                    if self.use_fc_head:
                        logits = pred_y
                    else:
                        logits = pred_y.softmax(dim=-1)

                    # Get predictions với shape đúng
                    _, top_labels = logits.topk(1, dim=-1)  # Shape: [batch_size, 1]
                    top_labels = top_labels.squeeze(-1)  # Shape: [batch_size]

                    # Compute accuracy với shape đã sửa
                    y_device = y.cuda(device=self.args.default_gpu)
                    top_labels_device = top_labels.cuda(device=self.args.default_gpu)
            
                    # Kiểm tra shape cuối cùng
                    assert y_device.shape[0] == top_labels_device.shape[0], f"Shape mismatch: y={y_device.shape}, top_labels={top_labels_device.shape}"
                    correct = (top_labels == y_device).sum().cpu().numpy()
                    acc_count += correct
                    total_count += batch_size
                    if self.args.viz_module_selection:
                        selected_module_id = self.map_class_id_to_module_id(top_labels_device)
                        selected_module_ids.append(selected_module_id)
                    if self.args.compute_ece:
                        task_calibration_errors.append(self.calibration_evaluator(logits, y_device))
                    if self.args.compute_ram:
                        visual_feats.append(feats[0])
                        textual_feats.append(feats[1])
                        indices.append(deepcopy(idx))
                        labels.append(deepcopy(y))
                        del idx 
                        del y 

                    if self.args.eval_ood_score and ood_test_loader is not None:
                        if pred_y_.dim() == 3:
                            pred_y_ood = pred_y_.permute(1, 0, 2)  # [batch_size, num_forward_times, num_classes]
                        else:
                            pred_y_ood = pred_y_.clone()
                        id_preds.append(pred_y_ood.clone().cpu())
                    
                    pred_y_taw = pred_y_.mean(0) if pred_y_.dim() == 3 else pred_y_.clone()
                    self.mask_classes(pred_y_taw, k)
                    _, taw_pred = pred_y_taw.topk(1, dim=-1)
                    taw_pred = taw_pred.squeeze(-1)  # Ensure correct shape
                    
                    # Đảm bảo shape cho TaW accuracy
                    if taw_pred.shape[0] != y_device.shape[0]:
                        min_size = min(taw_pred.shape[0], y_device.shape[0])
                        taw_pred = taw_pred[:min_size]
                        y_device_taw = y_device[:min_size]
                    else:
                        y_device_taw = y_device
                        
                    correct_mask_classes += (taw_pred == y_device_taw).sum().cpu().numpy()

                # Compute final accuracies
                acc = acc_count * 1.0 / total_count if total_count > 0 else 0
                acc = acc.item() if hasattr(acc, 'item') else acc
                accs.append(acc)

                acc_taw = correct_mask_classes * 1.0 / total_count if total_count > 0 else 0
                acc_taw = acc_taw.item() if hasattr(acc_taw, 'item') else acc_taw
                accs_mask_classes.append(acc_taw)

                if not only_eval and k == len(loaders) - 1:
                    self.task_to_original_acc[self.args.sess] = acc

                if self.args.compute_ece:
                    calibration_errors.extend(task_calibration_errors)

                if self.args.viz_module_selection:
                    selected_module_ids = torch.cat(selected_module_ids)
                    module_ids, counts = torch.unique(selected_module_ids, return_counts=True)
                    individual_task_allocations = {j: 0 for j in range(len(loaders))}
                    for module_id, count in zip(module_ids, counts):
                        individual_task_allocations[module_id.item()] += count.item()
                    task_to_module_accuracy[k] = {task_label: count / total_count * 100. for task_label, count in list(individual_task_allocations.items())}
                    
            print(f"Average inference time: {np.mean(inference_times)}")
            if self.args.viz_module_selection:
                self.time_step_to_test_id_to_module_id[self.args.sess] = task_to_module_accuracy
                print(self.time_step_to_test_id_to_module_id)

            if self.args.eval_ood_score and ood_test_loader is not None:
                self.compute_ood_scores(torch.cat(id_preds, 0), ood_test_loader, num_test=num_test, test_class=test_class)

            if self.args.compute_ram:
                visual_feats = torch.cat(visual_feats)
                textual_feats = torch.cat(textual_feats)
                indices = torch.cat(indices)
                labels = torch.cat(labels)
                self.args.ram_computer.compute_rotation_angle_matrix(self.args.sess, labels, visual_feats, textual_feats, indices)
            
            if self.args.compute_ece:
                print(f"Avg. Expected Calibration Error: {torch.stack(calibration_errors).mean()}")
            
            acc = np.mean(accs)
            self.time_step_to_acc[self.args.sess] = acc 

            acc_taw = np.mean(accs_mask_classes)
            self.time_step_to_taw_acc[self.args.sess] = acc_taw

            print(f"Acc avg: {np.mean(list(self.time_step_to_acc.values()))}, Acc last: {acc}")
            print(f"TaW Acc avg: {np.mean(list(self.time_step_to_taw_acc.values()))}, TaW Acc last: {acc_taw}")

            if self.args.sess > 0 and self.args.compute_bwt:
                self.compute_backward_transfer(accs)
            
            return acc

    @torch.no_grad()
    def _accuracy_mpc(self, loader):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        visual_feats, textual_feats, indices, labels = [],[], [], []
        
        for i, (x, y, idx) in tqdm(enumerate(loader), total=len(loader), desc='running inference'):
            pred_y, feats = self.inference(x.cuda(device=self.args.default_gpu), y)
            
            if self.args.compute_ram:
                visual_feats.append(feats[0])
                textual_feats.append(feats[1])
                indices.extend(idx)
                labels.extend(y)
            
            # SỬA: Xử lý shape consistency
            if pred_y.dim() == 3:
                pred_y = pred_y.mean(0)
            
            # Đảm bảo batch size khớp
            batch_size = pred_y.shape[0]
            if batch_size != y.shape[0]:
                min_batch_size = min(batch_size, y.shape[0])
                pred_y = pred_y[:min_batch_size]
                y = y[:min_batch_size]
                
            if self.use_fc_head:
                logits = pred_y  # FC head đã áp dụng
            else:
                logits = pred_y.softmax(dim=-1)
                
            _, top_labels = logits.topk(1, dim=-1)
            top_labels = top_labels.squeeze(-1)  # Ensure [batch_size] shape
            
            y_device = y.cuda(device=self.args.default_gpu)
            top_labels_device = top_labels.cuda(device=self.args.default_gpu)
            
            # Compute per-class accuracy
            for c in range(n_class):
                correct_mask = (top_labels_device == y_device) & (y_device == c)
                class_mask = (y_device == c)
                
                acc_per_class[c] += correct_mask.sum().item()
                count_per_class[c] += class_mask.sum().item()
        
        # Compute final accuracy
        acc = [a*1.0/c if c > 0 else 0 for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()

        if self.args.compute_ram:
            visual_feats = torch.cat(visual_feats)
            textual_feats = torch.cat(textual_feats)
            self.args.ram_computer.compute_rotation_angle_matrix(self.args.sess, labels, visual_feats, textual_feats, indices)
        
        return acc

    @torch.no_grad()
    def accuracy(self, loaders, num_test=None, test_class=None, mean_per_class=False, only_eval=False, ood_test_loader=None):
        if mean_per_class:
            return self._accuracy_mpc(loaders)
        else:
            return self._accuracy(loaders, num_test, test_class, only_eval=only_eval, ood_test_loader=ood_test_loader)

        
    def post_training(self, finalize=False):
        pass 

    def finetuning(self, data=None):
        pass 
