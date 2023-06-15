# Source: https://github.com/torch-points3d/torch-points3d

import numpy as np
import torch
import os
from torchmetrics.classification import MulticlassConfusionMatrix
from torch_scatter import scatter_add

class ConfusionMatrix(MulticlassConfusionMatrix):
    """TorchMetrics's MulticlassConfusionMatrix but tailored to our
    needs. In particular, new methods allow computing OA, mAcc, mIoU
    and per-class IoU. The `update` method also supports registering
    segment-level predictions along with label histograms. This allows
    speeding up metrics computation without flattening all predictions
    and labels histograms into potentially-huge point-wise tensors.

    :param num_classes: int
        Number of classes in the confusion matrix
    :param ignore_index: int
        Specifies a target value that is ignored and does not
        contribute to the metric calculation
    """

    def __init__(self, num_classes, ignore_index=None):
        super().__init__(
            num_classes, ignore_index=ignore_index, normalize=None,
            validate_args=False)

    def update(self, preds, target):
        """Update state with predictions and targets. Extends the
        `MulticlassConfusionMatrix.update()` with the possibility to
        pass histograms as targets. This is typically useful for
        computing point-wise metrics from segment-wise predictions and
        label histograms.

        :param preds: Tensor
            Predictions
        :param target: Tensor
            Ground truth
        """
        assert not target.is_floating_point()
        assert preds.shape[0] == target.shape[0]
        assert preds.dim() <= 2
        assert target.dim() <= 2
        if target.dim() == 2:
            assert target.shape[1] == 1 or target.shape[1] == self.num_classes

        # If logits or probas are passed for preds, take the argmax for
        # the majority class
        if preds.dim() == 2:
            preds = preds.argmax(dim=1)
        if preds.is_floating_point():
            preds = preds.long()

        # If target is a 2D histogram of labels, we directly compute the
        # confusion matrix from the histograms without computing the
        # corresponding atomic pred-target 1D-tensor pairs
        if target.dim() == 2 and target.shape[1] == self.num_classes:
            if self.ignore_index is not None and \
                    0 <= self.ignore_index < self.num_classes:
                target[self.ignore_index] = 0
            confmat = scatter_add(
                target.float(), preds, dim=0, dim_size=self.num_classes)
            self.confmat += confmat.T.long()
            return

        # Flatten single-column 2D target
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze()

        # Basic parent-class update on 1D tensors
        super().update(preds, target)

    @classmethod
    def from_confusion_matrix(cls, confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        assert not confusion_matrix.is_floating_point()
        cm = cls(confusion_matrix.shape[0])
        cm.confmat = confusion_matrix.long()
        return cm

    @classmethod
    def from_histogram(cls, h):
        """Create a ConfusionMatrix from 2D tensors representing label
        histograms. The metrics are computed assuming the prediction
        associated with each histogram is the dominant label.
        """
        assert h.ndim == 2
        assert not h.is_floating_point()
        cm = cls(h.shape[1])
        cm(h.argmax(dim=1), h)
        return cm

    def iou(self, as_percent=True):
        """Computes the Intersection over Union of each class in the
        confusion matrix

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]

        Return:
            (iou, missing_class_mask) - iou for class as well as a mask
            highlighting existing classes
        """
        TP_plus_FN = self.confmat.sum(dim=0)
        TP_plus_FP = self.confmat.sum(dim=1)
        TP = self.confmat.diag()
        union = TP_plus_FN + TP_plus_FP - TP
        iou = 1e-8 + TP / (union + 1e-8)
        existing_class_mask = union > 1e-3
        if as_percent:
            iou *= 100
        return iou, existing_class_mask

    def oa(self, as_percent=True):
        """Compute the Overall Accuracy of the confusion matrix.

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        confusion_matrix = self.confmat
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.num_classes):
            for column in range(self.num_classes):
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        if as_percent:
            matrix_diagonal *= 100
        return float(matrix_diagonal) / all_values

    def miou(self, missing_as_one=False, as_percent=True):
        """Computes the mean Intersection over Union of the confusion
        matrix. Get the mIoU metric by ignoring missing labels.

        :param missing_as_one: bool
            If True, then treats missing classes in the IoU as 1
        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        values, existing_classes_mask = self.iou(as_percent=as_percent)
        if existing_classes_mask.sum() == 0:
            return 0
        if missing_as_one:
            values[~existing_classes_mask] = 1
            existing_classes_mask[:] = True
        return values[existing_classes_mask].sum() / existing_classes_mask.sum()

    def macc(self, as_percent=True):
        """Compute the mean of per-class accuracy in the confusion
        matrix.

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        re = 0
        label_presents = 0
        for i in range(self.num_classes):
            total_gt = self.confmat[i, :].sum()
            if total_gt:
                label_presents += 1
                re = re + self.confmat[i][i] / max(1, total_gt)
        if label_presents == 0:
            return 0
        if as_percent:
            re *= 100
        return re / label_presents

    def print_metrics(self, class_names=None):
        """Helper to print the OA, mAcc, mIoU and per-class IoU.

        :param class_names: optional list(str)
            List of class names to be used for pretty-printing per-class
            IoU scores
        """
        oa = self.oa()
        macc = self.macc()
        miou = self.miou()
        class_iou = self.iou()

        print(f'OA: {oa:0.2f}')
        print(f'mAcc: {macc:0.2f}')
        print(f'mIoU: {miou:0.2f}')

        class_names = class_names if class_names is not None \
            else [f'class-{i}' for i in range(self.num_classes)]
        for c, iou, seen in zip(class_names, class_iou[0], class_iou[1]):
            if not seen:
                print(f'  {c:<13}: not seen')
                continue
            print(f'  {c:<13}: {iou:0.2f}')


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=5)

    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().float().numpy()

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")
