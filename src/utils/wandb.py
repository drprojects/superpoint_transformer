import wandb
import torch


__all__ = ['wandb_confusion_matrix']


def wandb_confusion_matrix(cm, class_names=None, title=None):
    """Replaces the "normal" wandb way of logging a confusion matrix:

    https://github.com/wandb/wandb/blob/main/wandb/plot/confusion_matrix.py

    Indeed, the native wandb confusion matrix logging requires the
    element-wise prediction and ground truth. This is not adapted when
    we already have the confusion matrix at hand or that the number of
    elements is too large (eg point clouds).

    :param cm:
    :return:
    """
    assert isinstance(cm, torch.Tensor)
    assert cm.dim() == 2
    assert cm.shape[0] == cm.shape[1]
    assert not cm.is_floating_point()

    # Move confusion matrix to CPU and convert to list
    cm = cm.cpu().tolist()
    num_classes = len(cm)

    # Prepare class names
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(0, num_classes)]

    # Convert to wandb table format
    data = []
    for i in range(num_classes):
        for j in range(num_classes):
            data.append([class_names[i], class_names[j], cm[i][j]])

    columns = ["Actual", "Predicted", "nPredictions"]
    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=columns, data=data),
        {x: x for x in columns},
        {"title": title or ""})
