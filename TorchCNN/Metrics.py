from torchmetrics import Metric
import torch

class Metrics(Metric):
    """
    A custom metric class for calculating per-class metrics such as true positives, false positives,
    false negatives, true negatives, and total counts. It also computes mean accuracy across classes.

    Attributes:
        num_classes (int): The number of classes in the classification task.
        dist_sync_on_step (bool): Whether to synchronize metrics across distributed processes during each step.

    States:
        true_positives (torch.Tensor): Tensor tracking the count of true positives for each class.
        false_positives (torch.Tensor): Tensor tracking the count of false positives for each class.
        false_negatives (torch.Tensor): Tensor tracking the count of false negatives for each class.
        true_negatives (torch.Tensor): Tensor tracking the count of true negatives for each class.
        total (torch.Tensor): Tensor tracking the total count of instances per class.
    """
    def __init__(self, num_classes, dist_sync_on_step=False):
        """
        Initialize the Metrics class.

        Args:
            num_classes (int): Number of classes in the classification task.
            dist_sync_on_step (bool, optional): Synchronize metric states across distributed processes. Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric states based on predictions and targets.

        Args:
            preds (torch.Tensor): Predicted outputs (logits or probabilities) of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
        """
        # Convert predictions to one-hot encoding
        preds_one_hot = torch.zeros_like(preds)
        preds_max_indices = torch.argmax(preds, dim=1)
        preds_one_hot.scatter_(1, preds_max_indices.unsqueeze(1), 1)

        # Calculate true positives, false positives, false negatives, and true negatives
        tp = (preds_one_hot == 1) & (target == 1)
        fp = (preds_one_hot == 1) & (target == 0)
        fn = (preds_one_hot == 0) & (target == 1)
        tn = (preds_one_hot == 0) & (target == 0)

        # Update the states
        self.true_positives += torch.sum(tp.int(), dim=0)
        self.false_positives += torch.sum(fp.int(), dim=0)
        self.false_negatives += torch.sum(fn.int(), dim=0)
        self.true_negatives += torch.sum(tn.int(), dim=0)
        self.total += torch.sum(target, dim=0)

    def compute(self):
        """
        Compute the mean accuracy across all classes.

        Returns:
            dict: A dictionary containing the mean accuracy with key `mean_accuracy`.
        """
        # Calculate mean accuracy
        accuracy = torch.sum(self.true_positives + self.true_negatives) / (torch.sum(self.total) * self.num_classes)
        accuracy = torch.sum(self.true_positives) / torch.sum(self.total)

        return {
            "mean_accuracy": accuracy
        }

    def reset(self):
        """
        Reset the metric states to their initial values.
        """
        self.true_positives = torch.zeros(self.num_classes)
        self.false_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
        self.true_negatives = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
