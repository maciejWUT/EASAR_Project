from torchmetrics import Metric
import torch

class Metrics(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_one_hot = torch.zeros_like(preds)
        preds_max_indices = torch.argmax(preds, dim=1)
        preds_one_hot.scatter_(1, preds_max_indices.unsqueeze(1), 1)

        tp = (preds_one_hot == 1) & (target == 1)
        fp = (preds_one_hot == 1) & (target == 0)
        fn = (preds_one_hot == 0) & (target == 1)
        tn = (preds_one_hot == 0) & (target == 0)

        self.true_positives += torch.sum(tp.int(), dim=0)
        self.false_positives += torch.sum(fp.int(), dim=0)
        self.false_negatives += torch.sum(fn.int(), dim=0)
        self.true_negatives += torch.sum(tn.int(), dim=0)
        self.total += torch.sum(target, dim=0)


    def compute(self):
        accuracy = torch.sum(self.true_positives + self.true_negatives) / (torch.sum(self.total)*self.num_classes)
        accuracy = torch.sum(self.true_positives) / torch.sum(self.total)

        return{
            "mean_accuracy": accuracy
        }

    def reset(self):
        self.true_positives = torch.zeros(self.num_classes)
        self.false_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
        self.true_negatives = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)