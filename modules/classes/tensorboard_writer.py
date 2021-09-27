"""Wrapper Class for Tensorboard's SummaryWriter."""
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    """Wrapper Class for Tensorboard's SummaryWriter."""

    def __init__(self, log_dir, targets):
        """
        Initializes the Writer.

        Parameters
        ----------
        log_dir: str
            Output directory
        targets: list[str]
            List of targets
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.targets = targets
        self.n_targets = len(targets)

    def save_metric_list(self, path, metrics, epoch):
        """
        Saves a list of metrics.

        Parameters
        ----------
        path: str
            Output folder
        metrics: object
            Metrics to save
        epoch: int
            Time Id for writer
        """
        if self.n_targets >= 2:
            self.writer.add_scalar(f"{path}/Compound", metrics.sum(), epoch)
        for i in range(self.n_targets):
            self.writer.add_scalar(f"{path}{self.targets[i]}", metrics[i], epoch)

    def save_model_weights(self, model, epoch):
        """
        Saves model weights.

        Parameters
        ----------
        model: object
            Model for which the weights are saved
        epoch: int
            Time Id for writer
        """
        for name, weight in model.named_parameters():
            name, label = name.split(".", 1)
            self.writer.add_histogram(f"{name}/{label}", weight, epoch)
            if epoch != 0 and weight.grad is not None:
                self.writer.add_histogram(f"{name}/{label}_grad", weight.grad, epoch)
