import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, predictions, targets):
        combined_loss = 0
        loss_list = torch.zeros(self.task_num)
        for i in range(self.task_num):
            if len(targets.shape) == 3:
                prediction = predictions[i][:, :, 0]
                target = targets[:, :, i]
            else:
                prediction = predictions[i][:, 0]
                target = targets[:, i]
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            scaled_loss = torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
            loss_list[i] = scaled_loss.clone().detach()
            combined_loss += scaled_loss

        return combined_loss, loss_list
