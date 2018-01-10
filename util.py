import torch


class Accuracy(object):
    def __init__(self, ignore_index=None):
        self.ignore_label = ignore_index

    def __call__(self, y, t):
        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = torch.sum(mask.float())
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            pred = pred.masked_fill(mask, self.ignore_label)
            count = torch.sum((pred == t).float()) - ignore_cnt
            total = torch.numel(t) - ignore_cnt

            if total == 0:
                return torch.FloatTensor([0.0])
            else:
                return count / total
        else:
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            return torch.mean((pred == t).float())


def accuracy(y, t, ignore_index=None):
    return Accuracy(ignore_index=ignore_index)(y, t)
