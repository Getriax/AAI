from torch import nn


class ZeroLoss(nn.Module):
    def __init__(self, scale=False):
        self.scale = scale
        super(ZeroLoss, self).__init__()

    def forward(self, inputs):
        zeros = inputs[inputs == 0]

        if self.scale:
            return len(zeros) / len(inputs)

        return len(zeros)
