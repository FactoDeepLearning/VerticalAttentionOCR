import torch
from torch.nn.functional import log_softmax
from torch.nn import Conv2d, AdaptiveMaxPool2d
from torch.nn import Module


class Decoder(Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.vocab_size = params["vocab_size"]

        self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.end_conv = Conv2d(in_channels=256, out_channels=self.vocab_size+1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.ada_pool(x)
        x = self.end_conv(x)
        x = torch.squeeze(x, dim=2)
        return log_softmax(x, dim=1)

