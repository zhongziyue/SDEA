import torch as t
import torch.nn.functional as F
from torch import nn


class Highway(nn.Module):
    def __init__(self, dim, device: t.device):
        super(Highway, self).__init__()

        self.device = device
        self.fc1 = self.init_Linear(in_fea=dim, out_fea=dim, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=dim, out_fea=dim, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        return linear.to(self.device)

    def forward(self, x):
        # normal layer in the formula is H
        # normal_fc = F.tanh(self.fc1(x))
        normal_fc = self.fc1(x).tanh()
        # transformation gate layer in the formula is T
        # transformation_layer = F.sigmoid(self.gate_layer(x))
        transformation_layer = F.sigmoid(self.gate_layer(x))
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = t.mul(normal_fc, transformation_layer)
        allow_carry = t.mul(x, carry_layer)
        information_flow = t.add(allow_transformation, allow_carry)
        return information_flow
