import argparse
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP suitable for recognizing single characters."""

    def __init__(
            self,
            data_config: Dict[str, Any],
            args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = int(np.prod(data_config["input_dims"]))
        output_dim = len(data_config["mapping"])
        self.dropout = nn.Dropout(0.5)

        fcs_dims = self.args.get("fcs_dims").split(",")
        fcs_dims = [int(d) for d in fcs_dims]

        self.fcs = nn.ModuleList()
        for i in range(len(fcs_dims)):
            if i == 0:
                self.fcs.append(nn.Linear(input_dim, fcs_dims[i]))
            else:
                self.fcs.append(nn.Linear(fcs_dims[i - 1], fcs_dims[i]))
        self.fcs.append(nn.Linear(fcs_dims[-1], output_dim))
        print(self.fcs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i, fc in enumerate(self.fcs):
            if i < len(self.fcs) - 1:
                x = F.relu(fc(x))
                x = self.dropout(x)
            else:
                x = fc(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fcs_dims", type=str, default="1024,128,32,32",
                            help="Comma-separated list of fully connected layer dimensions")
        return parser
