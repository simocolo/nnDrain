import torch
import torch.nn as nn

class SimplifyLinear(nn.Linear):
    def __init__(self,in_size, out_size, simplify_row, simplify_col, min_row, min_col, exclude_from_drain=False, bias=True, threshold=None):
        super(SimplifyLinear, self).__init__(in_size, out_size, bias=bias)
        self.simplify_row = simplify_row
        self.simplify_col = simplify_col
        self.min_row = min_row
        self.min_col = min_col
        self.exclude_from_drain = exclude_from_drain
        self.threshold = threshold


