import torch.nn as nn

class SimplifyLinear(nn.Linear):
    def __init__(self,in_size, out_size, simplify_row, simplify_col, exclude_from_drain, bias=True):
        super(SimplifyLinear, self).__init__(in_size, out_size, bias=bias)
        self.simplify_row = simplify_row
        self.simplify_col = simplify_col
        self.exclude_from_drain = exclude_from_drain