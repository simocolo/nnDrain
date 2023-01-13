import torch
import torch.nn as nn

class SimplifyLinear(nn.Linear):
    """A subclass of torch.nn.Linear that simplifies the weights of the linear layer.

    Parameters:
    - in_size (int): The size of the input tensor.
    - out_size (int): The size of the output tensor.
    - simplify_row (bool): If True, simplifies the rows of the weight matrix.
    - simplify_col (bool): If True, simplifies the columns of the weight matrix.
    - min_row (int): The minimum number of rows to keep when simplifying.
    - min_col (int): The minimum number of columns to keep when simplifying.
    - exclude_from_drain (bool): If True, excludes this layer from the drain.
    - bias (bool): If True, includes a bias term in the linear layer.
    - threshold (float): The threshold for simplification.
    """
    def __init__(
        self,
        in_size,
        out_size,
        simplify_row,
        simplify_col,
        min_row=1,
        min_col=1,
        exclude_from_drain=False,
        bias=True,
        threshold=None,
    ):
        super().__init__(in_size, out_size, bias=bias)
        self.simplify_row = simplify_row
        self.simplify_col = simplify_col
        self.min_row = min_row
        self.min_col = min_col
        self.exclude_from_drain = exclude_from_drain
        self.threshold = threshold

