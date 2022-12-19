import numpy as np
import torch
import torch.nn as nn

import nndrain.utils as utils
from cv2 import threshold


class TensorEdit:
    """A class for editing tensors.

    Parameters:
    - layers (list): A list of layers to be edited.
    """
    def __init__(self, layers):
        self.layers = layers

    def get_min_weight_row_threshold(self, W):
        """Returns the minimum weight and row index for the given weight tensor.

        Parameters:
        - W (torch.Tensor): The weight tensor.

        Returns:
        - tuple: A tuple containing the minimum weight and the index of the row
          with the minimum weight.
        """
        v, i = torch.max(torch.abs(W), dim=1)
        v_min, i_min = torch.min(v, dim=0)
        return v_min, i_min

    def get_min_weight_col_threshold(self, W):
        """Returns the minimum weight and column index for the given weight tensor.

        Parameters:
        - W (torch.Tensor): The weight tensor.

        Returns:
        - tuple: A tuple containing the minimum weight and the index of the column
          with the minimum weight.
        """
        v, i = torch.max(torch.abs(W), dim=0)
        v_min, i_min = torch.min(v, dim=0)
        return v_min, i_min    
    
    def get_max_weight_row_threshold(self, W):
        """Returns the maximum weight and row index for the given weight tensor.

        Parameters:
        - W (torch.Tensor): The weight tensor.

        Returns:
        - tuple: A tuple containing the maximum weight and the index of the row
          with the maximum weight.
        """
        v, i = torch.max(torch.abs(W), dim=1)
        v_max, i_max = torch.max(v, dim=0)
        return v_max, i_max

    def get_max_weight_col_threshold(self, W):
        """Returns the maximum weight and column index for the given weight tensor.

        Parameters:
        - W (torch.Tensor): The weight tensor.

        Returns:
        - tuple: A tuple containing the maximum weight and the index of the column
          with the maximum weight.
        """
        v, i = torch.max(torch.abs(W), dim=0)
        v_max, i_max = torch.max(v, dim=0)
        return v_max, i_max


    def del_weight_row_less_than(self, A, l_value=1e-2, max_removal=0.1, min_size=10):
        """Removes all rows of `A` that have all absolute values below the given threshold.

        Parameters:
        - A (torch.Tensor): The weight tensor.
        - l_value (float): The threshold value.
        - max_removal (float): The maximum fraction of rows to remove.
        - min_size (int): The minimum number of rows to keep.

        Returns:
        - tuple: A tuple containing the modified weight tensor and a list of the indexes
        of the removed rows.
        """
        valid_rows, removed_rows = [], []
        num_rows = A.size(0)
        stop_removing = False
        for row_idx in range(num_rows):
            if len(removed_rows) > num_rows * max_removal:
                stop_removing = True
            if (num_rows - len(removed_rows)) <= min_size:
                stop_removing = True
            if not torch.all(torch.abs(A[row_idx, :]) < l_value) or stop_removing:
                valid_rows.append(row_idx)
            else:
                removed_rows.append(row_idx)
        A = A[valid_rows, :]
        return A, removed_rows

    def del_weight_row_index(self, A, remove_index_list):
        """Removes the specified rows from the weight tensor.

        Parameters:
        - A (torch.Tensor): The weight tensor.
        - remove_index_list (list): A list of row indexes to remove.

        Returns:
        - torch.Tensor: The modified weight tensor.
        """
        num_rows = A.size(0)
        valid_rows = [row_idx for row_idx in range(num_rows) if row_idx not in remove_index_list]
        A = A[valid_rows, :]
        return A


    def del_weight_col_less_than(self, A, l_value=1e-2, max_removal=0.1, min_size=10):
        """Removes all columns of `A` that have all absolute values below the given threshold.

        Parameters:
        - A (torch.Tensor): The weight tensor.
        - l_value (float): The threshold value.
        - max_removal (float): The maximum fraction of columns to remove.
        - min_size (int): The minimum number of columns to keep.

        Returns:
        - tuple: A tuple containing the modified weight tensor and a list of the indexes
        of the removed columns.
        """
        valid_cols, removed_cols = [], []
        num_cols = A.size(1)
        stop_removing = False
        for col_idx in range(num_cols):
            if len(removed_cols) > num_cols * max_removal:
                stop_removing = True
            if (num_cols - len(removed_cols)) <= min_size:
                stop_removing = True
            if not torch.all(torch.abs(A[:, col_idx]) < l_value) or stop_removing:
                valid_cols.append(col_idx)
            else:
                removed_cols.append(col_idx)
        A = A[:, valid_cols]
        return A, removed_cols


    def del_weight_col_index(self, A, remove_index_list):
        """Removes specified columns from the weight tensor `A`.

        Parameters:
        - A (torch.Tensor): The weight tensor.
        - remove_index_list (list): A list of column indexes to remove.

        Returns:
        - torch.Tensor: The modified weight tensor.
        """
        num_cols = A.size(1)
        valid_cols = [col_idx for col_idx in range(num_cols) if col_idx not in remove_index_list]
        A = A[:, valid_cols]
        return A

    def del_bias_index(self, B, remove_index_list):
        """Removes specified bias values from the bias vector `B`.

        Parameters:
        - B (torch.Tensor): The bias vector.
        - remove_index_list (list): A list of bias indexes to remove.

        Returns:
        - torch.Tensor: The modified bias vector.
        """
        num_rows = B.size(0)
        valid_rows = [row_idx for row_idx in range(num_rows) if row_idx not in remove_index_list]
        B = B[valid_rows]
        return B


    def weights_decay(self, p_decay, decay_rate, condition=True):
        """Apply a weight decay with probability `p_decay` and decay rate `decay_rate`.

        Parameters:
        - p_decay (float): The probability of decay.
        - decay_rate (float): The decay value.
        - condition (bool, optional): A condition to execute the method. Defaults to True.

        Returns:
        - None
        """
        if condition and np.random.random() < p_decay:
            for i_l, l in enumerate(self.layers):
                size = l.weight.data.size()
                if size[0] > l.min_row or size[1] > l.min_col:
                    l.weight.data *= (1 - decay_rate)


    def weights_drain(self, p_drain, threshold_coeff=2.0, condition=True):
        """Apply a selective non-linear weight decay with probability `p_drain` in the range [-r_value; r_value].

        Parameters:
        - p_drain (float): The probability of decay.
        - threshold_coeff (float, optional): The threshold coefficient. Multiplying this by the tensor threshold allows us to obtain the actual range where the decay is applied [-r_value; r_value]. Defaults to 2.0.
        - condition (bool, optional): A condition to execute the method. Defaults to True.

        Returns:
        - None
        """
        if condition and np.random.random() < p_drain:
            for i_l, l in enumerate(self.layers):
                if l.threshold is None:
                    self.set_threshold(l)
                r_value = l.threshold * threshold_coeff
                size = l.weight.data.size()
                if not l.exclude_from_drain and (size[0] > l.min_row or size[1] > l.min_col):
                    scale = 0.45 / r_value
                    # l.weight.data = torch.where(abs(l.weight.data)*scale<=r_value, l.weight.data*(0.2*abs(l.weight.data)*scale + 0.9), l.weight.data)
                    l.weight.data = torch.where(abs(l.weight.data) * scale <= r_value, l.weight.data * (0.1 * torch.log(2 * abs(l.weight.data) * scale + 0.1) + 1.0), l.weight.data)

    def set_threshold(self, layer):
        """Set the remove threshold value of the layer.

        Parameters:
        - layer (torch.Tensor): The layer to set the threshold value.

        Returns:
        - None
        """
        threshold_row, _ = self.get_min_weight_row_threshold(layer.weight.data)
        threshold_col, _ = self.get_min_weight_col_threshold(layer.weight.data)
        layer.threshold = min(threshold_row.detach(), threshold_col.detach())


    @torch.no_grad()
    def weights_remove(self, p_remove=0.1, threshold_coeff=0.8, max_removal=1.0, condition=True, verbose=False):
        """Check if it is possible to remove rows or columns from all layers with probability `p_remove`.

        Parameters:
        - p_remove (float, optional): The probability to test if some row or column can be removed. Defaults to 0.1.
        - threshold_coeff (float, optional): The threshold coefficient. Multiplying this by the tensor threshold allows us to obtain the actual threshold for which if all the values of a row or column are below it, the row or column is removed. Defaults to 0.8.
        - max_removal (float, optional): The maximum fraction of row or column indexes that can be removed at a time. Defaults to 1.0.
        - condition (bool, optional): A condition to execute the method. Defaults to True.
        - verbose (bool, optional): If True, print the indexes of the removed rows and columns. Defaults to False.

        Returns:
        - bool: True if rows or columns have been removed.
        """     
        ret = False
        if condition and np.random.random() < p_remove:
            layers = self.layers
            for i_l, l in enumerate(layers):
                if l.threshold is None:
                    continue
                less_value = l.threshold * threshold_coeff
                if l.simplify_row:
                    w_data, removed = self.del_weight_row_less_than(layers[i_l].weight.data, l_value=less_value, max_removal=max_removal, min_size=l.min_row)
                    if removed != []:
                        if verbose:
                            print('removed rows from hidden layer {} -> {}'.format(i_l, removed))
                        layers[i_l].weight = nn.Parameter(w_data)
                        if layers[i_l].bias is not None:
                            layers[i_l].bias = nn.Parameter(self.del_bias_index(layers[i_l].bias.data, remove_index_list=removed))
                        layers[i_l+1].weight = nn.Parameter(self.del_weight_col_index(layers[i_l+1].weight.data, remove_index_list=removed))
                        layers[i_l].out_features -= len(removed)
                        layers[i_l+1].in_features -= len(removed)
                        ret = True


                if l.simplify_col:
                    w_data, removed = self.del_weight_col_less_than(layers[i_l].weight.data, l_value=less_value, max_removal=max_removal, min_size=l.min_col)
                    if removed != []:
                        if verbose: 
                            print('removed cols from hidden layer {} -> {}'.format(i_l, removed))
                        layers[i_l].weight = nn.Parameter(w_data)
                        layers[i_l-1].weight = nn.Parameter(self.del_weight_row_index(layers[i_l-1].weight.data, remove_index_list=removed))
                        if layers[i_l-1].bias is not None:
                            layers[i_l-1].bias = nn.Parameter(self.del_bias_index(layers[i_l-1].bias.data, remove_index_list=removed))
                        layers[i_l-1].out_features -= len(removed)
                        layers[i_l].in_features -= len(removed)
                        ret = True
        return ret
