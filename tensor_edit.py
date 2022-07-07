import torch
import torch.nn as nn
import numpy as np

class TensorEdit:
    def __init__(self, layers):
        """
        input the layers to be edited
        """
        self.layers = layers

    def del_weight_row_less_than(self, A, l_value=1e-2, max_removal=0.1, min_size=10):
        """
        remove all rows of A that have all abs(values) below threshold l_value
        
        :param A: weights tensor
        :param l_value: threshold value
        :param max_removal: maximum fraction of rows to remove
        :param min_size: minimum number of rows to keep
        :return: tuple (weights, indexes of removed rows)
        """
        valid_rows, removed_rows = [], []
        num_rows = A.size(0)
        stop_removing = False
        for row_idx in range(num_rows):
            if len(removed_rows)>num_rows*max_removal or (num_rows-len(removed_rows))<=min_size:
                stop_removing = True
            if not torch.all(torch.abs(A[row_idx, :]) < l_value) or stop_removing:
                valid_rows.append(row_idx)
            else:
                removed_rows.append(row_idx)
        A = A[valid_rows, :]
        return A, removed_rows

    def del_weight_row_index(self, A, remove_index_list):
        """
        remove all specified rows
        
        :param A: weights tensor
        :param remove_index_list: row indexes to remove
        :return: weights
        """
        num_rows = A.size(0)
        valid_rows = [row_idx for row_idx in range(num_rows) if row_idx not in remove_index_list]
        A = A[valid_rows, :]
        return A

    def del_weight_col_less_than(self, A, l_value=1e-2, max_removal=0.1, min_size=10):
        """
        remove all columns of A that have all abs(values) below threshold l_value
        
        :param A: weights tensor
        :param l_value: threshold value
        :param max_removal: maximum fraction of cols to remove
        :param min_size: minimum number of cols to keep
        :return: tuple (weights, indexes of removed cols)
        """
        valid_cols, removed_cols = [], []
        num_cols = A.size(1)
        stop_removing = False
        for col_idx in range(num_cols):
            if len(removed_cols)>num_cols*max_removal or (num_cols-len(removed_cols))<=min_size:
                stop_removing = True
            if not torch.all(torch.abs(A[:, col_idx]) < l_value) or stop_removing:
                valid_cols.append(col_idx)
            else:
                removed_cols.append(col_idx)
        A = A[:, valid_cols]
        return A, removed_cols

    def del_weight_col_index(self, A, remove_index_list):
        """
        remove all specified columns
        
        :param A: weights tensor
        :param remove_index_list: col indexes to remove
        :return: weights
        """
        num_cols = A.size(1)
        valid_cols = [col_idx for col_idx in range(num_cols) if col_idx not in remove_index_list]
        A = A[:, valid_cols]
        return A

    def del_bias_index(self, B, remove_index_list):
        """
        remove all specified bias values
        
        :param A: bias vector
        :param remove_index_list: bias indexes to remove
        :return: bias
        """
        num_rows = B.size(0)
        valid_rows = [row_idx for row_idx in range(num_rows) if row_idx not in remove_index_list]
        B = B[valid_rows]
        return B

    def is_all_tensor_less_than(self, A, l_value=1e-2):
        """
        check if all tensor values are smaller than l_value
        
        :param A: weights tensor
        :param l_value: threshold value
        :return: boolean answer
        """
        num_rows = A.size(0)
        a = True
        for row_idx in range(num_rows):
            if not torch.all(torch.abs(A[row_idx, :]) < l_value):
                a = False
        return a

    def weights_decay(self, p_decay, decay_rate, condition=True):
        """
        apply a weight decay with probability p_decay and decay_rate

        :param p_decay: probability of decay
        :param decay_rate: decay value
        :param condition: if condition to execute the method
        :return: None
        """
        if condition and np.random.random()<p_decay:
            for i_l, l in enumerate(self.layers):
                l.weight.data *= (1-decay_rate)

    def weights_drain(self, p_drain, r_value=0.45, condition=True):
        """
        apply a selective (hardcoded for now) non linear weight decay with probability p_drain in the range [-r_value; r_value] 

        :param p_drain: probability of decay
        :param r_value: establish the range where the decay is applied [-r_value; r_value]
        :param condition: if condition to execute the method
        :return: None
        """
        if condition and np.random.random()<p_drain:
            for i_l, l in enumerate(self.layers):
                if not l.exclude_from_drain and not self.is_all_tensor_less_than(l.weight.data, 0.001):
                    #l.weight.data = torch.where(abs(l.weight.data)<=r_value, l.weight.data*(0.2*abs(l.weight.data) + 0.9), l.weight.data)
                    l.weight.data = torch.where(abs(l.weight.data)<=r_value, l.weight.data*(0.1*torch.log(2*abs(l.weight.data)+0.1) + 1.0), l.weight.data)

    @torch.no_grad()
    def weights_remove(self, p_remove=0.1, less_value=5e-3, max_removal=0.1, min_size=10, condition=True, verbose=False):
        """
        check if it is possible to remove rows or columns from all layers with probability p_remove

        :param p_remove: probability to test if some row or column can be removed
        :param less_value: threshold value for which if all the values of a row or column are below it, the row or column is removed
        :param max_removal: maximum fraction of row or column indexes that can be removed at a time
        :param min_size: minimum value of row or column indexes that must be kept
        :param condition: if condition to execute the method
        :param verbose: if true print the indexes of the removed rows and columns
        :return: True if rows or columns have been removed
        """        
        ret = False
        if condition and np.random.random()<p_remove:
            layers = self.layers
            for i_l, l in enumerate(layers):
                if l.simplify_row:
                    w_data, removed = self.del_weight_row_less_than(layers[i_l].weight.data, l_value=less_value, max_removal=max_removal, min_size=min_size)
                    if removed != []:
                        if verbose: print('removed rows from hidden layer {} -> {}'.format(i_l, removed))
                        layers[i_l].weight = nn.Parameter(w_data)
                        if layers[i_l].bias is not None:
                            layers[i_l].bias = nn.Parameter(self.del_bias_index(layers[i_l].bias.data, remove_index_list=removed))                        
                        layers[i_l+1].weight = nn.Parameter(self.del_weight_col_index(layers[i_l+1].weight.data, remove_index_list=removed))
                        layers[i_l].out_features -= len(removed)
                        layers[i_l+1].in_features -= len(removed)
                        ret = True

                if l.simplify_col:
                    w_data, removed = self.del_weight_col_less_than(layers[i_l].weight.data, l_value=less_value, max_removal=max_removal, min_size=min_size)
                    if removed != []:
                        if verbose: print('removed cols from hidden layer {} -> {}'.format(i_l, removed))
                        layers[i_l].weight = nn.Parameter(w_data)
                        layers[i_l-1].weight = nn.Parameter(self.del_weight_row_index(layers[i_l-1].weight.data, remove_index_list=removed))
                        if layers[i_l-1].bias is not None:
                            layers[i_l-1].bias = nn.Parameter(self.del_bias_index(layers[i_l-1].bias.data, remove_index_list=removed))
                        layers[i_l-1].out_features -= len(removed)
                        layers[i_l].in_features -= len(removed)
                        ret = True

        return ret
