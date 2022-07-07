# nnDrain
A heuristic method to simplify neural networks.

This repository contains a library and a series of examples for structural pruning of neural networks.
Structural pruning means that neurons are removed from the network architecture and this occurs when their contribution (all incoming weights or all outgoing weights) is induced to be almost zero.
Biases are not currently considered.

An example of a neural network to solve the logical XOR operation. The network is over-parameterized for demonstration purposes.

start hidden layers: input 2 -> [150, 100, 50, 20] -> 2

final hidden layers: input 2 -> [3, 2, 2, 2] -> 2


![nnDrain](xor_simplify_weights_plot.gif)


Below is another example to solve MNIST, always with a fully connected neural network



![nnDrain](mnist_fc_simplify_weights_plot.gif)


Notebooks:
- `simplify_xor.ipynb` XOR Toy Example
- `simplify_MNIST_fc.ipynb` Fully connected NN to solve MNIST
- `simplify_MNIST_conv.ipynb` ConvNet to solve MNIST



### Usage

The way of inducing simplification is quite empirical. 
At each iteration, a decay with probability p_decay is applied to the weights. 

Similarly we apply a weight drain with probability p_drain. By 'drain the weights' we mean a selective decay applied in an interval [-r; r]. I found it more effective to use a higher decay as the weights are moving towards zero

At each epoch, the pruning or simplification of the net is carried out where the weights make a contribution almost equal to zero

```python

from tensor_edit import TensorEdit
from simplify_linear import SimplifyLinear

# dataset
train_dataset = ...
train_loader = ...

# construct a model with SimplifyLinear Modules
model = Net(...)

# select the layers that can be simplified from the model 
simplify_layers = [module for module in model.modules() if isinstance(module, SimplifyLinear)]
te = TensorEdit(simplify_layers)

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        # the force pulling down the weights in the range [-r_value; r_value] to zero
        te.weights_drain(p_drain=0.005, r_value=0.45, condition=epoch!=0 and epoch%5==0)
        # the following decay avoids getting stuck in the simplification process.
        te.weights_decay(p_decay=0.01, decay_rate=5e-3, condition=epoch!=0 and epoch%2==0)

    # simplify the net
    # remove weights if all values ​​in a row or column are less than the specified value
    if te.weights_remove(p_remove=1, less_value=0.05, max_removal=0.1, min_size=20, verbose=True):
        # re-instantiate the optimizer with the new model if I have deleted any rows or columns
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

```



