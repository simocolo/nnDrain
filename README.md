# nnDrain
Structural pruning is a method for simplifying neural networks by removing neurons that do not contribute significantly to the network's function. Pruning can lead to faster training times, improved performance, reduced overfitting, and smaller model sizes without sacrificing accuracy. A simple approach for structural pruning is presented that targets neurons with almost zero incoming or outgoing weights, which are induced to have zero weight values through the use of a decay function.

The effectiveness of this approach is demonstrated through a series of Jupyter notebooks:

- `simplify_xor.ipynb` XOR Toy Example
- `simplify_MNIST_fc.ipynb` Fully connected NN to solve MNIST
- `simplify_MNIST_conv.ipynb` ConvNet to solve MNIST
- `adder.ipynb` and `chargpt.ipynb` added simplification to Andrej Karpathy's minGPT examples ([karpathy/minGPT](https://github.com/karpathy/minGPT))

I encourage you to run the code and experiment with different pruning parameters to see the impact on the network's performance.

XOR

![nnDrain](out/xor/xor.gif)


MINST - FCNN

![nnDrain](out/mnist/fc/mnist_fc.gif)


minGPT model for the addition problem

![nnDrain](out/adder/adder.gif)




### Usage

In this approach to structural pruning, at each iteration, a decay probability of p_decay is applied to the weights.

Additionally, a weight drain probability of p_drain is used to selectively decay weights in a small range [-r; r]. 
This approach is found to be more effective as it further pushes the weights towards zero. 

At the end of each training epoch, the pruning or simplification of the network is conducted by removing neurons whose weights have a negligible contribution to the network's function.

```python

from nndrain.tensor_edit import TensorEdit
from nndrain.simplify_linear import SimplifyLinear

# dataset
train_dataset = ...
train_loader = ...

# construct a model with SimplifyLinear Modules
model = Net(...)

# select the layers that can be simplified from the model 
simplify_layers = [module for module in model.modules() if isinstance(module, SimplifyLinear)]
te = TensorEdit(simplify_layers)
drain_threshold_coeff = 3.0
remove_threshold_coeff = 0.95

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        # apply decays
        te.weights_drain(p_drain=0.5, threshold_coeff=drain_threshold_coeff)
        te.weights_decay(p_decay=0.5, decay_rate=5e-3)

    # simplify the net
    # remove neurons if all incoming or outgoing weights fall below a specified threshold
    if te.weights_remove(p_remove=0.5, threshold_coeff=remove_threshold_coeff, max_removal=1, verbose=True):
        # re-instantiate the optimizer with the new model if I have deleted any rows or columns
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

```



