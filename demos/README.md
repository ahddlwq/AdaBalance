# Demos

Here we provide some demos of using AdaBalance on several benchmark tasks.
The purpose of these demos is to give an example of how to use it your research, and also
illustrate the robust performance of AdaBalance.

AdaBalance is an optimizer with balance between AdaBound and AdaMod,  which recalculates learning rates 
like AdaMod and clips the bound of learning rates similar to AdaBound, thereby achieving a smooth transition 
from adaptive methods during the initial phase of training to SGD towards the end.

In most examples, you can observe that AdaBalance has a much faster training speed than SGD
in the early stage, and the learning curve is much smoother than that of SGD.
As for the final performance on unseen data, AdaBalance can achieve better or similar performance
compared with SGD, and has a considerable improvement over the adaptive methods.

## Demo List
- CIFAR-10 \[[notebook](./cifar10/visualization.ipynb)\] \[[code](./cifar10)\]

## Future Work

We will continue to update the demos with more popular benchmarks in the near future. 
If you have a specific task in mind that is not yet included, please feel free to leave an issue 
or email the first author ([Wuqi Liang](mailto:liangwuqi@ahou.edu.cn)).