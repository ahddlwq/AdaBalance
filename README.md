# AdaBalance
### Implements AdaBalance algorithm

An optimizer with balance between AdaBound and AdaMod, which recalculates learning rates like AdaMod 
and clips the bound of learning rates similar to AdaBound, for developing state-of-the-art deep learning models 
on a wide variety of popular tasks in the field of CV, NLP, and etc.

Based on  `AdaBalance: Novel Adaptive Gradient Optimization via Recalculation and Constrained Learning Rates`.

### Using source code

As AdaBalance is a Python class with only 100+ lines, an alternative way is directly downloading
[adabalance.py](./adabalance/adabalance.py) and copying it to your project.

## Usage

You can use AdaBalance just like any other PyTorch optimizers.

```python3
optimizer = adabalance.AdaSmooth(network.parameters(),lr=learning_rate,betas=(0.9, 0.999),beta3=0.999,final_lr=0.1, 
gamma=1e-3, eps=1e-8, weight_decay=0, amsbound=False)
```

As described in the paper, AdaBalance is an optimizer which recalculates learning rates like AdaMod and clips
the bound of learning rates similar to AdaBound, thereby achieving a smooth transition from adaptive methods 
during the initial phase of training to SGD towards the end. 

```python3
Parameters: 
lr (float, optional): Adam learning rate (default: 1e-3)
betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
beta3 (float, optional): smoothing coefficient for adaptive learning rates (default: 0.9999)
final_lr (float, optional): final (SGD) learning rate (default: 0.1)
gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
```

Our proposed method AdaBalance inherits the advantages of recalculating learning rates of AdaMod 
and clipping the bound of learning rates of AdaBound. Experimental findings validate the effectiveness 
of our method on multiple standard benchmarks across two key tasks: computer vision and natural language processing. 
However, the performance of an algorithm depends on many factors, including specific tasks, model structure, data distribution, 
and so on. When using our proposed algorithm, you may still need to decide which hyperparameters to use based on specific circumstances.


## Demos

Thanks to the awesome work by the GitHub team and the Jupyter team, the Jupyter notebook (`.ipynb`)
files can render directly on GitHub.
We provide several notebooks (like [this one](./demos/cifar10/visualization.ipynb)) for better visualization.
We hope to illustrate the robust performance of AdaBalance through these examples.

For the full list of demos, please refer to [this page](./demos).

## Contributors

[@ahddlwq](https://github.com/ahddlwq)

## License
[Apache 2.0](./LICENSE)
