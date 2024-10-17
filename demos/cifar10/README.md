# Examples on CIFAR-10

In this example, we test AdaBalance on the standard CIFAR-10 image classification dataset,
comparing with several baseline methods including: SGD, RMSProp, AdaGrad, Adam, AMSGrad, AdaBound and AdaMod.

Tested with PyTorch 0.4.1.

## Visualization

We provide a notebook to make it easier to visualize the performance of AdaBalance 
You can directly click [visualization.ipynb](./visualization.ipynb) and view the result on GitHub,
or clone the project and run on your local.

## Settings

We have already provided the results produced by AdaBalance with default settings and
baseline optimizers with their best hyperparameters.
The way of searching the best settings for baseline optimizers is described in the experiment
section of the paper.

## Running by Yourself

You may also run the experiment and visualize the result by yourself.
The following is an example to train ResNet-34 using AdaBalance with a learning rate of 0.001 and
a final learning rate of 0.1.

```bash
python main.py --model=resnet --optim=adabalance --lr=0.001 --final_lr=0.1
```

The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve
will be save in the `curve` folder.
