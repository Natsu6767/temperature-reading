# Temperature Reading Prediction

### Model Architectures
1. **Small_CNN:** _Conv(5, 5, 64)_ -> _Conv(3, 3, 64)_ -> _Conv(3, 3, 128)_ -> _Conv(3, 3, 256)_ -> _Conv(3, 3, 512)_ -> _Conv(3, 3, 512)_ -> _Linear(8192, 101)_
                  During all convolutions, the input is zero padded to make input-output spatial dimensions same and stride is 1. 
                  Each convolution layer is followed by a Batch Normalization layer, ReLU activation and 2x2 MaxPooling layer.

2. **Large_CNN:** _Conv(5, 5, 32)_ -> _Conv(3, 3, 64)_ -> _Conv(3, 3, 128)_ -> _Conv(3, 3, 256)_ -> _Conv(3, 3, 512)_ -> _Conv(3, 3, 512)_ -> _Linear(524288, 101)_
                  During all convolutions, the input is zero padded to make input-output spatial dimensions same and stride is 1. 
                  Each convolution layer is followed by a Batch Normalization layer, ReLU activation. 2x2 MaxPooling layer is added after every 2 
                  convolution layers.

The Adam optimizer with a learning rate of 1e-3 and default values of beta is used. Batch size of 32 is used during training.

### Data Pre-Processing
Input images are resized to 256 x 256 dimensions using bilinear interpolation.

### Loss Function Used
The loss function used is the Cross Entropy Loss. The problem is framed as a 101 class (0, 1, 2, . . ., 100) classification problem.
For the **Small_CNN** model the MSE Loss is also tried. In this case the problem is framed as a regression problem.

### Metric Used
The accuracy is used as the metric for the problem. When doing classification, taking the argmax of the models predictions gives the predicted class.
During regression, the models prediction is calculated by rounding the model output to the nearest integer.

### Results

All results are averaged over 3 random seeds.

Small CNN Training Curve

![small_cnn_training_loss](\images\small_cnn/small_cnn_loss.png)

Large CNN Training Curve

![large_cnn_loss_curve](\images\large_cnn/large_cnn_loss.png)

Small CNN with MSE Training Curve

![large_cnn_loss_curve](\images\small_cnn_mse/small_cnn_mse_loss.png)

Final Accuracy on Validation Set (Averaged over 3 seeds)

|       | Small CNN | Large CNN | Small CNN MSE |
|:-----:|:---------:|:---------:|:-------------:|
|  Mean |   91.99   |   30.10   |     44.68     |
| Error |  +/- 0.32 | +/- 12.71 |   +/- 30.63   |


Final Accuracy on Test Set (Averaged over 3 seeds)
|       | Small CNN | Large CNN | Small CNN MSE |
|:-----:|:---------:|:---------:|:-------------:|
|  Mean |   91.86   |   30.63   |     43.78     |
| Error |  +/- 0.93 | +/- 13.68 |   +/- 30.08   |



### Visualizations of Results

**Small CNN**

Correct Results

![Sample 1](\images\small_cnn/34_34.png) 34/34 ![Sample 2](\images\small_cnn/47_47.png) 47/47![Sample 3](\images\small_cnn/70_70.png) 70/70

Incorrect Results

