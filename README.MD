An AI application written in Julia for analyzing sequences of images and videos (4D tensors) and labeled data like time series (1D, 2D tensors). The entire application is written with a microservice architecture.


The app involved a CNN-LSTM model where key tensor operations are:

- Convolution (Flux.Conv): This performs a 2D convolution on the input image, using a kernel (filter) of given size and number of filters (channels). This results in a 3D tensor of dimensions (H, W, C) where C is the number of filters.

- ReLU activation (relu): Applies the rectified linear unit activation function on the convolution output. This is a non-linear activation that thresholds at 0.

- Flattening (Flux.flatten): Converts the 3D convolution output into a 2D matrix by collapsing the spatial dimensions.

- LSTM (Flux.LSTM): Applies a Long Short-Term Memory recurrent neural network on the 2D flattened convolution output. This allows the network to learn long-range dependencies in the sequence.

- Dense layer (Flux.Dense): A regular fully-connected feedforward layer that converts the LSTM output into the final output dimension.

So in summary, the model first extracts high-level features from the raw input using convolutional and pooling layers. It then passes these features into an LSTM to learn the temporal dynamics. Finally, a dense layer is used to map the LSTM output to the desired output dimension.

The calls to Flux.Chain simply chain these layers together into the CNN and LSTM sub-models, while the overloaded call operator () chains the sub-models together into the full hybrid model.