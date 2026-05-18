import torch
from torch import nn
import math
from tqdm import tqdm

torch.set_printoptions(linewidth=200, threshold=10_000)


def affine_forward(x, W, b):
    """ 
    Implements the forward propagation step of a linear layer.

    Calculates the ouptut of a linear layer by performing a matrix 
    multiplication and bias addition.

    Args:
        x (torch.tensor): Input tensor of shape (N, c_in), 
            where N is the batch size and d is the input dimension.
        W (torch.tensor): Weight matrix of shape (c_out, c_in), 
            where c is the output dimension.
        b (torch.tensor): Bias vector of shape (c_out).

    Returns:
        tuple: A tuple containing:
            * torch.tensor: Output of the layer of shape (N, c_out).
            * tuple: A cache of all values necessary for backpropagation.
    """

    ##########################################################################
    # TODO: Implement the forward pass for the affine layer (W*x + b).       #
    #       Consider using torch.einsum to handle the batch dimension.       #
    #       Think about which values you'll need to cache for                #
    #       backward propagation.                                            #
    ##########################################################################

    out = torch.matmul(x, torch.transpose(W, dim0=0, dim1=1)) + b
    cache = x, W

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return out, cache


def affine_backward(dout, cache):
    """ Computes the backward pass of a linear layer.

    Args:
        dout (torch.tensor): The upstream gradient of shape (N, c_out).
        cache (tuple): Values for backpropagation, cached during 
            the forward pass.

    Returns:
        tuple: A tuple containing:
            * dx (torch.tensor): Gradient of the input, shape (N, c_in).
            * dW (torch.tensor): Gradient of the weights, shape (c_out, c_in).
            * db (torch.tensor): Gradient of the bias, shape (c_out).
    """

    ##########################################################################
    # TODO: Implement the backward pass for the affine layer. Using          #
    #       torch.einsum can simplify the calculations. Pay close attention  #
    #       to the input and output shapes to guide your implementation.     #
    ##########################################################################
    N, c_out = dout.shape
    N, c_in = cache[0].shape

    dx = torch.matmul(input=dout.reshape(N, 1, c_out), other=cache[1])
    dx = torch.squeeze(dx)
    dW = torch.matmul(input=dout.reshape(N, c_out, 1), other=cache[0].reshape(N, 1, c_in))
    dW = torch.sum(input=dW, dim=0)
    db = dout
    db = torch.sum(input=db, dim=0)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return dx, dW, db


def relu_forward(x):
    """
    Computes the element-wise, out-of-place ReLU function.

    Args:
        x (torch.tensor): Input tensor of any shape.

    Returns:
        tuple: A tuple containing:
            * torch.tensor: Output tensor of the same shape as 'x'
            * tuple: All values needed for backpropagation through the layer.
    """

    ##########################################################################
    # TODO: Implement the forward pass for the ReLU layer (max(0, x)).       #
    #       Think about which values you'll need to cache for                #
    #       backward propagation.                                            #
    ##########################################################################

    out = torch.nn.ReLU()(x)
    cache = x

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass through the ReLU function.

    Args:
        dout (torch.tensor): Upstream gradient of the same shape 
            as the sigmoid output.
        cache (tuple): Values for backpropagation, cached 
            during the forward path.

    Returns:
        torch.tensor: Gradient of the input, same shape as 'dout'.
    """

    ##########################################################################
    # TODO: Implement the backward pass for the ReLU layer.                  #
    #       Remember that the function is equal to f(x) = x for positive     #
    #       input and f(x) = 0 for negative input.                           #
    #       We will assume a slope of 0 for the kink at x=0.                 #
    ##########################################################################
    mask = (cache > 0).to(torch.float)
    dx = dout * mask

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return dx


def sigmoid_forward(x):
    """ Computes the element-wise sigmoid function.

    Args:
        x (torch.tensor): Input tensor of any shape.

    Returns:
        tuple: A tuple containing:
            * torch.tensor: Output tensor of the same shape as 'x'
            * tuple: All values needed for backpropagation through the layer.
    """

    ##########################################################################
    # TODO: Implement the forward pass for the sigmoid layer.                #
    #       Think about which values you'll need to cache for                #
    #       backward propagation.                                            #
    ##########################################################################

    out = torch.nn.Sigmoid()(x)
    cache = out

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return out, cache


def sigmoid_backward(dout, cache):
    """Computes the backward pass through the sigmoid function.

    Args:
        dout (torch.tensor): Upstream gradient of the same shape as the 
            sigmoid output.
        cache (tuple): Values for backpropagation, cached during the 
            forward path.

    Returns:
        torch.tensor: Gradient of the input, same shape as 'dout'.
    """

    dx = dout * cache * (1 - cache)

    ##########################################################################
    # TODO: Implement the backward pass for the sigmoid layer.               #
    #       For this, make use of the property                               #
    #       sigmoid'(x) = sigmoid(x) * (1-sigmoid(x))                        #
    #       of the sigmoid function.                                         #
    ##########################################################################

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return dx


def l2_loss(y, y_hat):
    """
    Computes the L2-loss. 

    Args:
        y (torch.tensor): Inferred output, shape (N, num_classes).
        y_hat (torch.tensor): Ground-truth labels, shape (N,).

    Returns:
        tuple: A tuple containing the following values:
            * float: The L2-loss of the inference.
            * torch.tensor: The gradient of y, shape (N, num_classes).
    """
    N, num_classes = y.shape
    y_labels = torch.nn.functional.one_hot(y_hat, num_classes=num_classes)
    loss = torch.sum((y - y_labels) ** 2) / N
    dy = 2 * (y - y_labels) / N

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return loss, dy


def calculate_accuracy(model, input_data, labels):
    """
    Calculates the mean accuracy of the model on the input data.

    Args:
        model: The model that is being tested.
        input_data (torch.tensor): Input tensor of shape (N, c_in)
        labels (torch.tensor): Class labels of shape (N,)

    Returns:
        float: The mean accuracy.
    """

    output = model.forward(input_data)[0]
    predicted_classes = torch.argmax(output, dim=-1)
    agreements = (labels == predicted_classes).to(torch.float)
    accuracy = agreements.mean()

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return accuracy


class TwoLayerNet:

    def __init__(self, inp_dim, hidden_dim, out_dim):
        """
        Initialize a new feed-forward network with two layers.

        Args:
            inp_dim (int): Size of the input.
            hidden_dim (int): Size of the hidden dimension.
            out_dim (int): Size of the output.
        """

        """
        For the classification, we will build a two-layer feed-forward neural network. The architecture will be the following:

        (N, d) - Affine - (N, d_hidden) - ReLu - Affine - (N, num_classes) - Sigmoid
        
        The bracketed expressions note the shape of the inputs and outputs to the affine layers. 
        The last layer is a sigmoid layer, so that the outputs are normalized between 0 and 1.
        
        The two affine layers require parameters (weights and biases). 
        Implement the __init__ method in the class TwoLayerNet and initialize the weights and biases as following:
        
        Initialize both biases with zeros.
        Initialize the weight of the first layer normally distributed with std sqrt(2 / c_in)
        Initialize the weight of the second layer normally distributed with std sqrt(2 / (c_in + c_out))
        where c_in and c_out are the input and output dimensions of the layers respectively.
        
        The first variant is called He initialization and is mostly used for layers with ReLU activation,
        the second is called Xavier Normal and is primarly used for sigmoid and tanh activations.
        This choice of initialization tries to keep the variance of the output similar to the variance of the input.
        Proper initializations are crucial to avoid gradient explosion or gradient vanishing.
        This is particularly important for training deeper networks, and the effect can be mitigated by introducing batch normalization or layer normalization.
        
        However, we don't want to focus too much on initialization right.
        Implement the __init__ method using the values above and test your code with the following cell.
        Name your weights 'W1', 'b1', 'W2', and 'b2'.
        """

        self.grads = dict()

        first_weight = torch.randn((hidden_dim, inp_dim))
        first_weight = torch.sqrt(torch.tensor(2 / inp_dim)) * (first_weight / torch.std(first_weight, dim=None))

        first_bias = torch.zeros(hidden_dim)

        second_weight = torch.randn((out_dim, hidden_dim))
        second_weight = torch.sqrt(torch.tensor(2 / (out_dim + hidden_dim))) * (
                second_weight / torch.std(second_weight, dim=None))

        second_bias = torch.zeros(out_dim)

        self.params = {
            "W1": first_weight,
            "b1": first_bias,
            "W2": second_weight,
            "b2": second_bias,
        }

    def forward(self, x):
        """
        Computes the forward pass through the model.

        Args:
            x (torch.tensor): Input of shape (N, d).

        Returns:
            A tuple with consiting of the following entries:
                * The output of the model.
                * A cache of all the values necessary for backpropagation.
        """

        cache = {}

        output, cache["W1"] = affine_forward(x=x, W=self.params["W1"], b=self.params["b1"])
        output, cache["relu"] = relu_forward(x=output)

        output, cache["W2"] = affine_forward(x=output, W=self.params["W2"], b=self.params["b2"])
        output, cache["sigmoid"] = sigmoid_forward(x=output)

        out = output
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out, cache

    def backward(self, dout, cache):
        """
        Computes the backward pass through the model.

        Args:
            dout (torch.tensor): Gradient of the model's output, shape (N, c_out)
            cache (tuple): A tuple consisting of the cached values for backprop.
        """

        # Todo cache to be updated
        dx = sigmoid_backward(dout=dout, cache=cache["sigmoid"])

        # todo cache to be updated
        dx, self.grads["W2"], self.grads["b2"] = affine_backward(dout=dx, cache=cache["W2"])

        # todo cache to be update
        dx = relu_backward(dout=dx, cache=cache["relu"])

        # todo cache to be updated
        dx, self.grads["W1"], self.grads["b1"] = affine_backward(dout=dx, cache=cache["W1"])


def train_model(model, train_data, train_labels, validation_data, validation_labels, n_epochs, learning_rate=1e-3,
                batch_size=16):
    """
    Optimize a model using gradient descent.

    Args:
        model: The model to optimize. 
        train_data (torch.tensor): Tensor of shape (N, d) 
            containing the data for training.
        train_labels (torch.tensor): Tensor of shape (N,) 
            containing the labels for trainng.
        validation_data (torch.tensor): Tensor of shape (N, d) 
            containing the data for validation.
        validation_labels (torch.tensor): Tensor of shape (N,) 
            containing the labels for validatiovalidation
        n_epochs (int): The number of epochs.
        learning_rate (float, optional): The learning rate 
            for the optimization. Defaults to 1e-3.
        batch_size (int, optional): The batch size 
            for the optimization. Defaults to 16.
    """

    N, d = train_data.shape

    train_batch_image_data = torch.split(train_data, split_size_or_sections=batch_size)
    train_batch_label_data = torch.split(train_labels, split_size_or_sections=batch_size)

    for n_epoch in range(n_epochs):

        for index, (image_batch, label_batch) in tqdm(enumerate(zip(train_batch_image_data, train_batch_label_data)),
                                                      desc=f"Going Through Epoch {n_epoch}"):
            # Run Forward Pass
            forward_output, forward_cache = model.forward(image_batch)

            model_loss, loss_derivative = l2_loss(y=forward_output, y_hat=label_batch)

            # Get Gradients
            model.backward(dout=loss_derivative, cache=forward_cache)

            for key, value in model.grads.items():
                model.params[key] -= learning_rate * value

        train_model_accurary = calculate_accuracy(model, image_batch, label_batch)
        model_accuracy = calculate_accuracy(model, validation_data, validation_labels)

        print(100 * '-')
        print(f"Epoch {n_epoch}")
        print(f"- Model Loss : {model_loss:.4f}")
        print(f"- Train Model Accuracy : {100 * train_model_accurary:.2f}")
        print(f"- Validation Model Accuracy : {100 * model_accuracy:.2f}")
        print(100 * '-')
    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################
