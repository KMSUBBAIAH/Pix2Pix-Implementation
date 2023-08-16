### Pix2Pix-Implementation
* Quick Disclaimer - I was unable to achieve the expected results after training it for 3 days since my laptop crashed and my will to re-train died. </br>
#### All in, learned a couple of new things, this model is designed to classify patches of input images instead of the entire image, as real or fake. Also instead of training the generator to minimize log(1-D(x,G(x,z)), we maximize log(D(x,G(x,z)), basically maximizing discriminator log loss.
</br>

Implementing the Pix2Pix GAN (Generative Adversarial Network) for the task of generating map-oriented satellite images from street-view images involves several steps. 
The Pix2Pix model is particularly useful for image-to-image translation tasks, where you're trying to convert images from one domain to another while maintaining their 
underlying structure. In your case, you want to generate map-oriented images from street-view images.

#### config.py
It is a configuration setup for training a Pix2Pix GAN model using PyTorch and the Albumentations library. It defines various parameters and transformations:

Specifies whether to use GPU or CPU for computation.<br />
Sets directories for training and validation images.<br />
Defines learning rate, batch size, and other training parameters.<br />
Sets up image transformations for input and output images using Albumentations.<br />
Configures options for loading and saving model checkpoints.<br />
Provides variables for controlling the number of training epochs and loading pre-trained models.<br />

#### dataset.py
This custom dataset class is called MapDataset, used for handling street-view and map-oriented image pairs. It loads images from a specified directory, divides them into input 
and target images, applies transformations, and returns the preprocessed pairs. The main block demonstrates dataset loading and preprocessing by using this class, 
printing input image shapes, and saving images.

#### generator.py
This defines a Generator neural network module for image translation using the Pix2Pix architecture. The generator takes input images and aims to generate corresponding 
map-oriented images. Here's a summary of what each part of the code does:

class `Block(nn.Module)`: Defines a building block for the generator. This block includes a convolutional layer, batch normalization, and an activation function 
(ReLU or LeakyReLU). Optionally, dropout is applied.<br />
class `Generator(nn.Module)`: Defines the main Generator module. It consists of an initial downsampling layer, a series of down-sampling blocks, a bottleneck layer,
and a series of up-sampling blocks. The generator architecture follows an encoder-decoder structure:<br />
Downsampling: A sequence of down-sampling blocks (down1 to down6) reduces the spatial dimensions while increasing the number of features.
Bottleneck: A bottleneck layer further processes the encoded features.<br />
Upsampling: A sequence of up-sampling blocks (up1 to up7) reconstructs the output by progressively upsampling and combining features from the encoder.
Final Upsampling: A transposed convolution layer followed by the Tanh activation function produces the final output image.
forward(self, x): Implements the forward pass of the generator. It passes input through the different blocks and concatenates features from the encoder's downsampling 
path with those from the upsampling path.

#### discriminator.py
This defines a Discriminator neural network module for evaluating the realism of image pairs in the Pix2Pix GAN architecture. The discriminator takes paired images 
(input and target) and assesses whether they are real or generated. Here's a summary of what each part of the code does:<br />

class `CNNBlock(nn.Module)`: Defines a building block for the discriminator. This block includes a convolutional layer, batch normalization, and LeakyReLU activation.<br />
class `Discriminator(nn.Module)`: Defines the main Discriminator module. It consists of an initial convolutional layer, followed by multiple CNN blocks, and ends with
a final convolutional layer that outputs a single-channel prediction.<br />
Initial Layer: A convolutional layer with LeakyReLU activation, designed to handle concatenated input images (input and target).<br />
CNN Blocks: A series of CNN blocks (CNNBlock) with increasing feature channels and stride values, capturing hierarchical information.<br />
Final Layer: A convolutional layer that outputs a single-channel prediction.<br />
forward(self, x, y): Implements the forward pass of the discriminator. Concatenates input and target images along the channel dimension, passes through the initial 
layer and the sequence of CNN blocks, and produces the final prediction.

#### utils.py
`save_some_examples(gen, val_loader, epoch, folder)`: Generates images using the trained generator (`gen`) from validation data and saves examples in a specified folder,
adjusting pixel values.<br />
`save_checkpoint(model, optimizer, filename)`: Saves a model and its optimizer's state to a checkpoint file.<br />
`load_checkpoint(checkpoint_file, model, optimizer, lr)`: Loads a model's state and optimizer's state from a checkpoint file, adjusting the learning rate.
These functions assist in saving/loading model progress, visualizing generated images, and ensuring proper continuation of training.<br />

#### train.py
This script implements the training process of a Pix2Pix GAN model for the translation of street-view images to map-oriented images.<br />
* Import necessary modules and functions from libraries like `torch`, `torch.nn`, and custom modules.<br />
* Set up the GPU device and various configurations from the `config` module.<br />
* Define the `train_fn` function, responsible for training the GAN:
   - Loops through the training data batches.
   - Generates fake map-oriented images using the generator.
   - Calculates the discriminator's loss for real and fake images.
   - Optimizes the discriminator and generator networks using gradient scaling.
   - Periodically updates the progress displayed in the console.

* Define the `main` function:
   - Initializes the discriminator and generator models.
   - Sets up optimizers and loss functions.
   - Optionally loads pre-trained models.
   - Prepares data loaders for training and validation.
   - Initializes gradient scalers for the generator and discriminator.
   - Initiates the training loop for a specified number of epochs.
   - During each epoch, trains the model, saves checkpoints, and saves example images generated by the generator.


