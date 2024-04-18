# Gan-Cgan

## Description
This is a project for class MOS 3.4 "Apprentissage Automatique" at Centrale Lyon. The objective is to implement, train and test generative adversarial networks (GAN) on a given dataset.

The notebook was run on Google Colab using the T4 GPU.

## Requirements

This notebook uses Python to run.
The following dependencies are required: Numpy, matplotlib, PyTorch, torchvision, imageio, openCV and Pillow.

## Usage
This notebook can be used as a base for future projects using generative adversarial networks. It is divided into two parts, respectively focusing on DC-GAN and cGAN architectures.

### DC-GAN

The Deep Convolutional Generative Adversarial Network (DC-GAN) is a type of GAN with certain characteristic:
-the generator is comprised of convolutional-transpose and batch normalization layers with ReLU activation.
-the discrimator is comprised of convoltional and batch normalization layers with LeakyReLU activation.

The generator takes a latent vector as input and returns an image of size 32*32.
The discriminator takes an image as input and returns a value between 0 and 1.

The objective of the generator is to generate a number that is indistinguishable from those present in the MNIST dataset, while the discriminator learns to distinguish what is produced by the generator from the real images.

The training parameters and the training script are provided in the notebook, as well as the visual output of the generator for testing.
The results aren't great but the training was limited to 5 epochs.

### cGAN

The Conditional Generative Adversarial Network (cGAN) is a supervised GAN with the objective of mapping a label picture to a real one (or the opposite). In our case, the generator generates a real image from noise using a label image. The discriminator takes as input a pair of images (one real, one label) and tries to predict if the pair was generated or not. 

The generator architecture is a UNet and the discriminator architecture is that of a PatchGAN.

The model implementation, training and testing scripts are included in the notebook.

We notice that the generated images are very realistic on the training data after 200 epochs, but remain very blurry on the validation set. We can thus consider that the model has overfitted on the training data. However, the generated images differ from one another, therefore the cGAN architecture has successfully avoided model collapse.

## Authors and acknowledgment
Amaury Giard

Based on project by Emmanuel Dellandrea available at https://gitlab.ec-lyon.fr/edelland/mso_3_4-td2.

Project done under the supervision of Leo Schneider.
