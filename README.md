# CONDITIONAL VARIATIONAL AUTOENCODER FOR IMAGE MANIPULATION

## Introduction
 Conditional variational autoencoder (CVAE) is an extension of VAE developed in the field of Neural machine translation (Zhang et al. (2016)). 
 The key idea is to assume that our model is conditioned not only on the input x ∈ X, but also on an observable conditioning variable
 c, which guides the translation process. 

## Dataset and Tools used
CelebA dataset contains more than 200K celebrity images, each with 40 binary face attributes
 annotations (like Male, Smiling, eye galsses etc.). Each image has its own face attributes
 annotation, which is encoded as a 40-dimensional binary vector: 0 means that the image
 does not show the corresponding attribute, 1 means that it does. The database can be
 downloaded from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (For this project I have downloaded AlignedCropped
 Images and Attributes Annotations).

## Implementation
### Task- 1: Develop and train a CVAE to encode and manipulate images in CelebA database. 
This task focuses on image reconstruction using encoder and a decoder. It involves:

1. Image Preprocessing: Resize images to [3,64,64] shape and normalize pixel values. Add attribute annotations as a fourth channel.

2. Encoder: Use a 5-layer Conv2D model to encode images. Flatten outputs to obtain distribution parameters for latent space.

3. Latent Space Encoding: Sample latent space using reparameterization trick. Concatenate attribute vector to latent space.

4. Decoder: Reconstruct images using Transposed 2D Convolutional layers and a final Convolutional layer.

5. Loss Function: Utilize Mean Squared Error for reconstruction loss.

### Task- 2: Using the previously trained CVAE to manipulate an image by changing the attribute vector input to the encoded image
In this section, we explore the ability of our generative model to manipulate images by altering their attributes. This involves reconstructing a batch of images with new attributes. Here's how to perform this task:

1. Create a Batch of Images: Select a batch of images from the test set, ensuring it is heterogeneous to test the network's ability on inputs with different features.

2. Encode Images: Utilize the pre-trained model encoder to encode the selected images and store the resulting mean vectors.

3. Specify New Attributes: Specify the desired new attributes by changing the original labels. Examples include non-smiling to smiling, non-mustached to mustached, and non-glass wearing to glass wearing.

4. Decode Mean Vectors with New Attributes: Use the pre-trained model decoder to decode the mean vectors, combined with the new attribute labels.

### Task- 3: Slowly morph one image to another
As an additional exploration of our generative model's capabilities, we can create a smooth transition between two images by morphing one into another. This process utilizes linear interpolation in the latent space learned by the generative model. Here's how to perform this task:

1. Select Images A and B: Choose two images A and B from the dataset.

2. Extract Latent Mean Vectors: Use the pre-trained generative model to extract the latent mean vectors corresponding to images A and B.

3. Generate Interpolation Vector: Define an interpolation vector µA→B using the formula: µA→B = (1 - δ) · µA + δ · µB
where δ varies smoothly between 0 and 1.

4. Inject and Decode: Inject the resulting vectors into the generative model.

5. Decode these latent vectors to generate a set of new images that create a smooth transition between the two original images.

Example:
Let's consider transitioning from Image A (non-smiling) to Image B (smiling):
Set δ = 0, generate an image similar to A (non-smiling).
Gradually increase δ from 0 to 1, generating images that morph from A to B (smiling).

## Acknowledgments

* [Dr. Hamid Krim- NCSU](https://ece.ncsu.edu/people/ahk/)
