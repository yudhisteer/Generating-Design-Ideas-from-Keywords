# Generating Design Ideas from Keywords

## Abstract
## Research Questions
## Datasets
## Methods
## Plan of Action
1. Understanding a Basic GAN
    - Generative models
    - Discriminator
    - Generator
    - Cross Entropy Cost Function

2. Deep Convolutional GANs

    - Activation Functions
    - Batch Normalization
    - Convolutions

3. Wasserstein GANs with Gradient Penalty

    - ddd

4. Conditional GAN and Controllable Generation


### 1. Understanding a Basic GAN
If you got a crush on any of those people below then I got a bad news for you. These people are not real! The images have actually been downloade from the website [thispersondoesnotexist.com](https://this-person-does-not-exist.com/en). It is hard to believe that an AI can generate such realistic fake images of a person in matter of seconds but that is the reality in which we are actually living. This AI face generator is powered by ```StyleGAN```, a neural network from NVIDIA developed in 2018.

**Fun fact:** The main goal was to train the AI to recognize fake faces and faces in general. The company needed this to improve the performance of its video cards by automatically recognizing faces and applying other rendering algorithms to them. However, since the StyleGAN code is publicly available, an engineer at Uber was able to take it and create a random face generator that rocked the internet!

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158312102-b4e058ba-6e4b-44cf-b0e4-3d6322b4dd60.png" />
</p>

Another interesting application of GAN is ```Deepfake```. Haven't you ever wondered what the movie ```American Psycho``` would look like if we had ```Tom Cruise``` as the protagonist instead of ```Christian Bale```. Here's a preview:


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158317414-0a8ac5e5-fb99-479a-bb6b-2e8c1c9ddc41.gif" />
</p>



Pretty great, right? Now the most important question we must ask is: **How to recognize the fake from the real?** It is almost impossible to recognise an image of a fake person. AI is so developed that 90% of fakes are not recognized by an ordinary person and 50% are not recognized by an experienced photographer. However, occasionally a neural network makes mistakes, which is why artifacts appear: an incorrectly bent pattern, a strange hair color, and so on.

[whichfaceisreal.com](https://www.whichfaceisreal.com/) has been developed by Jevin West and Carl Bergstrom at the University of Washington, as part of the Calling Bullshit project, which focus on teaching people to be more analytical of potentially false portraits. I tested it and it is not that straightforward!

#### 1.1 Types of AI

I would now like to take a step  back and consider fundamentally what is the ```type``` of learning that can occur when we are training  neural networks to perform tasks such as shown above. 

##### 1.1.1 Supervised Learning
Supervised learning problems are instances in which we are given a set of ```data``` and a set of ```labels``` associated with that data and our goal is to learn a functional ```mapping``` that  moves from data to labels and those labels. And these labels can take many different types of forms. We will take examples of supervised learning relating to images.


1. **Classification:** our input is an image and we want to output Y, a class label for the category.

2. **Object detection:** our input is still an image but here we want to output the bounding boxes of instances of up to multiple dogs or cats.

3. **Semantic segmentation**: we have a label for every pixel the category that every pixel belongs to.

4.  **Image captioning:** our label is now a sentence and so it's now in the form of natural language.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158325846-a909b56e-53ea-4c3f-92ea-3150d233a15f.png" width="500" height="500"/>
</p>

##### 1.1.1 Unsupervised Learning
In unsupervised learning we're given only data **no** labels and our goal is to understand or build up a representation of the hidden and underlying structure in that data to extract insights into the foundational structure of the data itself.

1. **Clustering:**  the goal is to find groups within the data that are similar through some type of metric. 

2. **Dimensionality Reduction:** we start off with data in three dimensions and we're going to find two axes of variation in this case and reduce our data projected down to 2D.

3. **Feature Extraction:** with autoencoders we are trying to reconstruct the input data to basically learn features. So we're learning a feature representation without
using any additional external labels. 

5. **Density Estimation:**  we're trying to estimate and model this density. We want to fit a model such that the density is higher where there's more points concentrated.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158328574-edf605f7-2e85-4919-8c61-86fdd9fb4331.png" width="500" height="500"/>
</p>

To summarize, in ```supervised``` learning we want to use ```label data``` to learn a function mapping from ```X to Y``` and in ```unsupervised``` learning we use ```no labels``` and instead we try to learn some ```underlying hidden structure``` of the data.


#### 1.2 Generative Models vs Discriminative Models
In ```generative models``` which is a class of models for ```unsupervised learning``` where given training data our goal is to try and generate ```new samples``` from the same distribution. We have training data over generated from some distribution <img src="https://latex.codecogs.com/svg.image?P_{data}" title="https://latex.codecogs.com/svg.image?P_{data}" /> and we want to learn a model, <img src="https://latex.codecogs.com/svg.image?P_{model}" title="https://latex.codecogs.com/svg.image?P_{model}" /> to generate samples from the same distribution and so we want to learn <img src="https://latex.codecogs.com/svg.image?P_{model}" title="https://latex.codecogs.com/svg.image?P_{model}" /> to be similar to <img src="https://latex.codecogs.com/svg.image?P_{data}" title="https://latex.codecogs.com/svg.image?P_{data}" />. Hence, it has the capability of creating data similar to the training data it received since it has learnt the distribution from which the data is provided. 

We can use generative models to do ```explicit density estimation``` where we're going to explicitly define and solve for our <img src="https://latex.codecogs.com/svg.image?P_{model}" title="https://latex.codecogs.com/svg.image?P_{model}" /> or we can also do ```implicit density estimation``` where in this case we'll learn a model that can produce samples from <img src="https://latex.codecogs.com/svg.image?P_{model}" title="https://latex.codecogs.com/svg.image?P_{model}" /> without explicitly defining it.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158363720-4a6f790c-39c1-4b0c-9c6b-3dec9d86beb7.png" width="500" height="200"/>
</p>


 Our generative model takes in a ```noise``` which represents a random set of values going into the generative model. The generative model can also sometimes takes in a class ```Y``` such as a dog.  From these inputs, it's goal is to generate a set of features ```X```(wet nose or a tongue sticking out. ) that look like a realistic dog.  But why do we need this noise in the first place? The noise is here to ensure that what is generated isn't actually the same image each time. Else, what is the point of generating the same image agai nand again. As explained above, gnerative models try to capture the probability distribution of ```X```, the different features of having a wet nose, the tongue sticking out, maybe pointy ears sometimes but not all the time, given that class ```Y``` of a dog. With the added noise, these models would generate realistic and diverse representations of this class ```Y```. Note: if we are only generating one class Y of a dog, then we don't need this conditioning on Y - <img src="https://latex.codecogs.com/png.image?\dpi{110}P(X|Y)" title="https://latex.codecogs.com/png.image?\dpi{110}P(X|Y)" /> - and instead it's just the probability over all the features X - <img src="https://latex.codecogs.com/png.image?\dpi{110}P(X)" title="https://latex.codecogs.com/png.image?\dpi{110}P(X)" />. If we continue to run our model multiple times without any restrictions, then we'll end up getting more pictures representing the dataset our generative model was trained on. 
 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158368054-8ca5bfd1-b41d-4f61-97e3-a71e4555718e.png" />
</p>

There are many types of generative models. The most popular ones are ```Variational Autoencoders (VAE)``` or ```GANs```.

##### 1.2.1 Autoencoders
Variational Autoencoders are related to a type of unsupervised learning model called ```autoencoders```. With autoencoders we don't generate data, but it's an unsupervised approach for learning a ```lower dimensional``` feature representation from unlabeled training data. We feed in as input raw data for example an image that's going to be  passed through many successive deep neural network layers. At the output of that succession  of neural network layers we are going to generate a low dimensional latent space - a ```feature representation```. We call this portion of the network an ```encoder``` since it's mapping the data ```x``` into a encoded  vector of latent variables ```z```.

**Note:** It is important to ensure the low dimensionality of this latent space ```z``` so that we are able to compress the data into a small latent vector where we can learn a very compact and rich feature representation. We want to learn features that can capture meaningful factors of variation in the data.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158852280-5d3d1777-fbe8-45f6-8e8f-a0d708105409.png" width="250" height="200"/>
</p>

To train such a model we need to learn a decoder network that will actually reconstruct the original image. Again for the decoder we are basically using same types of networks as encoders so it's usually a little bit symmetric. We call our reconstructed output <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{x}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{x}" />  because it's our prediction and it's an imperfect reconstruction of our input ```x``` and the way that we can actually train this network is by looking at the original input ```x``` and our reconstructed output  <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{x}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{x}" /> and simply comparing the two and minimizing the distance between these two images using ```L2 loss function```.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158853576-5e4c7943-9fa3-413d-bedf-adff62825924.png" width="600" height="300"/>
</p>

**Note:** Notice that by using this reconstruction loss - the difference between the reconstructed output and our original input - we do not require any labels for our data beyond the data itself. It is just using the raw data to supervise itself.

In practice, the  lower the dimensionality of our latent space, the poorer and worse quality reconstruction we're going to get out. These autoencoder structures use this sort of bottlenecking hidden layer to learn a compressed  latent representation of the data and we can  self-supervise the training of this network by using a reconstruction loss that forces the autoencoder network to encode as much information about the data as possible into a lower dimensional latent space while still being able to build up faithful reconstructions.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158857249-767cc6fd-f3c2-42f4-be5b-f3f19a32c7a0.png" width="500" height="200"/>
</p>




To sum up, we are going to take our input data, pass it through our **encoder** first` which can be like a three layers **convolutional** network, to get these features and then we're going to pass it through a **decoder** which is a three layers of **upconvolutionalnetwork** and then get a reconstructed data out at the end of this. The reason why we have a convolutional network for the encoder and an upconvolutional network for the decoder is because at the encoder we're basically taking it from this high dimensional input to these lower dimensional features and now we want to go the other way go from our low dimensional features back out to our high dimensional reconstructed input.

##### 1.2.2 Variational Autoencoders
In autoencoders, this latent layer is just a normal layer in a neural network just like any other layer.  It is ```deterministic```. If we're going to feed in a particular input to this network we're going  to get the same output so long as the weights are the same. Therefore, effectively a traditional autoencoder learns this deterministic encoding which allows for reconstruction and reproduction of the input.

Variational auto encoders impose a ```stochastic``` or variational twist on this architecture. The idea behind doing so is to generate smoother representations of the input data and improve the quality of not only the reconstructions but also to actually generate new images that are similar to the input data set  but not direct reconstructions of the input data. Variational autoencoders replace that deterministic layer ```z``` with a stochastic sampling operation. Instead of learning the  latent variables ```z``` directly for each variable, the variational autoencoder learns a ```mean``` and  a ```variance``` associated with that latent variable. And the mean and variance parameterize a ```probability distribution``` for that latent variable. We can actually generate new data instances by ```sampling``` from the distribution defined by these <img src="https://latex.codecogs.com/svg.image?\mu&space;_{s}" title="https://latex.codecogs.com/svg.image?\mu _{s}" /> and <img src="https://latex.codecogs.com/svg.image?\sigma&space;_{s}" title="https://latex.codecogs.com/svg.image?\sigma _{s}" /> to generate a latent sample and get  probabilistic representations of the latent space.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/158946396-0caa3e82-da4d-42cd-9615-c1a6206ded2c.png" width="600" height="300"/>
</p>


Our encoder is now going to be trying to learn a probability distribution of the latent space ```z``` given the input data ```x``` while the decoder is going to take that learned latent representation and compute a new probability distribution of the input ```x``` given that latent distribution ```z``` and these networks - the encoder the decoder - are going to be  defined by separate sets of weights <img src="https://latex.codecogs.com/svg.image?\phi&space;" title="https://latex.codecogs.com/svg.image?\phi " /> and <img src="https://latex.codecogs.com/svg.image?\theta&space;" title="https://latex.codecogs.com/svg.image?\theta " /> and the way that we can train this variational autoencoder is by defining a loss function that's going to be  a function of the data ```x``` as well as these sets of weights <img src="https://latex.codecogs.com/svg.image?\phi&space;" title="https://latex.codecogs.com/svg.image?\phi " /> and <img src="https://latex.codecogs.com/svg.image?\theta&space;" title="https://latex.codecogs.com/svg.image?\theta " />. The reconstruction loss just as before will force  the latent space to learn and represent faithful representations of the input data  ultimately resulting in faithful reconstructions.

<!--- However, when introducing this stochastic sampling layer we now have a problem where we can't back propagate gradients through a sampling layer. ```Backpropagation``` requires deterministic nodes ```deterministic``` layers for which we can iteratively apply the chain rule to optimize gradients  optimize the loss via gradient descent. --->

**To sum up:** In Variational Autoencoders we inject some ```noise``` into this whole model and training process. Instead of having the encoder encode the image into a single point in that latent space, the encoder actually encodes the image onto a whole distribution and then samples a point on that distribution to feed into the decoder to then produce a realistic image. This adds a little bit of noise since different points can be sampled on this distribution. 


##### 1.2.3 Discriminative Models


##### 1.2.4 GANs
Another instance of generative models is GANs where we don't want to explicitly model the density or the distribution underlying some data but instead just learn a ```representation``` that can be successful in generating new instances that are similar to the data. What we care about is to be able to sample from a ```complex high dimensional training distribution```. However, there's no direct way that we can do this. Therefore, we're going to have to build up some approximation of this distribution, i.e, we sample from simpler distributions. For example ```random noise```. We're going to learn a transformation from these simple distributions
directly to the training distribution that we want. And to model this kind of complex function or transformation we will use a ```neural network```. To sum up, we start from something extremely simple: random noise and try to build a generative neural network that can learn a functional transformation that goes from noise to the data distribution and by learning this functional generative mapping we can then sample in order to generate fake instances synthetic instances that are going to be as close to the real data distribution.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159155369-c100201b-d2d9-4d14-a411-8a6a2618bbb9.png" width="500" height="175"/>
</p>

GANs are composed of two neural networks models: a ```generator``` which generates images like the ```decoder``` and a ```discriminator``` that's actually a ```discriminative``` model hidden inside of it. The generator and discriminator are effectively competing against each other which is why they're called ```adversarial```. The generator ```G``` is going to be trained to go from random noise to produce an imitation of the data and then the discriminator is going to take that ```synthetic fake data``` as well as ```real data``` and be trained to actually **distinguish between fake and real**. If our generator network is able to generate well and generate fake images that can successfully ```fool``` this discriminator, then we have a good generative model. This means that we're generating images that look like images from the training set. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159155621-99466a29-eb02-476d-9d9e-0a407ec28ba7.png" width="500" height="350"/>
</p>

With time we reach a point where we don't need the discriminator anyemore. The generator can take in any random noise and produce a realistic image. Note that The generator's role in some sense it's very similar to the decoder in the ```VAE```. What's different is that there's no guiding encoder this time that determines what noise vector should look like, that's input into the generator. Instead, there's a discriminator looking at fake and real images and simultaneously trying to figure out which ones are real and which ones are fake. Overall the effect is that the discriminator is going to get better and better at learning how to classify real and fake data and the better it becomes at doing that it's going to force the generator to try to produce better and better synthetic data to  try to fool the discriminator and so on.



# Conclusion

# References
1. https://www.youtube.com/watch?v=xkqflKC64IM&t=489s
2. https://www.youtube.com/watch?v=CDMVaQOvtxU
3. https://www.whichfaceisreal.com/
4. https://this-person-does-not-exist.com/en
5. 
6. 
