# Generative AI: Generating Design Ideas from Keywords

## Abstract
## Research Questions
## Datasets
## Methods
## Plan of Action
1. Understanding a Basic GAN
    - Types of AI
    - Generative models vs Discriminative models
    - The Discriminator
    - The Generator
    - Cross Entropy Cost Function

2. Wasserstein GANs with Gradient Penalty

    - Mode Collapse
    - Limitation of BCE Loss
    - Earth Mover Distance
    - Wasserstein Loss
    - Condition on Wasserstein Critic
    - 1-Lipschitz Continuity Enforcement
    - Coding a WGAN

3. Controllable and Conditional GAN

Although we are generating very good fake images from our WGAN, we do not really have a control of the type of faces to generate. 

3. Multimodal Generation


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
A discriminative model is one typically used for ```classification``` in machine learning. They learn how to distinguish between classes such as dogs and cats, and are often called ```classifiers```. Discriminative models take a set of features ```X```, such as having a wet nose or whether it purrs and from these features determine a category of whether the image is of a dog or a cat. In other words, they try to model the probability of class ```Y``` given a set of features ```X``` - ```P(Y|X)```.

In simple words, a discriminative model makes predictions on the unseen data based on conditional probability and can be used either for classification or regression problem statements. These models are not capable of generating new data points. Therefore, the ultimate objective of discriminative models is to separate one class from another.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159219293-8d850423-d05d-4916-88d2-3934bc14a50c.png" width="600" height="250"/>
</p>


Below are examples of generative and discriminative classifiers:

**Generative classifiers**
- Naïve Bayes
- Bayesian networks
- Markov random fields
- Hidden Markov Models (HMM)

**Discriminative Classifiers**
- Logistic regression
- Support Vector Machine (SVM)
- Traditional neural networks
- K-Nearest Neighbour (KNN)
- Conditional Random Fields (CRF)

To summarise:

- Generative models model the distribution of individual classes.
- Discriminative models learn the boundaries between classes.
- With Generative models, we have less chance of overfitting if our data distribution is similar to real data distribution. However, outliers can affect our model performance.
- With Discriminative models, we can work with small dataset but we should be careful of overfitting.


##### 1.2.4 GANs
Another instance of generative models is GANs where we don't want to explicitly model the density or the distribution underlying some data but instead just learn a ```representation``` that can be successful in generating new instances that are similar to the data. What we care about is to be able to sample from a ```complex high dimensional training distribution```. However, there's no direct way that we can do this. Therefore, we're going to have to build up some approximation of this distribution, i.e, we sample from simpler distributions. For example ```random noise```. We're going to learn a transformation from these simple distributions
directly to the training distribution that we want. And to model this kind of complex function or transformation we will use a ```neural network```. To sum up, we start from something extremely simple: random noise and try to build a generative neural network that can learn a functional transformation that goes from noise to the data distribution and by learning this functional generative mapping we can then sample in order to generate fake instances synthetic instances that are going to be as close to the real data distribution.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159155369-c100201b-d2d9-4d14-a411-8a6a2618bbb9.png" width="500" height="175"/>
</p>

GANs are composed of two neural networks models: a ```generator``` which generates images like the ```decoder``` and a ```discriminator``` that's actually a ```discriminative``` model hidden inside of it. The generator and discriminator are effectively competing against each other which is why they're called ```adversarial```. The generator ```G``` is going to be trained to go from random noise to produce an imitation of the data and then the discriminator is going to take that ```synthetic fake data``` as well as ```real data``` and be trained to actually **distinguish between fake and real**. If our generator network is able to generate well and generate fake images that can successfully ```fool``` this discriminator, then we have a good generative model. This means that we're generating images that look like images from the training set. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159155884-9103aee9-e2cb-43ba-87e7-964ff839b93b.png" width="500" height="350"/>
</p>

With time we reach a point where we don't need the discriminator anyemore. The generator can take in any random noise and produce a realistic image. Note that The generator's role in some sense it's very similar to the decoder in the ```VAE```. What's different is that there's no guiding encoder this time that determines what noise vector should look like, that's input into the generator. Instead, there's a discriminator looking at fake and real images and simultaneously trying to figure out which ones are real and which ones are fake. Overall the effect is that the discriminator is going to get better and better at learning how to classify real and fake data and the better it becomes at doing that it's going to force the generator to try to produce better and better synthetic data to  try to fool the discriminator and so on.

##### 1.2.5 Intuition behind GANs
As exaplained above, the generator learns to generate fakes that look real, to fool the discriminator. And the discriminator learns to distinguish between what's real and what's fake. So you can think of the generator as a painting forger and the discriminator as an art inspector. So the generator forges fake images to try to look as realistic as possible, and it does this in the hopes of fooling the discriminator. So we can think of the generator as a painting forger and the discriminator as an art inspector. So the generator forges fake images to try to look as realistic as possible, and it does this in the hopes of fooling the discriminator.

The video below really depicts how a GAN works. ```Geoffrey Rush``` plays an ```art inspector``` which can detect fake portraits in a split of a second in the movie ```The Best Offer```. Geoffrey Rush can be seen as the discriminator in our GAN.

P.S. Sound on for the video below.

https://user-images.githubusercontent.com/59663734/159166914-406d3b01-8c66-4122-9b31-1ad33af48a2d.mp4


1. The generator is  going to start from some completely ```random noise``` and produce ```fake data```. At the beginning of this game, the generator actually isn't very sophisticated. It doesn't know how to produce real looking artwork. Additionally, the generator isn't allowed to see the real images. It doesn't know how this painting should even look. So at the very beginning, the elementary generator initially just paint a masterpiece of **scribbles**.


2. The discriminator is going to see the fake data from the Generator as well as ```real data```  that we would feed in and then it's going  to be trained to output a probability that the data it sees are real or fake. 

3. If it decides an image to be real then we can actually tell it ```yes``` that's ```real``` or ```no``` that's ```fake```. This way we can get a discriminator that's able to **differentiate** a poorly drawn image like this, from the ones that are slightly better and eventually also the real ones. 

4. In the beginning it's not going to be trained accurately so the predictions are not going to be mediocre but then we're going to train it till it starts increasing the probabilities of real versus not real appropriately such that we get this perfect separation where the discriminator is able to perfectly distinguish  what is real and what is fake.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159167795-a1f75871-dc12-4c21-8bac-f3e7f150fadd.png" />
</p>

5. The generator take instances of where the real data lies as inputs to train and then it's going to  try to improve its imitation of the data trying to move the fake data that is generated closer and closer to the real data.

6. When the generator produces a batch of paintings, the generator will know in what direction to go on and improve by looking at the ```scores``` assigned to her work by the discriminator.

7. Once again the discriminator is now going to receive these new points and it's going to estimate a probability that each of these points is real and again learn to decrease the probability of the fake points being real further and further. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159172289-5badead4-2768-4f6b-8ae6-be75431e1a3e.png" />
</p>

8. Eventually, the generator is going to start moving these fake points closer and closer to the real data such that the fake data are almost following the distribution of the real data. It is going to be really really hard for the discriminator to effectively  distinguish between what is real and what is fake while the generator is going to continue to try to create fake data instances to fool the discriminator.

9. The discriminator also improves over time because it receives more and more realistic images at each round from the generator. Essentially it tries to develop a keener and keener eye as these images get better.

10. When the discriminator says that the image created by the generator is 60% real. We actually that it's wrong that it's not necessarily real, that it's actually fake. And then after many rounds the generator, will start producing paintings that are harder and harder to distinguish if not impossible for the discriminator to distinguish from the real ones. Training ends when the first neural network begins to constantly deceive the second.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159173036-b0228b7f-abcd-480d-ba6a-847ecd9bafde.png" />
</p>

To summarize how we train GANs: the generator is going to try to synthesize fake instances to fool the discriminator which is going to be trained to identify the synthesized instances and discriminate these as fake.


#### 1.3 The Discriminator
We will now explore more in depth the ```discriminator``` part of the GAN. The discriminator is a classifier whose goal is to distinguish between different classes. Given the image of a cat, the classifier should be able to tell whether it's a cat or a dog. We can have a more complex case where we want to differentiate cat from multiple classes or the simplest case where we just want to predict cat or not cat.

In the video below, Jian Yang built quite a good ```binary classifier``` which can differentiate between hot dog and not hot dog, however much to the despair of Erlich.

https://user-images.githubusercontent.com/59663734/159260958-c3a00256-74f9-4e26-a01a-176e090a3519.mp4

One type of model for a classifier is using a neural network and this neural network can taken some features ```X``` and a set of labels ```Y``` associated with each of our classes.. It computes a series of nonlinearities and outputs the probabilities for a set of categories. The neural network learns these set of parameters or weights theta, <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta&space;" title="https://latex.codecogs.com/png.image?\dpi{110}\theta " />. These parameters data are trying to map these features X to those labels ```Y``` and those predictions are called <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" /> because they're not exactly the exact ```Y``` labels. They're trying to be the ```Y``` labels. And so the goal is to reach a point where the difference between the true values ```Y``` and the predictions <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" /> is minimized.

A cost function is computed by comparing how closely <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" /> is to Y. It will tell the discriminative model, the neural network, how close it is in predicting the correct class. From this cost function we can update those parameters - the nodes in that neural network according to the gradient of this cost function. This just means generally which direction those parameters should go to try to get to the right answer, to try to get to a <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" />, that's as close as possible to ```Y```. And then we repeat this process until our classifier is in good shape.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159347060-e27b77fc-f935-4474-b301-a228c69e279c.png" width="600" height="300"/>
</p>

The goal of the discriminator is to model the probability of each class and this is a conditional probability distribution because it's predicting the probability of class Y conditioned on a certain set of features. In the GAN context the discriminator is a classifier that inspects the examples. They are fake and real examples and it determines whether they belong to the real or fake class. The discriminator models the probability of an example being fake given that set of inputs X - ```P(Fake|Features)```. In the example below, the discriminator look at the fake Mona Lisa and determine that with 85% probability this isn't the real one - 0.85 fake. So in this case, it will be classified as fake and that information of being fake along with this fakeness probability, 0.85, will be given to the generator to improve its efforts. That is, the output probabilities from the discriminator are the ones that help the generator learn to produce better looking examples overtime

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159348473-49774c75-2c36-4d56-abbf-ec3071cd2c39.png" width="600" height="200"/>
</p>

##### 1.3.1 Training of Discriminator
Now we are going to see the whole process of the training of the discriminator which incorporates the output from the generator as well:

1. To train a discriminator we take a noise vector ```Z``` and pass it through the generator.

2. The generator produces a fake output, <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" />.

3. We also take a set of real images, <img src="https://latex.codecogs.com/png.image?\dpi{110}X" title="https://latex.codecogs.com/png.image?\dpi{110}X" /> from the original dataset and input both into the discriminator.

4. The discriminator is going to receive a set of both fake and real images and produce outputs <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" />.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159497436-230dd831-818a-4f76-bf4a-d57936f86678.png" width="800" height="270"/>
</p>


5. The output <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}" /> has a range from ```0``` to ```1``` where ```0``` represents the event of a fake image and ```1``` the maximum probability for a real image.

6. We then take that to a mathematical function that calculates the loss where we are going to compare fake inputs to the number ```0``` and real inputs to the number ```1```. 

**Note:** The discriminator wants to be able to predict that the fake inputs are fake, therefore have a propabilty of ```0``` and that the real inputs are real with a probability of ```1```.

8. Once we have calculated the loss we are going to use ```backpropagation``` to update the parameters, <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta&space;_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\theta _{d}" /> of the discriminator only.

#### 1.4 The Generator
The generator in a GAN is like it's ```heart```. It's a model that's used to generate examples and the one that we should be investing in and helping achieve a really high performance at the end of the training process.  

The generators final goal is to be able to produce examples from a certain class. So if we trained it from the class of a cat, then the generator will do some computations and output a representation of a cat that looks real.  So ideally, the generator won't output the same cat at every run, and so to ensure it's able to produce different examples every single time, we will actually input different sets of random values - a noise vector. Then this noise vector is fed in as ```input``` sometimes with our class ```Y``` for cat into the generators neural network. The generator in this neural network will compute a series of nonlinearities from those inputs and return some variables, for example, three million nodes at the end that do not necessarily represent classes but each pixel's value which represents the image of a cat.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159424620-4b8543c5-27ee-4d7a-aae6-61333a002a33.gif" />
</p>

##### 1.3.1 Training of Generator

1. We are going to begin with ```Z``` which represents a noise vector - a mathematical vector made of random numbers.

2. We pass this into a generator represented by a neural network to produce a set of features that can pose as an image of a cat or an attempt at a cat. This output image,  <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" /> is fake. It doesn't belong to the original real training data and we want to use it to fool the discriminator.

3. This image, <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{X}" /> is fed into the discriminator, which determines how real and how fake it thinks it is based on its inspection of it.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/161687658-bc5da96a-5d0c-4fac-9db0-52be7906c132.gif" />
</p>

4. The discriminator output <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" /> which is in the range of ```0``` to ```1``` will be used to compute a ```cost function``` that basically looks at how far the examples produced by the generator are being considered real by the discriminator because the generator wants this to seem as real as possible. That is, how good is the performance of the generator?

5. The generator wants <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" /> to be as close to ```1```, meaning ```real``` as possible. Whereas, the discriminator is trying to get this to be ```0``` - ```fake```.  Hence, the predictions are compared using the loss function with all the labels equal to real. Because the generator is trying to get these fake images to be equal to real or label of 1 as closely as possible.

6. The cost function uses the difference between these two to then update the parameters of the generator using backpropagation. It gets to improve over time and know which direction to move it's parameters to generate something that looks more real and will fool the discriminator.

7. The difference between the output of the discriminator and the value ```1``` is going to be a smaller and smaller and the loss is going to be smaller and smaller. As such, the performance of the generator is going to keep on improving.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159497152-4792bbcb-ecd1-4640-a951-855e0dfc3231.png" width="800" height="250"/>
</p>

So once we get a generator that looks pretty good, we can save the parameters theta of the generator and then ```sample``` from this safe generator. What sampling basically means is that we have these random noise vectors and when we input that into the saved generator, it can generate all sorts of different examples. 

More generally, the generator is trying to model the probability of features X given the class Y - ```P(X|Y)```. Note: if we have only one class then we dont need that class ```Y``` so we will just model ```P(X)```. For the example above, generator will model the probability of features ```X``` without any additional conditions and this is because the class ```Y``` will always be cat, so it's implicit for all probabilities ```X```. In this case, it'll try to approximate the real distribution of cats. So the most common cat breeds will actually have more chances of being generated because they're more common in the dataset. Certain features such as having pointy ears will be extra common because most cats have that. But then more rare breeds, the sphix for example, will have a less likely chance of being sampled. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159434062-433e0da7-5178-42c6-b90f-861dbbae7ea2.png" />
</p>

To summarise:

- the generator produces fake data that tries to look real. 
- It learns to mimic that distribution of features X from the class of your data. 
- In order to produce different outputs each time it takes random features as input. 


#### 1.4 Cross Entropy Cost Function
To understand the ```Binary Cross Entropy``` cost function, we will first explore what is entropy. 

##### 1.4.1 Information
First, ```information``` is defined as the number of bits required to encode and transmit an event.
- Low Probability Event (**surprising**): ```More information```.
- Higher Probability Event (**unsurprising**): ```Less information```.
Information h(x) can be calculated for an event x, given the probability of the event P(x) as follows:

<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}h(x)&space;=&space;-log(P(x))" title="https://latex.codecogs.com/png.image?\dpi{110}h(x) = -log(P(x))" />
</p>

- Low Probability Event: P(x) = **0.1** | h(x) = -log(0.1) = **1** : More information
- Higher Probability Event: P(x) = **0.9** | h(x) = -log(0.9) = **0.045** : Less information

Figure below shows a - log(x) graph.
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159455785-b1559ee6-8b9a-404a-a3a4-236e298d4ee9.png" width="450" height="300"/>
</p>

**Note:**
Imagine that we are encoding a particular event. If the probability of that event happening is low, this means that it is more surprising, because we are not sure when it is going to happen. And we will also need to use more bits to encode it because we need to encode a more surprising pattern, which has more variation and requires more bits to be expressed.

Conversely, if we know that the event happens very often, with high probability, then it will be less surprising because we are almost sure that it is going to happen next time we check. And that high probability event has less information because we require less bits to express a pattern that happens almost always or always in contrast to a pattern that is more unexpected and complex.

##### 1.4.2 Entropy
Now let's explore entropy. ```Entropy``` is the number of bits required to transmit a randomly selected event from a probability distribution. 

- Skewed Probability Distribution (**unsurprising**): ```Low entropy```.
- Balanced Probability Distribution (**surprising**): ```High entropy```.

Entropy H(x) can be calculated for a random variable with a set of x in X discrete states discrete states and their probability P(x) as follows:

<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}H(X)&space;=&space;-\sum_{i=1}^{n}&space;P(x_{i})&space;*&space;log(P(x_{i}))" title="https://latex.codecogs.com/png.image?\dpi{110}H(X) = \sum_{i=1}^{n} P(x_{i}) * log(P(x_{i}))" />
</p>

- **Skewed Distribution**: one high probability event (0.9) and two low probability events (0.05): ```Low Entropy - Unsurprising```

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159460075-478d9837-a513-4b82-a9dd-bd0556f88fe9.png" />
</p>

- **Balanced Distribution**: all events have same probability (0.33) ```High Entropy - Surprising```

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159462252-684b32b2-7010-4aa4-bc9e-7594b1b5bc0c.png" />
</p>

##### 1.4.3 Cross Entropy
Cross-entropy is a measure of the **difference** between two probability distributions for a given random variable or set of events.

_The intuition for this definition comes if we consider a target or underlying probability distribution P and an approximation of the target distribution Q, then the cross-entropy of Q from P is the number of additional bits to represent an event using Q instead of P._

The cross-entropy between two probability distributions, such as Q from P, can be stated formally as:
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159463250-bfd9e0c4-c4aa-4f59-826c-370f02cf1121.png" />
</p>

Where ```H()``` is the ```cross-entropy``` function, ```P``` may be the ```target``` distribution and ```Q``` is the ```approximation``` of the target distribution.


##### 1.4.4 Binary Cross Entropy
We will use Binary Cross Entropy because the discriminator wants to predict two things: real images are real and the fake images are fake. 

**Recall:** Cross entropy is the product of the **target probability** times the **logarithm of the approximating probability**.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159465858-1c6506c8-a17a-4c69-a541-fa7c31f4d24a.png" />
</p>

where ```y``` is the target (target probability) label which is ```1``` for real and ```0``` for fake and <img src="https://latex.codecogs.com/svg.image?\hat{y}" title="https://latex.codecogs.com/svg.image?\hat{y}" /> represents the prediction (approx. probability) - the output of the discriminator. So we can see our cost function as:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159466939-5e314a92-4f43-4dd8-942a-6540e763380d.png" />
</p>

We notice that our cost function has two parts: one more focused on the real images and one more focused on the fake. We will now look at the function when y = 1 and when y = 0.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

**```y = 1:```**

 When the label is equal to ```1``` we have only the first part of the equation which is:
 
<p align="center">
  <img src= "https://latex.codecogs.com/svg.image?y&space;*&space;log(\hat{y})" title="https://latex.codecogs.com/svg.image?y * log(\hat{y})" />
</p>
 
 that is:
 
<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}target&space;*&space;log(prediction)" title="https://latex.codecogs.com/png.image?\dpi{110}target * log(prediction)" />
</p>

1. We see that when ```y = 0```, we output ```0```. 
2. If we have a label of ```1``` and we have a really high prediction that is close to ```1``` -  of 0.99, then we also get a value that's close to ```0```.
3. In the case where it actually is real, i.e, y = 1, but our prediction is terrible, and it's 0, so far from 1, you think it's fake, but it's actually real, then this value is extremely large. 

This term <img src= "https://latex.codecogs.com/svg.image?y&space;*&space;log(\hat{y})" title="https://latex.codecogs.com/svg.image?y * log(\hat{y})" /> is mainly for when the prediction is actually just 1, and it makes it 0 if our prediction is good, and it makes it negative infinity if our prediction is bad.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159486275-25b8ce6d-aa62-4bf7-9aab-fa2b12df949e.png" />
</p>

In this plot, we have our prediction value on the x-axis and the loss associated with that training example on the y-axis. In this case, the loss simplifies to the negative log of the prediction. When the prediction is close to ```1```, here at the tail, the loss is close to ```0``` because our prediction is close to the label. However, when the prediction is close to ```0``` out here, unfortunately our loss approaches infinity, so a really high value because the prediction and the label are very different. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159491019-7e513dbe-b5cf-48fd-93a1-a99f352ba741.png" />
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------------

**```y = 0:```**

 When the label is equal to ```0``` we have only the second part of the equation which is:
 
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159487419-8aaed528-e48d-4131-b483-8799d1d64f63.png" />
</p>

that is:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159487659-4fa0120b-009e-4cb2-8e14-f162b9fffe0b.png" />
</p>

Similarly:

1.  If our label is 1, then 1-y = 0. And if our prediction is anything, this will evaluate to ```0```.
2.  If our prediction is close to zero and our label is 0, then this value is close to 0. 
3.  However, if it's fake, but our prediction is really far off, and thinks it's real, then this term evaluates to negative infinity. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159486345-757a8e95-da6f-4fa8-97d5-c566b98f1c35.png" />
</p>

When the label is 0, and the loss function reduces to the negative log of 1 minus that prediction. Hence, when the prediction is close to 0, the loss is also close to 0. That means we're doing great. But when our prediction is closer to 1, but the ground truth is 0, it will approach infinity again.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159517960-54a5c554-487c-419a-8738-5c228acd40a7.png" />
</p>

Basically, each of these terms - <img src="https://latex.codecogs.com/png.image?\dpi{110}(1-y)*log(1-\hat{y})" title="https://latex.codecogs.com/png.image?\dpi{110}(1-y)*log(1-\hat{y})" /> and <img src="https://latex.codecogs.com/png.image?\dpi{110}(1-y)*log(1-\hat{y})" title="https://latex.codecogs.com/png.image?\dpi{110}(1-y)*log(1-\hat{y})" /> evaluates to negative infinity if for their relevant label, the prediction is really bad. 

**Why do we have a negative sign in front of our cost function?**

If either of these values evaluates to something really big in the negative direction, then this negative sign is crucial to making sure that it becomes a positive number and positive infinity. Because for our cost function, what we typically want is a high-value being bad, and our neural network is trying to reduce this value as much as possible. Getting predictions that are closer, evaluating to ```0``` makes sense here, because we want to minimize our cost function as we learn.


-------------------------------------------------------------------------------------------------------------------------------------------------------------
In summary, one term in the cost function is relevant when the label ```0```, the other one is relevant when it's ```1```, and in either case, the logarithm of a value between 1-0 was calculated, which returns that negative result. That's why we want this negative term at the beginning, to make sure that this is high, or greater than, or equal to 0. When prediction and the label are similar, the BCE loss is close to 0. When they're very different, that BCE loss approaches infinity. The BCE loss is performed across a mini-batch of several examples - n examples. It then takes the average of all those n examples.

Our cost function needs to define a ```global optimum``` such that the generator could perfectly reproduce the true data distribution such that the discriminator  cannot absolutely  tell what's synthetic versus what's real.

##### 1.4.5 Discriminator Loss
If we consider the loss  from the perspective of the discriminator we want to try to ```maximize``` the probability that the fake data is identified as fake and real data is identified as real.

_We train D to maximize the probability of assigning the correct label to both training examples and samples from G._

Therefore, the discriminator wants to maximize the average of the log probability for real images and the log of the inverted probabilities of fake images:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159638390-352aa2d8-1c31-48c6-adb7-1b9c82e1719b.png" />
</p>

- ```log(D(X))``` is the discriminator's output for real data X. This is going to be likelihood of real data being real from the data distribution. So we want it to output ```1```.

- ```G(z)```  defines the generator's output and so ```D(G(z))``` is the discriminator's estimate of the probability that a fake instance is actually fake. So we want it to output ```0``` and then ```1 - D(G(z))``` becomes ```1``` so we are left with ```log(1)``` which also equals to ```0```.

Therefore, the dircriminator wants to ```maximize``` objective such that ```D(x)``` is close to ```1``` (**real**)  and ```D(G(z))``` is close to ```0``` (**fake**).

##### 1.4.6 Generator Loss
The generator seeks to minimize the log of the inverse probability predicted by the discriminator for fake images. This has the effect of encouraging the generator to generate samples that have a low probability of being fake.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159644272-7835afdd-4a84-4f31-9465-9382c96fdea0.png" />
</p>


- ```G(z)```  defines the generator's output and so ```D(G(z))``` is the discriminator's estimate of the probability that a fake instance is actually fake. So the generator wants it to output ```1```(wants to fool discriminator that data is real). ```1 - D(G(z))``` becomes ```0``` so we are left with ```log(0)``` which is undefined (-inf).

Therefore, the generator wants to ```minimize``` objective such that ```D(G(z))``` is close to ```1```(Discriminator is fooled into thinking generated G(z) is real).


For the GAN, the generator and discriminator are the two players and take turns involving updates to their model weights. The min and max refer to the minimization of the generator loss and the maximization of the discriminator’s loss.


Now, we have these two players and so we're going to train this jointly in a ```minimax``` game formulation. It's going to be minimum over <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta&space;_{g}" title="https://latex.codecogs.com/png.image?\dpi{110}\theta _{g}" />, our parameters of our generator network G, and maximum over parameter <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta&space;_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\theta _{d}" /> of our Discriminator network D.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159647158-fd5fa24b-a339-4399-bd9c-87e4b629b9b6.png" />
</p>


In order to train this, we're going to alternate between gradient ascent on our discriminator to maximize this objective and then gradient descent on the generator to minimize the objective.

1. **Gradient Ascent** on Discriminator:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159638390-352aa2d8-1c31-48c6-adb7-1b9c82e1719b.png" />
</p>


2. **Gradient Descent** on Generator:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159644272-7835afdd-4a84-4f31-9465-9382c96fdea0.png" />
</p>

##### 1.4.7 Non-Saturating GAN Loss
In practice, this loss function for the generator saturates. This means that if it cannot learn as quickly as the discriminator, the discriminator wins, the game ends, and the model cannot be trained effectively. Let's see how:

- When plotting the graph ```log(1-D(G(z)))```, we see that the slope of this loss is actually going to be higher towards the right. That is, the slope is high when ```D(G(z))``` - our generator - is doing a good job of fooling the discriminator. 
 
 - And on the other hand when we have bad samples, i.e, when our generator has not learned a good job yet, therefore when the discriminator can easily tell it is fake data, the gradient is closer to this zero region on the X axis. This actually means that our gradient signal is dominated by region where the sample is already pretty good. Whereas we actually want it to learn a lot when the samples are bad. And thus, this makes it hard to learn.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159667940-65392410-b1ae-44f3-b0bf-6ecf08061a53.png" />
</p>

In order to improve learning, we're going to define a different objective function for the gradient where we are now going to do **gradient ascent** on the generator instead. In the previous case, the generator sought to minimize the probability of images being predicted as fake. Here, the generator seeks to maximize the probability of images being predicted as real. So instead of seeing the glass half empty, we want to see it half full.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159670578-77a32bed-cb94-4024-a43c-13ad3cee2cba.png" />
</p>

If we plot this function on the right here, then we have a high gradient signal in this region on the left where we had bad samples, and now the flatter region is to the right where we would have good samples. So now we're going to learn more from regions of bad samples and so this has the same objective of fooling the discriminator but it actually works much better in practice.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159671193-c0dcca4d-9b18-44ba-b0cc-a9c10670f29f.png" />
</p>


We need to alternate their training, only one model is trained at a time, while the other one is held constant. 

**Note:** It's important to keep in mind that both models should improve ```together``` and should be kept at ```similar``` skill levels from the beginning of training. And so the reasoning behind this is if we had a discriminator that is superior than the generator, we'll get predictions from it telling us that all the fake examples are 100% fake. the generator doesn't know how to improve. Everything just looks super fake, there isn't anything telling it to know which direction to go in.

On the other hand, if we had a superior generator that completely outskills the discriminator, then we'll get predictions telling us that all the generated images are 100% real. The discriminator has a much easier task, it's just trying to figure out which ones are real, which ones are fake, as opposed to model the entire space of what a class could look like.  And so having output from the discriminator be much more informative, like 0.87 fake or 0.2 fake as opposed to just 100% fake or of probability one fake, is much more informative to the generator in terms of updating its weights and having it learn to generate realistic images over time.


##### 1.4.8 Generating New Data
After training  we can actually use the generator network which is now fully trained to produce new data instances that have never been seen before.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159675456-740a97c3-1b36-42de-8ef7-22d4ff98628a.png" width="500" height="350" />
</p>

When the trained generator of a GAN synthesizes new instances, it's effectively learning a transformation from a distribution of noise to a target data distribution and that transformation - that mapping is going to be what's learned over the course of training. If we consider one point from a latent noise distribution it's going to result in a particular output in the target data space and if we consider another point of random noise and feed it through the generator, it is going to result in a new instance. That new instance is going to fall somewhere else on the data manifold.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159677674-150fef07-783e-445c-ba96-b97853eac72a.gif" />
</p>


### 2. Wasserstein GANs with Gradient Penalty
A major issue with GAN is when a GAN generates the same thing each time. For example, a GAN trained on all different cat breeds will only generate a Sphinx cat. This issue happens because the discriminator improves but it gets stuck between saying an image of a cat looks ```extremely fake``` or ```extremely real```. 

The discriminator being a ```classifier``` is encouraged to say it's ```1``` - **real** or ```0``` - **fake** as it gets better. But in a single round of training, if the discriminator only thinks the generator's data looks real, even if it doesn't even look that real, then the generator will **cling** on to that image and **only** produce that type of data. 

Now when the discriminator learns that the data is fake in the next round of training, the generator won't know where to go because there's really nothing else it has in its arsenal of different images and that's the end of learning. Digging one level deeper, this happens because of **binary cross-entropy loss**, where the discriminator is forced to produce a value between zero or one, and even though there's an infinite number of decimal values between zero and one, it'll approach zero and one as it gets better.

#### 2.1 Mode Collapse
What is mode? The mode is the value that appears most often in a set of data values. If X is a discrete random variable, the mode is the value x at which the probability mass function takes its ```maximum``` value. In other words, it is the value that is ```most likely``` to be sampled.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/161033722-07ced60d-9d35-49db-9f4b-a61386637cd0.png" />
</p>

As shown above, the mean value in a normal distribution is the single mode of that distribution. There are instances whereby for a probability density distribution we have two modes (```Bimodal```) and mean does not necessarily have to be one of them. So more intuitively, any peak on the probability density distribution over features is a mode of that distribution.

Figure below shows handwritten digits represented by features <img src="https://latex.codecogs.com/png.image?\dpi{110}x_{1}" title="https://latex.codecogs.com/png.image?\dpi{110}x_{1}" /> and <img src="https://latex.codecogs.com/png.image?\dpi{110}x_{2}" title="https://latex.codecogs.com/png.image?\dpi{110}x_{2}" />. The probability density distribution in this case will be a surface with many peaks corresponding to each digit. This is ```multimodal``` with 10 different modes for each number from ```0``` to ```9```.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/161037371-4c829b6b-8199-4a49-919c-39c62da3eefa.png"  width="500" height="300"/>
</p>

We can imagine each of these peaks coming out at us in a 3D representation where the darker circle represents higher altitudes. So average looking ```7``` represented in red wiill be at the mode of the distribution.

To understand mode collapse let's take for example a discriminator who can perfectly classify each handwritten digits except ones and sevens.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/161051519-a66a3e8e-8015-49c7-99f5-01f346ec8d4f.png"/>
</p>

Eventually the discriminator will probably catch on and learn to catch the generator's fake handwritten number ones by getting out of that ```local minima```. But the generator could also migrate to another mode of the distribution and again would collapse again to a different mode. Or the generator would not be able to figure out where else to diversify. 


 To sum up:
 
- Modes are peaks of the probability distribution of our features. 
- Real-world datasets have many modes related to each possible class within them. 
- Mode collapse happens when the generator learns to fool the discriminator by producing examples from a ```single class``` from the whole training dataset like handwritten number ones. This is unfortunate because, while the generator is optimizing to fool the discriminator, that's not what we ultimately want our generator to do.


#### 2.2 Limitation of BCE Loss
Recall the BCE loss function is just an average of the cost for the discriminator for misclassifying real and fake observations. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159465858-1c6506c8-a17a-4c69-a541-fa7c31f4d24a.png" />
</p>

The first term <p align="center">
  <img src= "https://latex.codecogs.com/svg.image?y&space;*&space;log(\hat{y})" title="https://latex.codecogs.com/svg.image?y * log(\hat{y})" />
</p>  is for reals and the second term   <p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159487419-8aaed528-e48d-4131-b483-8799d1d64f63.png" />
</p> is for the fakes. The higher this cost value is, the worse the discriminator is doing at it.  

. 

The generator wants to ```maximize``` this cost because that means the discriminator is doing poorly and is classifying it's fake values into reals. Whereas the discriminator wants to ```minimize``` this cost function because that means it's classifying things correctly. Note that the generator only sees the fake side of things, so it actually doesn't see anything about the reals. This maximization and minimization is often called a ```minmax game```.

The discriminator needs to output just a **single value** prediction within ```0``` and ```1```. Whereas the generator actually needs to produce a pretty **complex** output composed of multiple features to try to fool the discriminator. As a result that discriminators job tends to be a little bit easier. To put it in another way: _critisizing is more straightforward_. As such, during training it's possible for the discriminator to outperform the generator.

##### 2.2.1 Beginning of training
We have two distributions: the ```real distribution``` and the ```generator distribution```. The objective of the GAN is to bring them together, i.e, to make the generator distribution be as close as possible to the real distribution so that the fake images are as similar as possible to the real images. 

At the beginning of training the discriminator has trouble distinguishing the generated and real distributions. There is some overlap and it is not quite sure. As a result, it's able to give useful feedback in the form of a ```non-zero gradient``` back to the generator.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162175861-6d0b3751-a9a2-403f-94ca-33ef10de8ddf.png" width="680" height="210"/>
</p>


##### 2.2.2 With more training
As it gets better at training, it starts to delineate the generated and real distributions a little bit more such that it can start distinguishing them much more. The real distribution will be centered around ```1``` and the generated distribution will start to approach ```0```. As a result, when the discriminator is getting better, it will start giving less informative feedback. In fact, it might give **gradients closer to zero**, and that becomes unhelpful for the generator because then the generator doesn't know how to improve. This is how the ```vanishing gradient``` problem will arise. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162177007-8c208317-6d3a-4702-91ee-e0c41619982e.png" width="680" height="230"/>
</p>


To sum up:

- GANs try to make real and generated distribution look similar.
- When the discriminator improves too much, the function approximated by BCE loss will contain flat regions.
- These flat regions cause vanishing gradient whereby the generator stops improving.

#### 2.3 Earth Mover's Distance
When using BCE loss to train a GAN, we often encounter ```mode collapse``` and ```vanishing gradient``` problems due to the underlying cost function of the whole architecture. Even though there is an infinite number of decimal values between ```0``` and ```1```, the discriminator, as it improves, will be pushing towards those ends.

The Earth Mover's distance measures how different these two distributions are by estimating the amount of ```effort``` it takes to make the generated distribution equal to the real. Recall that the objective of the GAN is to make the generator distribution as equal as possible to the real distribution. The function depends on both the ```distance``` and the ```amount that the generated distribution needs to be moved```. In terms of an analogy, the generated distribution can be considered a pile of dirt and the Earth mover's distance means how difficult would it be to move that pile of dirt and mold it into the shape and location of the real distribution.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162197487-bf6f436e-c0ab-456c-887e-682a65032569.png" width="680" height="250"/>
</p>


The problem with BCE loss is that as a discriminator improves, it would start giving more **extreme values** between ```0``` and ```1```. As a result, this become less helpful feedback for the generator and the generator would stop learning due to vanishing gradient problems. With Earth mover's distance, however, there's no such ceiling to the ```0``` and ```1```. The cost function continues to **grow** regardless of how far apart these distributions are. The gradient of this measure won't approach ```0``` and as a result, GANs are less prone to vanishing gradient problems and from vanishing gradient problems, mode collapse.

In summary:

- Earth mover’s distance is a measure of how different two distributions are by estimating the effort it takes to make the generated distribution equal to the real one.
- Earth mover’s distance does not have flat regions when the distributions are different.


#### 2.4 Wasserstein Loss
An alternative loss function called ```Wasserstein Loss``` - ```W-Loss``` approximates the Earth Mover's Distance. Instead of using a discriminator to classify or predict the probability of generated images as being real or fake, the WGAN changes or replaces the discriminator model with a ```critic``` that **scores** the ```realness``` or ```fakeness``` of a given image. Specifically, the lower the loss of the critic when evaluating generated images, the higher the expected quality of the generated images. 

The discriminator is no longer bounded between ```0``` and ```1```, i.e, it is no longer discriminating between these two classes. And so, our neural network cannot be called a discriminator because it doesn't discriminate between the classes. And so, for W-Loss, the equivalent to a discriminator is called a ```critic```, and what the Wasserstein loss function seeks to do is ```increase``` the gap between the scores for real and generated images.

We can summarize the function as it is described in the Wasserstein GAN paper as follows:

- Critic Loss = [average critic score on real images] – [average critic score on fake images]
- Generator Loss = -[average critic score on fake images]

Where the average scores are calculated across a mini-batch of samples.

So the discriminator wants to ```maximize``` the distance between its thoughts on the reals versus its thoughts on the fakes. So it's trying to push away these two distributions to be as far apart as possible. 

<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}\underset{C}{max}(\mathbb{E}\cdot&space;C(x))&space;-&space;(\mathbb{E}\cdot&space;C(G(z)))" title="https://latex.codecogs.com/png.image?\dpi{110}\underset{C}{max}(\mathbb{E}\cdot C(x)) - (\mathbb{E}\cdot C(G(z)))"/>
</p>

In the case of the critic, a larger score for real images results in a larger resulting loss for the critic, penalizing the model. This encourages the critic to output **smaller scores for real images**. For example, an average score of 20 for real images and 50 for fake images results in a loss of -30; an average score of 10 for real images and 50 for fake images results in a loss of -40, which is better, and so on.

Meanwhile, the generator wants to ```minimize``` this difference because it wants the discriminator to think that its fake images are as close as possible to the reals.

<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}\underset{G}{min}-(\mathbb{E}\cdot&space;C(G(z))" title="https://latex.codecogs.com/png.image?\dpi{110}\underset{G}{min}(\mathbb{E}\cdot C(G(z))"/>
</p>


In the case of the generator, a ```larger score``` from the **critic** will result in a ```smaller loss``` for the **generator**, encouraging the critic to output larger scores for fake images. For example, an average score of 10 becomes -10, an average score of 50 becomes -50, which is smaller, and so on.


Nore that the sign of the loss does not matter in this case, as long as ```loss for real images``` is a **small** number and the ```loss for fake images``` is a **large** number. The Wasserstein loss encourages the critic to separate these numbers.


<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}\underset{G}{min}\cdot&space;\underset{C}{max}[\mathbb{E}\cdot&space;C(x)&space;-&space;\mathbb{E}\cdot&space;C(G(z))]" title="https://latex.codecogs.com/png.image?\dpi{110}\underset{G}{min}\cdot \underset{C}{max}[\mathbb{E}\cdot C(x) - \mathbb{E}\cdot C(G(z))]"/>
</p>

In these functions:

- C(x) is the critic's output for a real instance.
- G(z) is the generator's output when given noise z.
- C(G(z)) is the critic's output for a fake instance.


The discriminator model is a neural network that learns a binary classification problem, using a ```sigmoid activation function``` in the output layer, and is fit using a ```binary cross entropy``` loss function. As such, the model predicts a probability that a given input is real (or fake as 1 minus the predicted) as a value between 0 and 1. ```W-Loss```, however, doesn't have that requirement at all, so we can actually have a ```linear layer``` at the end of the discriminator's neural network and that could produce any real value output. And we can interpret that output as how real an image is considered by the critic.

Note: Some of the explanations above are based from the blog of _machinelearningmastery_.

In summary:

- the discriminator under **BCE** Loss outputs a value between ```0``` and ```1```, while the critic in **W-Loss** will output ```any number```.
- because it's not bounded, the critic is allowed to improve without degrading its feedback back to the generator. 
- It doesn't have a vanishing gradient problem, and this will mitigate against mode collapse, because the generator will always get useful feedback.
- The ```generator``` tries to ```minimize``` the W-Loss - trying to get the generative examples to be as close as possible to the real examples while the ```critic``` wants to ```maximize``` this expression because it wants to differentiate between the reals and the fakes, it wants the distance to be as large as possible. 


#### 2.5 Condition on Wasserstein Critic
Recall W-Loss is a simple expression that computes the difference between the expected values of the critics output for the real examples ```x``` and its predictions on the fake examples ```G(z)```. The **generator** tries to ```minimize``` this expression: trying to get the generative examples to be as close as possible to the real examples while the **critic** wants to ```maximize``` this expression: it wants to differentiate between the reals and the fakes - it wants the distance to be as large as possible. 

However, the condition is that the critic needs to be ``` 1-Lipschitz Continuous ``` or ```1-L Continuous``` which means that the ```norm of its gradient``` needs to be **at most** ```1```. That is, the slope or gradient can't be greater than ```1``` at any point. In order to check a function is 1-Lipschitz Continuous, we want to go along every point in the function and make sure its slope or gradient is <img src="https://latex.codecogs.com/svg.image?\leq&space;" title="https://latex.codecogs.com/svg.image?\leq " />  ```1```.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162484119-5b6110e3-4394-485d-8cba-cbb54dcb25c5.png"/>
</p>


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162478767-916e9a1c-6ec8-4168-b864-d723d6f0986d.png" width="400" height="380"/>
</p>


In order to check that, we drew two lines of gradient ```1``` and ```-1``` respectively then we want to make sure that the ```growth``` of this function never goes out of **bounds** from these lines because staying within these lines means that the function is growing ```linearly```.   The function above is not 1-Lipschitz Continuous because it's not staying within this green area which suggests that it's growing more than linearly. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162481575-48e90719-c8f1-4000-b3cc-5c4f157729e0.png" width="400" height="400"/>
</p>

Above is a smooth curve function. We want to again check every single point on this function before we can determine whether or not that this is 1-Lipschitz Continuous. We take every single value and the function never grows more than linearly hence , this function is 1-Lipschitz Continuous.

This condition on the critics neural network is important for W-Loss because it assures that the W-Loss function is not only ```continuous``` and ```differentiable``` but also that it doesn't grow too much and maintain some ```stability``` during training. This is what makes the underlying Earth Movers Distance valid, which is what W-Loss is founded on. This is required for training both the critic and generators neural networks and it also ```increases stability``` because the variation as the GAN learns will be **bounded**.

#### 2.6 1-Lipschitz Continuity Enforcement
Two common ways of ensuring this condition are ```weight clipping``` and ```gradient penalty```:

##### 2.6.1 Weight clipping
With weight clipping, the ```weights``` of the critics neural network are forced to take values between a ```fixed interval```. After we update the weights during gradient descent, we will actually clip any weights outside of the desired interval, i.e, weights that are either too high or too low will be set to the ```maximum``` or the ```minimum``` amount allowed. However this has a couple of downside. 

- Forcing the weights of the critic to a limited range of values could ```limit``` the critics ability to ```learn``` and ultimately for the gradient to perform.
- If the critic can't take on many different parameter values, it's weights can't take on many different values then it might not be able to improve easily or find a good global optima for it to be in.
- Or on the other hand, it might actually ```limit``` the critic ```too little``` if we don't clip the weights enough.

##### 2.6.2 Gradient penalty
The gradient penalty is a much softer way to enforce the critic to be 1-lipschitz continuous. All we need to do is add a ```regularization``` term to our loss function which will **penalize** the critic when it's gradient norm is higher than ```1```.
 
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162486035-ef2c55c7-dc53-443e-9344-c3dd5afa0e83.png"/>
</p>

where ```reg``` is the regularization term and ```lambda``` is just a hyperparameter value of how much to weigh this regularization term against the main loss function. 

In order to check the critics gradient at every possible point of the feature space, that's virtually impossible or at least not practical. Instead with gradient penalty what we will do is ```sample``` some points by ```interpolating``` between real and fake examples using a random number ```epsilon```. It is on <img src="https://latex.codecogs.com/svg.image?\hat{x}" title="https://latex.codecogs.com/svg.image?\hat{x}" /> - the interpolated image - that we want to get the critics gradient to be <img src="https://latex.codecogs.com/svg.image?\leq&space;1&space;" title="https://latex.codecogs.com/svg.image?\leq 1 " />.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162605691-7f9bbd06-e269-4276-b9e9-e4d37c3f22fb.png" width="700" height="350"/>
</p>

where:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162605675-257e162b-6c18-4c9d-8cc2-96c788801c4b.png"/>
</p>

**Note:** Since checking the critic’s gradient at each possible point of the feature space is virtually impossible, we can approximate this by using interpolated images.


We get the gradient of the critics prediction on <img src="https://latex.codecogs.com/svg.image?\hat{x}" title="https://latex.codecogs.com/svg.image?\hat{x}" />, and then we take the norm of that gradient and we want the norm to be ```1```.  In fact it is penalizing any value outside of ```1```.  With this method, we're not strictly enforcing ```1-L continuity```, but you're just **encouraging** it. This has proven to work well and much better than weight clipping.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162608155-07a8f0f0-9b22-47af-aa06-aa8ede1497ec.png"/>
</p>

The complete expression of the loss function that we use for training with ```W-loss``` gradient penalty now has these two components:

1.  First, we approximate ```Earth Mover's distance``` with this main W-loss component. This makes the GAN less prone to mode collapse and vanishing gradient.
2.  The second part of this loss function is a ```regularization term``` that meets the condition for what the critic desires in order to make this main term valid. Of course, this is a **soft constraint** on making the critic ```1-lipschitz continuous``` for the loss function to be ```continuous``` and ```differentiable```. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162608514-cec2d004-b140-4565-9b75-09ba514a52ab.png"/>
</p>




#### 2.7 Coding a WGAN
Recall that a GAN consists of two networks that train together:

 - **Generator** — Given a vector of random values - ```noise``` as input, this network generates data with the same distribution as the training data. We train the generator to generate data that "_fools_" the discriminator.

- **Discriminator** — Given batches of data containing observations from both the training data and generated data from the generator, this network attempts to classify the observations as ```real``` or ```fake```.We train the discriminator to distinguish between real and generated data.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162626227-7ac760f2-4c27-44f8-9cdc-a16a93ff9a1d.png" width="650" height="150"/>
</p>


Ideally, twe want a generator that generates convincingly realistic data and a discriminator that has learned strong feature representations that are characteristic of the training data.

We will use the ```CelebA Dataset``` to create a GAN that will generate persons' faces. We will build a **Generator** and **Critic** using ```Transposed Convolutions``` and ```Convolutions``` respectively. More explanations on convolutions can be found at this link: [Lane-Detection-with-Semantic-Segmentation](https://github.com/yudhisteer/Lane-Detection-with-Semantic-Segmentation)


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162926443-94916263-987e-4ccb-8051-bb12be797035.png" width="350" height="350"/>
</p>

##### 2.7.1 The Generator Model
We will first define the generator network architecture which generates images from ```1x1x200``` arrays of random values. The network:

 - Converts the random vectors of size ```200``` to ```1x1x128``` arrays using a project and reshape - forward function.

- Upscales the resulting arrays to ```64x64x3``` arrays using a series of transposed convolution layers and ReLU layers.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/162615819-36200885-5968-404d-8b41-11bf0bdab7f2.png"/>
</p>



- For the transposed convolution layers, we specify ```4x4``` **filters**(F) with a decreasing number of filters for each layer, a **stride** (S) of ```2```, and **padding** (P) amount.

 - For the final transposed convolution layer, we specify three ```4x4``` filters corresponding to the ```3``` **RGB** channels of the generated images, and the output size of the previous layer.

- At the end of the network, include a ```tanh``` layer.

![image](https://user-images.githubusercontent.com/59663734/162618739-9d906c91-6d0a-404e-a579-56138f5ec45f.png)

```python
# generator model

class Generator(nn.Module):
  def __init__(self, z_dim=200, d_dim=16):
    super(Generator, self).__init__()
    self.z_dim=z_dim

    self.gen = nn.Sequential(
            ## ConvTranspose2d: in_channels, out_channels, kernel_size, stride=1, padding=0
            ## Calculating new width and height: (n-1)*stride -2*padding +ks
            ## n = width or height
            ## ks = kernel size
            ## we begin with a 1x1 image with z_dim number of channels (200) - initlalized z_dim = 200 | 1x1x200
            ##  - we decrease no. of channels but increase size of image

            nn.ConvTranspose2d(z_dim, d_dim * 32, 4, 1, 0), ## 4x4 image (ch: 200 to 512) | 4x4x512
            nn.BatchNorm2d(d_dim*32),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim*32, d_dim*16, 4, 2, 1), ## 8x8 image (ch: 512 to 256) | 8x8x256
            nn.BatchNorm2d(d_dim*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim*16, d_dim*8, 4, 2, 1), ## 16x16 image (ch: 256 to 128) | 16x16x128
            #(n-1)*stride -2*padding +ks = (8-1)*2-2*1+4=16
            nn.BatchNorm2d(d_dim*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim*8, d_dim*4, 4, 2, 1), ## 32x32 image (ch: 128 to 64) | 32x32x64
            nn.BatchNorm2d(d_dim*4),
            nn.ReLU(True),            

            nn.ConvTranspose2d(d_dim*4, d_dim*2, 4, 2, 1), ## 64x64 image (ch: 64 to 32) | 64x64x32
            nn.BatchNorm2d(d_dim*2),
            nn.ReLU(True),            

            nn.ConvTranspose2d(d_dim*2, 3, 4, 2, 1), ## 128x128 image (ch: 32 to 3) | 128x128x3
            nn.Tanh() ### produce result in the range from -1 to 1
    )

  #--- Function to project and reshape noise
  def forward(self, noise):
    x=noise.view(len(noise), self.z_dim, 1, 1)  # 128 batch x 200 no. of channels x 1 x 1 | len(noise) = batch size = 128
    print('Noise size: ', x.shape)
    return self.gen(x)
```



##### 2.7.2 The Critic Model
For the discriminator, we create a network that takes ```128x128x3``` images and returns a ```scalar``` prediction score using a series of convolution layers with Instance Normalization and Leaky ReLU layers. 

- For the convolution layers, we specify ```4x4``` filters with an increasing number of filters for each layer. We also specify a stride of ```2``` and padding of the output.

![image](https://user-images.githubusercontent.com/59663734/162629181-f2d2d2e5-d158-4135-b55b-9d5728588ad7.png)

- For the leaky ReLU layers we have a negative slope of ```0.2```.

- For the final convolution layer, specify a **one** ```4x4``` filter with no padding.

![image](https://user-images.githubusercontent.com/59663734/162634863-98785fc5-96e4-4de1-943b-69f3a48afaa5.png)

```python
## critic model

class Critic(nn.Module):
  def __init__(self, d_dim=16):
    super(Critic, self).__init__()

    self.crit = nn.Sequential(
      # Conv2d: in_channels, out_channels, kernel_size, stride=1, padding=0
      ## New width and height: # (n+2*pad-ks)//stride +1
      ## we decrease size of image and increase number of channels

      #-- we start with image of 128x128x3
      nn.Conv2d(3, d_dim, 4, 2, 1), #(n+2*pad-ks)//stride +1 = (128+2*1-4)//2+1=64x64 (ch: 3 to 16) | 64x64x16
      nn.InstanceNorm2d(d_dim), 
      nn.LeakyReLU(0.2),

      nn.Conv2d(d_dim, d_dim*2, 4, 2, 1), ## 32x32 (ch: 16 to 32) | 32x32x32
      nn.InstanceNorm2d(d_dim*2), # Norm applied to previous layers
      nn.LeakyReLU(0.2),

      nn.Conv2d(d_dim*2, d_dim*4, 4, 2, 1), ## 16x16 (ch: 32 to 64) | 16x16x64
      nn.InstanceNorm2d(d_dim*4), 
      nn.LeakyReLU(0.2),
              
      nn.Conv2d(d_dim*4, d_dim*8, 4, 2, 1), ## 8x8 (ch: 64 to 128) | 8x8x128
      nn.InstanceNorm2d(d_dim*8), 
      nn.LeakyReLU(0.2),

      nn.Conv2d(d_dim*8, d_dim*16, 4, 2, 1), ## 4x4 (ch: 128 to 256) | 4x4x256
      nn.InstanceNorm2d(d_dim*16), 
      nn.LeakyReLU(0.2),

      nn.Conv2d(d_dim*16, 1, 4, 1, 0), #(n+2*pad-ks)//stride +1=(4+2*0-4)//1+1= 1X1 (ch: 256 to 1) | 1x1x1
      #-- we end with image of 1x1x1 - single output(real or fake)
    )


  def forward(self, image):
    # image: 128 x 3 x 128 x 128: batch x channels x width x height
    crit_pred = self.crit(image) # 128 x 1 x 1 x 1: batch x  channel x width x height | one single value for each 128 image in batch
    return crit_pred.view(len(crit_pred),-1) ## 128 x 1  
```

##### 2.7.3 The Gradient Penalty
The gradient penalty improves stability by penalizing gradients with large norm values. The lambda value controls the magnitude of the gradient penalty added to the discriminator loss. Recall that we need to create an interpolated image using real and fake images weighted by ```epsilon```. Then based on the gradient of the prediction of the critic on the interpolated image we will add a regularization term in our loss function.


```python
## gradient penalty calculation

def get_gp(real, fake, crit, epsilon, lambda=10):
  interpolated_images = real * epsilon + fake * (1-epsilon) # 128 x 3 x 128 x 128 | Linear Interpolation
  interpolated_scores = crit(interpolated_images) # 128 x 1 | prediction of critic

  # Analyze gradients if too large
  gradient = torch.autograd.grad(
      inputs = interpolated_images,
      outputs = interpolated_scores,
      grad_outputs=torch.ones_like(interpolated_scores),
      retain_graph=True,
      create_graph=True,
  )[0] # 128 x 3 x 128 x 128

  gradient = gradient.view(len(gradient), -1)   # 128 x 49152
  gradient_norm = gradient.norm(2, dim=1)  # L2 norm
  gp = lambda * ((gradient_norm-1)**2).mean()

  return gp
```


##### 2.7.4 Critic Training
We will now train the critic using the following steps:

1. Initialize **gradients** to ```0```.
2. Create ```noise``` using the **gen_noise** function.
3. Project and reshape noise and pass it in our **Generator** model to create a ```Fake``` image.
4. Get **predictions** on ```fake``` and ```real``` image.
5. Generate random **epsilon** and calculate ```gradient penalty```.
6. Calculate the critic ```loss``` using **gradient penalty**.
7. Use ```backpropagation``` to update our critic **parameters**.

```python
    '''Critic Training'''

    mean_crit_loss = 0
    for _ in range(crit_cycles):
      crit_opt.zero_grad()
      
      #--- Create Noise
      noise=gen_noise(cur_bs, z_dim)
      #---Create Fake Image from Noise
      fake = gen(noise)

      #--- Get prediction on fake and real image
      crit_fake_pred = crit(fake.detach())
      crit_real_pred = crit(real)

      #--- Calculate gradient penalty
      epsilon=torch.rand(len(real),1,1,1,device=device, requires_grad=True) # 128 x 1 x 1 x 1
      gp = get_gp(real, fake.detach(), crit, epsilon)

      #--- Calculate Loss
      crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp
      mean_crit_loss+=crit_loss.item() / crit_cycles

      #--- Backpropagation
      crit_loss.backward(retain_graph=True)
      #--- Update parameter of critic
      crit_opt.step()

    #--- Append Critic Loss
    crit_losses+=[mean_crit_loss]

```

##### 2.7.5 Generator Training
The training of the generator is much simpler:

1. Initialize **gradients** to ```0```.
2. Create ```noise``` **vector**.
3. Generate ```fake``` **image** from noise vector.
4. Get critic's ```prediction``` on **fake** image.
5. Calculate **generator's** ```loss```.
6. Use ```backpropagation``` to update generator's **parameters**.

```python
    '''Generator Training'''
    #--- Initialize Gradients to 0
    gen_opt.zero_grad()

    #---Create Noise Vector
    noise = gen_noise(cur_bs, z_dim)
    #---Create Fake image from Noise vector
    fake = gen(noise)

    #---Critic's prediction on fake image
    crit_fake_pred = crit(fake)

    #--- Calculate Generator Loss
    gen_loss = -crit_fake_pred.mean()

    #---Backpropagation
    gen_loss.backward()
    #--- Update generator's paramaters
    gen_opt.step()

    #--- Append Generator Loss
    gen_losses+=[gen_loss.item()]
```

##### 2.7.6 Training results
We being training our GAN with the following hyperparameters:

- Number of images: 10000
- Number of epochs: 50000
- Batch size: 128
- Number of steps per epoch: Number of epochs/Batch size = 50000/128 = 390.625
- Learning rate: 0.0001
- Dimension of noise vector: 200
- Optimizer: Adam
- Critic cycles: 5 (we train critic 5 times + 1 train of generator - so that critic is not overpowered by generator)

We plot the graph of the Generator loss and Critic loss w.r.t to the number of steps. Some important features of the graph are:

1. The critic loss (red) is initially positive (not clearly shown on the graph). This is because of the loss function of our critic:

<p align="center">
  <img src= "https://latex.codecogs.com/png.image?\dpi{110}\underset{C}{max}(\mathbb{E}\cdot&space;C(x))&space;-&space;(\mathbb{E}\cdot&space;C(G(z)))" title="https://latex.codecogs.com/png.image?\dpi{110}\underset{C}{max}(\mathbb{E}\cdot C(x)) - (\mathbb{E}\cdot C(G(z)))"/>
</p>

Remember that for the critic a larger score for real images results in a larger resulting loss. Hence. for the function above to be positive, ```E(C(x))``` should be positive, that is, the critic outputs a large score score for real images (it thinks the real images are fakes).

With time, the loss of the critic drops to become negative, i.e, it outputs a smaller score for real images which means it starts to correctly identify between reals and fakes.

2. The loss of the generator is positive because a larger score from the critic will result in a smaller loss for the generator, encouraging the critic to output larger scores for fake images.

3. The absolute value of the critic (average ```7```) is much **lower** than the absolute value of the generator (average ```22```). This is because we are training the critic 5 times more for everyt training of the generator. It allows the critic to not be overpowered by the generator. 

4. Unfortunately, the loss for both the critic and the generator does not approach **zero**. We observe that the loss of the generator approaches its minimum at about 6000 steps while the loss of the critic remains mainly constant.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/165896193-ce40cfa1-880f-4aad-9718-9287407dbe75.jpg" width="700" height="350" />
</p>



Below is the results of the training. The first ```200``` steps are just **noise** with no particular structure in the data. But with time, we can clearly see some facial features appearing in the noise at about ```800``` steps. At the end of ```6000``` steps, we successfully generate faces while not very high definition. 

https://user-images.githubusercontent.com/59663734/165896265-f9494889-6ab6-4958-914a-b00985b9d06f.mp4


##### 2.7.7 Testing
With our model saved, we will use the generator and scrape out the discriminator to generate new faces from noise. 

```python
#### Generate new faces
noise = gen_noise(batch_size, z_dim)
fake = gen(noise)
show(fake)
```

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/165916810-c286c8bf-116e-4c91-b21d-6f20d20f4cdf.png" width="650" height="300" />
</p>

Note that we are actually displaying 25 images at a time but the generator output 1 images at each step. Although the picture is highly pixelated, it will be hard to distinguish whether it is real or fake.

##### 2.7.8 Morphing
We can also interpolate between two points in the latent space to see how the image of a picture morph to become another one. Note that the AI has only been trained with 10000 images and since our loss function did not approach to zero as near as possible, the interpolation is quite rudimentary. Yet, it is still impressive to see the result.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/165918820-c3a2cb7e-5785-4a5c-86ab-123279358db3.png" width="800" height="250" />
</p>


### 3. Controllable and Conditional GAN

Although we are generating very good fake images from our WGAN, we do not really have a **control** of the type of faces to generate. For example, if I want to generate only ```women``` faces I would not be able to do that. What we have been doing is called ```unconditional generation```. What we want to achieve is ```Conditional``` generation, that is, we tell our model to generate different items we specify or condition and we adapt the training process so it actually does that. There is also ```Controllable``` generation where we figure out how to adapt the inputs to our model without changing the model itself. 

#### 3.1 Conditional GAN
We will now control the output and get examples from a particular class or for those examples to take on certain features. Below are the key differences between unconditional and conditional GAN:

1. With unconditional we get examples from random classes whereas with conditional gan we get examples from the classes we specify.
2. The training dataset is not labeled for unconditional whereas for conditional gan is should be labeled and the label are the different class we want. 

With unconditional generation, the generator needs a **noise** vector to produce random examples. For conditional generation, we also need a vector to tell the generator from which class the generated examples should come from. Usually this is a ```one-hot vector```, which means that there are ```zeros``` in every position except for one position corresponding to the class we want. In the example below, we specify a ```one```at Sphinx cat because that's the class that we want the generator to create images of. 

The noise vector is the one that adds ```randomness``` in the generation, similar to before; to let us produce a diverse set of examples. But now it's a diverse set **within** the certain class, **conditioned** on the certain class and **restricted** by the second class. The input to the generator in a conditional GAN is actually a ```concatenated vector``` of both the noise and the one-hot class information.

In the example below, we generate a Sphinx cat from one noise vector but when we change that noise vector while the class information stays the same, it produces another picture of a Sphinx cat.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/165928657-3b537065-1782-489e-af2b-7bdf9adf66a9.gif" />
</p>

The discriminator in a similar way will take the examples, but now it is paired with the ```class``` information as inputs to determine if the examples are either real or fake representations of that particular class.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/165930816-08c24be3-c709-452c-a5a1-66ffd7424016.gif" />
</p>



#### 3.2 Controllable GAN






# Conclusion

# References
1. https://www.youtube.com/watch?v=xkqflKC64IM&t=489s
2. https://www.youtube.com/watch?v=CDMVaQOvtxU
3. https://www.whichfaceisreal.com/
4. https://this-person-does-not-exist.com/en
5. https://www.youtube.com/watch?v=HHNESCbZqUg
6. https://machinelearningmastery.com/generative-adversarial-network-loss-functions/
7. https://www.analyticsvidhya.com/blog/2021/07/deep-understanding-of-discriminative-and-generative-models-in-machine-learning/
8. http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf
9. https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3
10. https://www.youtube.com/watch?v=z5UQyCESW64
11. https://machinelearningmastery.com/cross-entropy-for-machine-learning/
12. https://towardsdatascience.com/keywords-to-know-before-you-start-reading-papers-on-gans-8a08a665b40c#:~:text=Latent%20space%20is%20simply%20any,dataset%20it%20was%20trained%20on).
13. https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d
14. https://medium.com/swlh/how-i-would-explain-gans-from-scratch-to-a-5-year-old-part-1-ce6a6bccebbb
15. https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
16. https://developers.google.com/machine-learning/gan/loss
17. https://arxiv.org/abs/1701.07875
18. https://arxiv.org/abs/1704.00028
19. https://lilianweng.github.io/posts/2017-08-20-gan/
