# Generating Design Ideas from Keywords

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

    - ddd
3. Conditional GAN and Controllable Generation


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
  <img src= "https://user-images.githubusercontent.com/59663734/159412761-baaf867f-54ba-4e8a-9a49-6ba6ba126745.png" width="800" height="270"/>
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

4. The discriminator output <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" /> which is in the range of ```0``` to ```1``` will be used to compute a ```cost function``` that basically looks at how far the examples produced by the generator are being considered real by the discriminator because the generator wants this to seem as real as possible. That is, how good is the performance of the generator?

5. The generator wants <img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{Y}_{d}" /> to be as close to ```1```, meaning ```real``` as possible. Whereas, the discriminator is trying to get this to be ```0``` - ```fake```.

6. The cost function uses the difference between these two to then update the parameters of the generator using backpropagation. It gets to improve over time and know which direction to move it's parameters to generate something that looks more real and will fool the discriminator.

7. The difference between the output of the discriminator and the value ```1``` is going to be a smaller and smaller and the loss is going to be smaller and smaller. As such, the performance of the generator is going to keep on improving.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/159430976-0dbff385-2417-4ed7-a2df-bb1bd0e3cddf.png" width="800" height="250"/>
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

# Conclusion

# References
1. https://www.youtube.com/watch?v=xkqflKC64IM&t=489s
2. https://www.youtube.com/watch?v=CDMVaQOvtxU
3. https://www.whichfaceisreal.com/
4. https://this-person-does-not-exist.com/en
5. https://www.youtube.com/watch?v=HHNESCbZqUg
6. https://www.analyticsvidhya.com/blog/2021/07/deep-understanding-of-discriminative-and-generative-models-in-machine-learning/
7. http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf
8. https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3
9. https://www.youtube.com/watch?v=z5UQyCESW64
