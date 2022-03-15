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

Another interesting application of GAN is ```Deepfake```. Haven't you even wondered what the movie ```American Psycho``` would look like if we had ```Tom Cruise``` as the protagonist instead of ```Christian Bale```. Here's a preview:


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


#### 1.2 Generative Models
In ```generative models``` which is a class of models for ```unsupervised learning``` where given training data our goal is to try and generate ```new samples``` from the same distribution.







# Conclusion

# References
1. https://www.youtube.com/watch?v=xkqflKC64IM&t=489s
2. https://www.youtube.com/watch?v=CDMVaQOvtxU
3. https://www.whichfaceisreal.com/
4. https://this-person-does-not-exist.com/en
5. 
6. 
