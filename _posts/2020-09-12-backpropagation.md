---
layout: post
mathjax: true
title:  "Understanding Backpropagation of Neural Networks Using an Example with Step-by-step Derivation and Custom Implementations"
date:   2020-09-12 17:00:41 -0700
categories: github pages
author: Xiyun Song
comments: true
---

<p>A neural network model typically learns from iterative training processes to optimize values for its trainable parameters. The major steps in each iteration include forward pass, loss calculation, backward pass and parameter update. </p>
<ul>
<li>The forward pass is the process to compute outputs from input. </li>
<li>Compute loss that is simply a scalar number indicating the disparity betwen the model's predicted output and the ground truth. </li>
<li>The backward pass is the process to compute gradients of the loss with respect to (w.r.t.) each trainable parameter, which indicates how much the parameters need to change in the negative direction to minimize the loss. </li>
<li>Update the trainable parameters based on their gradients using a certain algorithm e.g. Adam so that the loss decreases.</li>
</ul> 

<p>Among these steps, the backward pass, which is typically done using backpropagation with chain rule to effectively compute gradients, is the most complex. Fortunately, we are required to define only the forward pass when building a neural network and popular deep learning frameworks such as PyTorch and Tensorflow will do backpropagation for us automatically, referred to as Autograd. While this brings convenience, it also causes confusion around backpropagation, the heart of every neural network. In fact, a clear understanding and hands-on experience with backpropagation is critical for anyone who would like to be an AI expert. Therefore, in this post, I would like to use a simple network example to demonstrate how to compute gradients using the chain rule and manually implement it. Hopefully, this could help those of us with some basic knowledge of neural network and calculus to gain a more solid understanding of it. </p>

<p>This post is organized in the following sections: </p>
<ul>
<li><a href="#_The_simple_network">The simple network model</a></li>
<li><a href="#_Derivation_of_the_gradients">Derivation of gradients using backpropagation chain rule</a></li>
<li><a href="#_Custom_implementations_and_validation">Custom implementations and validation </a></li>
<li><a href="#_Summary">Summary</a></li>
<li><a href="#_Extra">Extra</a></li>
</ul>

<h3><a name="_The_simple_network"></a>1. The simple network model</h3>  
<p>The example network model is illustrated in Figure 1. It consists of two layers (BatchNorm module and activator Sigmoid) followed by a mean square error (MSE) loss module. Let’s take a look at the forward pass (from bottom to top).</p>

<p align="center">
 <img src="{{ "/assets/images/Model1.png" | relative_url }}" style="border:solid; color:gray" width="350"> 
<br>Figure 1 Illustration of the example network. 
</p> 


<p><strong>Input \(\vec x\)</strong>: a multi-dimensional dataset of \({N_s}\) samples, referred to as a &ldquo;batch&rdquo;, and each sample contains \({N_c}\) feature channels of \({N_k} = {N_{height} } \times {N_{row} } \times {N_{col} }\)  elements. </p>
<div class="alert alert-secondary equation">
	<span>\(\vec x = \left\{ { {x_{s,c,k} } } \right\}\)	</span><span class="ref-num"> (1) </span>
</div>

<p>where \(s \in \left[ {1,\;{N_s} } \right]\), \(c \in \left[ {1,\;{N_c} } \right]\) and \(k \in \left[ {1,\;{N_k} } \right]\) are the indices along individual dimesions.</p>

<p><strong>BatchNorm layer</strong>: a module that was proposed by [Ioffe and Szegedy] and is widely used in neural networks to help make the training faster and more stable through normalization of the batch data by re-centering and re-scaling. BatchNorm is applied across all samples in the batch but for each feature channel separately. The blue color of the data in Figure 1 represents one channel. The output \(\vec y\) from BatchNorm is computed as </p>

<div class="alert alert-secondary equation">
	<span>\({y_{s,c,k} } = {w_c} \cdot \frac{ { {x_{s,c,k} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } + {\beta _c}\) </span><span class="ref-num"> (2)</span>
</div>

<p>where \({w_c}\) and \({\beta _c}\) are the trainable parameter, \({\mu _c}\) and \(\sigma _c^2\) are the mean and biased variance of the \(c\)-th channel of the given batch data, </p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({\mu _c} = \frac{1}{ { {N_s} \times {N_k} } }\mathop \sum \limits_{s = 1}^{ {N_s} } \mathop \sum \limits_{k = 1}^{ {N_k} } {x_{s,c,k} }\) </span><span class="ref-num"> (3) </span></p>
		
	<p class="equation"><span> \(\sigma _c^2 = \frac{1}{ { {N_s} \times {N_k} } }\mathop \sum \limits_{s = 1}^{ {N_s} } \mathop \sum \limits_{k = 1}^{ {N_k} } {\left( { {x_{s,c,k} } - {\mu _c} } \right)^2}\) </span><span class="ref-num"> (4)	</span></p>
</div>
	
	
<p>Please note that because all computation are cross the dimensions of \(s\) and \(k\), we can combine these dimensions into a single index \(i \in \left[ {1,{N_s} \times {N_k} } \right]\) and rewrite Equations (2) through (4) equivalently as the following for simplicity:</p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({y_{c,i} } = {w_c} \cdot \frac{ { {x_{c,i} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } + {\beta _c}\) </span><span class="ref-num">(5)</span></p>

	<p class="equation"><span> \({\mu _c} = \frac{1}{ { {N_i} } }\mathop \sum \limits_{j = 1}^{ {N_i} } {x_{c,j} }\) </span><span class="ref-num">(6)</span></p>
	<p class="equation">
	<span> \(\sigma _c^2 = \frac{1}{ { {N_i} } }\mathop \sum \limits_{j = 1}^{ {N_i} } {\left( { {x_{c,j} } - {\mu _c} } \right)^2}\)	</span><span class="ref-num">(7) </span></p>
</div>

<p>where \({N_i} = {N_s} \times {N_k}\) and \(j \in \left[ {1,{N_i} } \right]\).</p>

<p>We also want to mention that BatchNorm is selected to be included in this example because while its forward computing is much simpler than some other modules such as ConvNd and ConvTransNd, its backward computing for gradients is surprisingly more complex than those modules. By going through the harder example of BatchNorm, we hope it helps with the other easier cases. </p>

<p><strong>Sigmoid layer</strong>: an activator function whose output \(\vec z\) is defined as </p>

<div class="alert alert-secondary equation">
	<span> \({z_{c,i} } = \frac{1}{ {1 + {e^{ - {y_{c,i} } } } } }\)</span><span class="ref-num">(8)</span>
</div>

<p>\(\vec z\) is also the output of the network. Sigmoid is applied to each element in each channel of each sample independently. The index of \(c\) is kept in Equation (8) for convenience of gradient derivation using backpropagation later. So far the equations (5) and (8) form the forward pass of the network.</p>

<p><strong>MSE Loss function</strong>: the scalar MSE loss, \(L\), is computed as </p>

<div class="alert alert-secondary equation">
	<span> \(L = \frac{1}{N}\mathop \sum \limits_{c = 1}^{ {N_c} } \mathop \sum \limits_{i = 1}^{ {N_i} } {\left( { {z_{c,i} } - {t_{c,i} } } \right)^2}\)</span><span class="ref-num">(9)</span>
</div>

<p>where \(N = {N_c} \times {N_i}\). In other words, the loss is the mean of the squared error of all elements in all channels of all samples between the network output \(\vec z\) and the target dataset \(\vec t\).</p>

<p>With all components of this example network being defined, the main goal here is to compute the gradients of the loss \(L\) w.r.t. the trainable parameters \(\vec w\) and \(\vec \beta \) and the input \(\vec x\) : \(\frac{ {\partial L} }{ {\partial \vec w} }\), \(\frac{ {\partial L} }{ {\partial \vec \beta } }\), and \(\frac{ {\partial L} }{ {\partial \vec x} }\), respectively. You might have a question: since \(\vec x\) is an input instead of a trainable parameter, why bother to compute the partial derivative \(\frac{ {\partial L} }{ {\partial \vec x} }\)? The answer is that BatchNorm in a network typically takes the output from other layers (e.g. Conv2d, which is not included in this network for simplicity) as its input \(\vec x\), and calculation of gradients in those layers requires \(\frac{ {\partial L} }{ {\partial \vec x} }\) being available. </p>

<p>BTW, this post is not focused on parameter update from the gradients, so that part is not included in this post. We can write separate posts about that too.    </p>

<h3><a name="_Derivation_of_the_gradients"></a>2. Derivation of the gradients</h3>
<h4>2.1 The chain rule</h4>
<p>We will follow the chain rule to do backpropagation to compute gradients. Since there is a huge amount of online resources available talking about the chain rule, we just summarize its main idea here. Given a function \(L\left( { {x_1},{x_2}, \ldots {x_N} } \right)\) as</p>

<div class="alert alert-secondary equation">
	<span> \(L\left( { {x_1}, \ldots {x_N} } \right) = L\left( { {f_1}\left( { {x_1}, \ldots {x_N} } \right),{f_2}\left( { {x_1}, \ldots {x_N} } \right), \ldots ,{f_M}\left( { {x_1}, \ldots {x_N} } \right)} \right)\)</span><span class="ref-num">(10)</span>
</div>

<p>Then the gradient of \(L\) w.r.t \({x_i}\) can be computed as </p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {x_i} } } = \frac{ {\partial L} }{ {\partial {f_1} } }\frac{ {\partial {f_1} } }{ {\partial {x_i} } } + \frac{ {\partial L} }{ {\partial {f_2} } }\frac{ {\partial {f_2} } }{ {\partial {x_i} } } +\cdots + \frac{ {\partial L} }{ {\partial {f_M} } }\frac{ {\partial {f_M} } }{ {\partial {x_i} } } = \mathop \sum \limits_{m = 1}^M \frac{ {\partial L} }{ {\partial {f_m} } }\frac{ {\partial {f_m} } }{ {\partial {x_i} } }\)</span><span class="ref-num">(11)</span>
</div>

<p>Equation (11) can be understood from two perspectives:</p>
<ul>
<li>Summation means that all possible paths through which \({x_i}\) contributes to \(L\) should be included</li>
<li>Product means that, along each path \(m\), the output gradient equals the upstream passed in, \(\frac{ {\partial L} }{ {\partial {f_m} } }\), times the local gradient, \(\frac{ {\partial {f_m} } }{ {\partial {x_i} } }\). </li>
</ul>
<br>

<h4>2.2 Dimensions of the gradients</h4>
<p>Please note that, the loss function in a neural network gives a scalar value output loss. The gradient of the loss w.r.t. any trainable parameter or input variable should have the same dimension as that parameter or variable, whether it is another scalar, or a 1-D vector, or N-D array. </p>
 
<p>For example, the trainable parameter \(\vec w = \left( { {w_1},{w_2}, \ldots {w_{ {N_c} } } } \right)\)  of the Batch Norm module is a 1-D vector containing \({N_c}\) elements with one element corresponding to one channel; therefore, the gradient \(\frac{ {\partial L} }{ {\partial \vec w} }\) should also be a 1-D vector containing \({N_c}\) elements, \(\frac{ {\partial L} }{ {\partial \vec w} } = \left( {\frac{ {\partial L} }{ {\partial {w_1} } },\frac{ {\partial L} }{ {\partial {w_2} } }, \ldots \frac{ {\partial L} }{ {\partial {w_{ {N_c} } } } } } \right)\). </p>
 
<h4>2.3 Derivation of the gradient \(\frac{ {\partial L} }{ {\partial \vec w} }\)  </h4> 
<p>Let’s first derive the upstream gradients using the chain rule to derive \(\frac{ {\partial L} }{ {\partial \vec w} }\) or element-wise \(\frac{ {\partial L} }{ {\partial {w_c} } }\).</p>
<p>From Equation (9) for the MSE loss function, we have the partial derivative</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {z_{c,i} } } } = \frac{2}{N}\left( { {z_{c,i} } - {t_{c,i} } } \right)\)</span><span class="ref-num">(12)</span>
</div>
	
<p>From Equation (8), we have the local partial derivative of the sigmoid function</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial {z_{c,i} } } }{ {\partial {y_{c,i} } } } = \frac{ { {e^{ - {y_{c,i} } } } } }{ { { {\left( {1 + {e^{ - {y_{c,i} } } } } \right)}^2} } } = \frac{1}{ {\left( {1 + {e^{ - {y_{c,i} } } } } \right)} } \cdot \frac{ {1 - \left( {1 - {e^{ - {y_{c,i} } } } } \right)} }{ {\left( {1 + {e^{ - {y_{c,i} } } } } \right)} } = {z_{c,i} }\left( {1 - {z_{c,i} } } \right)\)</span><span class="ref-num">(13)</span>
</div>

<p>Note that, there is only one path from \({y_{c,i} }\) to \(L\), which is via \({z_{c,i} }\). Therefore, the gradient of \(L\) w.r.t. \({y_{c,i} }\) is simply the product of \(\frac{ {\partial L} }{ {\partial {z_{c,i} } } }\) and \(\frac{ {\partial {z_{c,i} } } }{ {\partial {y_{c,i} } } }\) without summation.</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {y_{c,i} } } } = \frac{ {\partial L} }{ {\partial {z_{c,i} } } }\cdot\frac{ {\partial {z_{c,i} } } }{ {\partial {y_{c,i} } } } = \frac{ {\partial L} }{ {\partial {z_{c,i} } } }\cdot {z_{c,i} }\left( {1 - {z_{c,i} } } \right)\)</span><span class="ref-num">(14)</span>
</div>

<p>This is also the upstream gradient for the BatchNorm layer. The local gradient for \(\vec w\) can be derived from Equation (5) </p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial {y_{c,i} } } }{ {\partial {w_c} } } = \frac{ { {x_{c,i} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } = \frac{ { {y_{c,i} } - {\beta _c} } }{ { {w_c} } }\)</span><span class="ref-num">(15)</span>
</div>

<p>Now, using the chain rule, we take the product of the upstream gradient from Equation (14) and the local gradient from Equation (15), and sum all paths together to get </p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {w_c} } } = \mathop \sum \limits_{i = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ {\partial {y_{c,i} } } }{ {\partial {w_c} } } = \mathop \sum \limits_{i = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ { {y_{c,i} } - {\beta _c} } }{ { {w_c} } }\)</span><span class="ref-num">(16)</span>
</div>

<p>The upstream gradient terms in Equations (14) and (16) are not substituted with their expanded format because their values have been computed and are available from upstream calculation, so there is no need to re-compute them every time they are used. That’s why the chain rule is an effective method for backpropagation.</p>

<h4>2.4 Derivation of gradient \(\frac{ {\partial L} }{ {\partial \vec \beta } }\)  </h4> 
<p>Similarly, the local gradient for \(\vec \beta \) can be derived from Equation (5) for BatchNorm</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial {y_{c,i} } } }{ {\partial {\beta _c} } } = 1\)</span><span class="ref-num">(17)</span>
</div>

<p>Now using the chain rule, we have</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {\beta _c} } } = \mathop \sum \limits_{i = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ {\partial {y_{c,i} } } }{ {\partial {\beta _c} } } = \mathop \sum \limits_{i = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\cdot1 = \mathop \sum \limits_{i = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\)</span><span class="ref-num">(18)</span>
</div>

<h4>2.5 Derivation of gradient \(\frac{ {\partial L} }{ {\partial \vec x} }\)   </h4>
<p>Derivation of gradient w.r.t \(\vec x\) is more complicated than \(\vec w\) and \(\vec \beta \) because each \({x_{c,i} }\) directly contributes to \({y_{c,i} }\) and also indirectly contributes to other \({y_{c,k} }\) elements via \({\mu _c}\left( { {x_{c,1} }, \ldots ,{x_{c,{N_i} } } } \right)\) and \({\sigma ^2}_c\left( { {x_{c,1} }, \ldots ,{x_{c,{N_i} } } } \right)\), as shown in Figure 2. To help understand the underneath logic, we will demonstrate two different ways to solve this challenge and achieve the same solution. </p>

<p align="center">
 <img src="{{ "/assets/images/Model2.png" | relative_url }}" style="border:solid; color:gray" width="350">
 <div class="thecap" style="text-align:center">Figure 2 Illustration of the contribution paths</div>
</p> 


<h5>2.5.1 Solution 1 for derivation of gradient \(\frac{ {\partial L} }{ {\partial \vec x} }\)   </h5>
Using the chain rule, we have

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {x_{c,i} } } } = \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ {\partial {y_{c,k} } } }{ {\partial {x_{c,i} } } } = \frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } }\frac{ {\partial {y_{c,{\rm{i} } } } } }{ {\partial {x_{c,i} } } } + \mathop \sum \limits_{k \ne i} \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ {\partial {y_{c,k} } } }{ {\partial {x_{c,i} } } }\)</span><span class="ref-num">(19)</span>
</div>

<p>Based on Equation (5), for \(k = i\), we have</p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial {y_{c,i} } } }{ {\partial {x_{c,i} } } } = {w_c}\frac{ {\sqrt {\sigma _c^2} \cdot \frac{\partial }{ {\partial {x_{c,i} } } }\left( { {x_{c,i} } - {\mu _c} } \right) - \left( { {x_{c,i} } - {\mu _c} } \right)\frac{\partial }{ {\partial {x_{c,i} } } }\sqrt {\sigma _c^2} } }{ {\sigma _c^2} }\)</span></p>

		<p><span> \( = {w_c}\frac{ {\frac{ { {N_i} - 1} }{ { {N_i} } }\sqrt {\sigma _c^2}  - \left( { {x_{c,i} } - {\mu _c} } \right) \cdot\frac{1}{2} \cdot \frac{1}{ {\sqrt {\sigma _c^2} } } \cdot \frac{ {2\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i} } } } }{ {\sigma _c^2} }\)</span></p>	

		<span>\( = {w_c}\frac{ {\left( { {N_i} - 1} \right)\sigma _c^2 - { {\left( { {x_{c,i} } - {\mu _c} } \right)}^2} } }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\)</span>
	</span>
	<span class="ref-num">(20)</span>
</div>

<p>For \(k \ne i\), we have</p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial {y_{c,k} } } }{ {\partial {x_{c,i} } } } = {w_c}\frac{ {\sqrt {\sigma _c^2} \frac{\partial }{ {\partial {x_{c,i} } } }\left( { {x_{c,k} } - {\mu _c} } \right) - \left( { {x_{c,k} } - {\mu _c} } \right)\frac{\partial }{ {\partial {x_{c,i} } } }\sqrt {\sigma _c^2} } }{ {\sigma _c^2} }\)</span></p>	

		<p><span> \( = {w_c}\frac{ {\sqrt {\sigma _c^2} \left( { - \frac{1}{ { {N_i} } } } \right) - \left( { {x_{c,i} } - {\mu _c} } \right) \cdot \frac{1}{2} \cdot \frac{1}{ {\sqrt {\sigma _c^2} } } \cdot \frac{ {2\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i} } } } }{ {\sigma _c^2} }\)</span></p>

		<span>\( =  - {w_c}\frac{ {\sigma _c^2 + \left( { {x_{c,i} } - {\mu _c} } \right)\left( { {x_{c,k} } - {\mu _c} } \right)} }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\)</span>
	</span>
	<span class="ref-num">(21)</span>
</div>

<p>Substitute Equations(20) and (21) into (19), we have</p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial {x_{c,i} } } } = \frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } }{w_c}\frac{ {\left( { {N_i} - 1} \right)\sigma _c^2 - { {\left( { {x_{c,i} } - {\mu _c} } \right)}^2} } }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } } - \mathop \sum \limits_{k \ne i} \frac{ {\partial L} }{ {\partial {y_{c,k} } } }{w_c}\frac{ {\sigma _c^2 + \left( { {x_{c,i} } - {\mu _c} } \right)\left( { {x_{c,k} } - {\mu _c} } \right)} }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\)	</span></p>

		<p><span> \( = \left( {1 - \frac{1}{ { {N_i} } } } \right)\frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { { {\left( { {x_{c,i} } - {\mu _c} } \right)}^2} } }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k \ne i} \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \frac{ { {w_c}\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\mathop \sum \limits_{k \ne i} \left( { {x_{c,k} } - {\mu _c} } \right)\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\)	</span></p>

		<p><span> \( = \frac{ { {w_c} } }{ {\sqrt { {\sigma ^2}_c} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c}{ {\left( { {x_{c,i} } - {\mu _c} } \right)}^2} } }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k \ne {\rm{i} } } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \frac{ { {w_c}\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3} } }\mathop \sum \limits_{k \ne {\rm{i} } } \left( { {x_{c,k} } - {\mu _c} } \right)\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\)	</span></p>

		<p><span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \frac{ { {w_c}\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i}{ {\sqrt {\sigma _c^2} }^3}{w_c} } }\mathop \sum \limits_{k = 1}^{ {N_i} } {w_c}\left( { {x_{c,k} } - {\mu _c} } \right)\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\)	</span></p>

		<span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,{\rm{i} } } } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \frac{ { {y_{c,i} } - {\beta _c} } }{ { {N_i}\sqrt {\sigma _c^2} {w_c} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \left( { {y_{c,k} } - {\beta _c} } \right)\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\)</span>
	</span>
	<span class="ref-num">(22)</span>
</div>

<h5>2.5.2 Solution 2 for derivation of gradient \(\frac{ {\partial L} }{ {\partial \vec x} }\)   </h5>
From Figure 2, we see that, given \({x_{c,i} }\), it has only 3 direct contribution paths to \({y_{c,i} }\), \({\mu _c}\) and \({\sigma ^2}_c\), as highlighted using bold red arrow. Therefore using the chain rule from this perspective, we have the second solution:

<div class="alert alert-secondary equation">
	<span> \(\frac{ {\partial L} }{ {\partial {x_{c,i} } } } = \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ {\partial {y_{c,i} } } }{ {\partial {x_{c,i} } } } + \frac{ {\partial L} }{ {\partial {\mu _c} } }\frac{ {\partial {\mu _c} } }{ {\partial {x_{c,i} } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\frac{ {\partial \sigma _c^2} }{ {\partial {x_{c,i} } } }\)</span><span class="ref-num">(23)</span>
</div>

<p>The 1st item  \(\frac{ {\partial L} }{ {\partial {y_{c,i} } } }\) is the upstream gradient and available from Equation (14). Let’s derive the rest components one by one. </p>
<p>The 2nd item \(\frac{ {\partial {y_{c,i} } } }{ {\partial {x_{c,i} } } }\) can be derived from Equation (5) </p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ {\partial {y_{c,i} } } }{ {\partial {x_{c,i} } } } = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\)</span><span class="ref-num">(24)</span>
</div>

<p>Now, let’s derive the 3rd item \(\frac{ {\partial L} }{ {\partial {\mu _c} } }\). Note that, \({\mu _c}\) has direct contribution paths to both \(\vec y\) and \(\sigma _c^2\), so we have </p>

<div class="alert alert-secondary equation">
<span>
	<p><span> \(\frac{ {\partial L} }{ {\partial {\mu _c} } } = {\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ {\partial {y_{c,k} } } }{ {\partial {\mu _c} } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\frac{ {\partial \sigma _c^2} }{ {\partial {\mu _c} } }\)</span></p>

	<p><span> \( = {\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ {\partial {y_{c,k} } } }{ {\partial {\mu _c} } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\frac{\partial }{ {\partial {\mu _c} } }\left( {\frac{1}{N}\mathop \sum \limits_{k = 1}^{ {N_i} } { {\left( { {x_k} - {\mu _c} } \right)}^2} } \right)\)</span></p>

	<p><span> \( = { - \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\cdot\frac{1}{ { {N_i} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{\partial }{ {\partial {\mu _c} } }{\left( { {x_k} - {\mu _c} } \right)^2}\)</span></p>

	<p><span> \( = { - \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\cdot\frac{1}{ { {N_i} } }\mathop \sum \limits_{k = 1}^{ {N_i} } 2\left( { {\mu _c} - {x_k} } \right)\)</span></p>

	<p><span> \( = { - \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\cdot2\left( {\frac{1}{ { {N_i} } }\mathop \sum \limits_{k = 1}^{ {N_i} } {\mu _c} - \frac{1}{ { {N_i} } }\mathop \sum \limits_{k = 1}^{ {N_i} } {x_k} } \right)\)</span></p>

	<p><span> \( = { - \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\cdot2\left( { {\mu _c} - {\mu _c} } \right)\)</span></p>

	<span> \( = { - \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } } + 0\)</span>
	</span>
	<span class="ref-num">(25)</span>
</div>

<p>BTW, from the above derivation, we learned that \(\frac{ {\partial \sigma _c^2} }{ {\partial {\mu _c} } } = 0\).</p>
<p>Let’s move on to the 4th item \(\frac{ {\partial {\mu _c} } }{ {\partial {x_{c,i} } } }\). Because \({\mu _c} = \frac{1}{ { {N_i} } }\mathop \sum \limits_{i = 1}^{ {N_i} } {x_{c,i} }\), it’s easy to have</p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ {\partial {\mu _c} } }{ {\partial {x_{c,i} } } } = \frac{1}{ { {N_i} } }\)</span><span class="ref-num">(26)</span>
</div>

<p>Next, for the 5th item \(\frac{ {\partial L} }{ {\partial \sigma _c^2} }\). </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial \sigma _c^2} } = \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{ {\partial {y_{c,k} } } }{ {\partial \sigma _c^2} }\)</span></p>

		<p><span> \( = \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\frac{\partial }{ {\partial \sigma _c^2} }\left( { {w_c}\frac{ { {x_{c,k} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } + {\beta _c} } \right)\)</span></p>

		<p><span> \( = \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\left( { {x_{c,k} } - {\mu _c} } \right)\frac{\partial }{ {\partial \sigma _c^2} }\left( {\frac{1}{ {\sqrt {\sigma _c^2} } } } \right)\)</span></p>

		<p><span> \( = \mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\left( { {x_{c,k} } - {\mu _c} } \right)\left( { - \frac{1}{2}\cdot\frac{1}{ { { {\left( {\sqrt {\sigma _c^2} } \right)}^3} } } } \right)\)</span></p>

		<span> \( =  - \frac{1}{2}\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\frac{ { {x_{c,k} } - {\mu _c} } }{ { { {\left( {\sqrt {\sigma _c^2} } \right)}^3} } }\)</span>
	</span>
	<span class="ref-num">(27)</span>
</div>


<p>For the 6th and also last item \(\frac{ {\partial \sigma _c^2} }{ {\partial {x_{c,i} } } }\), we have</p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ {\partial \sigma _c^2} }{ {\partial {x_{c,i} } } } = \frac{\partial }{ {\partial {x_{c,i} } } }\left( {\frac{1}{N}\mathop \sum \limits_{k = 1}^{ {N_i} } { {\left( { {x_{c,k} } - {\mu _c} } \right)}^2} } \right) = \frac{ {2\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i} } }\)</span><span class="ref-num">(28)</span>
</div>

<p>Derivation of Equation (28) used the intermediate result of \(\frac{ {\partial \sigma _c^2} }{ {\partial {\mu _c} } } = 0\) from Equation (25). </p>
<p>Now, substituting these derivations back into Equation (23), we have</p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial {x_{c,i} } } } = \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ {\partial {y_{c,i} } } }{ {\partial {x_{c,i} } } } + \frac{ {\partial L} }{ {\partial {\mu _c} } }\frac{ {\partial {\mu _c} } }{ {\partial {x_{c,i} } } } + \frac{ {\partial L} }{ {\partial \sigma _c^2} }\frac{ {\partial \sigma _c^2} }{ {\partial {x_{c,i} } } }\)</span></p>

		<p><span> \( = \frac{ {\partial L} }{ {\partial {y_{c,i} } } }\frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } } + \left( { - \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } } \right)\frac{1}{ { {N_i} } } + \left( { - \frac{1}{2}\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\frac{ { {x_{c,k} } - {\mu _c} } }{ { { {\left( {\sqrt {\sigma _c^2} } \right)}^3} } } } \right)\cdot\frac{ {2\left( { {x_{c,i} } - {\mu _c} } \right)} }{ { {N_i} } }\)	</span></p>

		<p><span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,i} } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \left( {\frac{1}{ { {N_i}\sqrt {\sigma _c^2} } }\frac{ { {x_{c,i} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \left( {\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\frac{ { {x_{c,k} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } } \right)} \right)\)</span></p>

		<p><span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,i} } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \left( {\frac{1}{ { {N_i}\sqrt {\sigma _c^2} } }\cdot\frac{ { {w_c} } }{ { {w_c} } }\cdot\frac{ { {x_{c,i} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \left( {\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot{w_c}\frac{ { {x_{c,k} } - {\mu _c} } }{ {\sqrt {\sigma _c^2} } } } \right)} \right)\)	</span></p>

		<p><span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,i} } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \left( {\frac{ { {y_{c,i} } - {\beta _c} } }{ { {N_i}{w_c}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \left( {\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\cdot\left( { {y_{c,k} } - {\beta _c} } \right)} \right)} \right)\)</span></p>

		<span> \( = \frac{ { {w_c} } }{ {\sqrt {\sigma _c^2} } }\frac{ {\partial L} }{ {\partial {y_{c,i} } } } - \frac{ { {w_c} } }{ { {N_i}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \frac{ {\partial L} }{ {\partial {y_{c,k} } } } - \frac{ { {y_{c,i} } - {\beta _c} } }{ { {N_i}{w_c}\sqrt {\sigma _c^2} } }\mathop \sum \limits_{k = 1}^{ {N_i} } \left( { {y_{c,k} } - {\beta _c} } \right)\frac{ {\partial L} }{ {\partial {y_{c,k} } } }\)</span>
	</span>
	<span class="ref-num">(29)</span>
</div>

<p>Equation (29) is identical to Equation (22): two solutions, the same answer. We took this effort to demonstrate that a problem can be solved using different solutions. Derivation is done! </p>

<h3><a name="_Custom_implementations_and_validation"></a>3. Custom implementations and validation  </h3>
<p>Two separate implementations of the forward pass, loss calculation, and backpropagation are demonstrated in this post. The 1st one uses PyTorch and the 2nd one uses Numpy. The functions and corresponding equations are summarized in Table 1. </p>

<table>
  <tr>
    <th>Functions</th>
    <th>Equations</th>
  </tr>
  <tr>
    <td>BatchNorm Forward</td>
    <td>(5)</td>
  </tr>
  <tr>
    <td>Sigmoid Forward</td>
    <td>(8)</td>
  </tr>
  <tr>
    <td>MSE Loss</td>
    <td>(9)</td>
  </tr>
  <tr>
    <td>MSE Gradient</td>
    <td>(12)</td>
  </tr>
  <tr>
    <td>Sigmoid Gradient</td>
    <td>(14)</td>
  </tr>
  <tr>
    <td>BatchNorm Gradients</td>
    <td>(16), (18), (22)</td>
  </tr>
</table>

<p>While the Numpy version is quite self-explanatory, it is worth mentioning that the PyTorch version of our custom autograd functions of BatchNorm, Sigmoid and MSELoss are implemented by subclassing torch.autograd.Function, respectively. The forward() and backward() functions are overridden based on the equations above and will be executed when the forward pass and backpropagation are triggered. Please also note that the input list of upstream gradients of the backward() function must match the output list of the forward() function, and the output gradient list of the backward() function must match the input list of the forward() function. In addition, the input, output and intermediate data of the forward() function can be cached using the save_for_backward() function so that they can be used in the backward() function without re-calculation.  </p>

<p>A small constant number \(\epsilon = 1E-5\) is added to \(\sigma _c^2\) in order to avoid divide-by-zero exception just in case, just as done by the PyTorch built-in BatchNorm. </p>

<p>To serve as a reference for comparison, the same network was also implemented using PyTorch’s built-in modules of BatchNorm, Sigmoid and MSELoss. The network output, loss, gradients of the loss w.r.t. \(\vec w\), \(\vec \beta \) and \(\vec x\) from the custom implementations were compared to the reference and the results matched. </p>

<p>You can find <a href="https://github.com/coolgpu/backpropagation_w_example/blob/master/src/batchnorm_sigmoid_mse_network.py">the source code on GitHub</a>. If you like, you can also implement using Tensorflow as well to gain hands-on experience. </p> 


<h3><a name="_Summary"></a>4. Summary </h3>
<p>In this post, we used a simple BatchNorm-Sigmoid-MSELoss network to demonstrate how to use the chain rule to derive gradients in backpropagation and how to implement the custom autograd functions. We hope that, by going over this example, it can help obtain a deeper understanding of the fundamentals of neural networks, especially about backpropagation. </p>

<h3><a name="_Extra"></a>5. Extra</h3> 
<p>In the derivation above we found that the partial derivatives of the biased variance \({\sigma ^2}\) w.r.t to the input \(\vec x\) and mean \(\mu \) are \(\frac{ {\partial {\sigma ^2} } }{ {\partial {x_i} } } = \frac{ {2\left( { {x_i} - \mu } \right)} }{N}\) and \(\frac{ {\partial {\sigma ^2} } }{ {\partial \mu } } = 0\), respectively. If it sounds a little bit surprising to see these results and especially \(\frac{ {\partial {\sigma ^2} } }{ {\partial \mu } } = 0\), you can write a few lines of Python codes to verify it, just as below.</p>
<pre class="pre-scrollable">
	<code class="python">
		import torch
		
		x = torch.rand(10, requires_grad=True, dtype=torch.float64)
		m = x.mean()
		m.retain_grad()
		v = ((x-m)**2).sum()/x.numel()
		v.backward()
		manual_grad_dvdx = (x-m)*2/x.numel()
		print('dv/dm  = ', m.grad)
		print('torch  dv/dx= ', x.grad.detach().numpy())
		print('manual dv/dx= ', manual_grad_dvdx.detach().numpy())
	</code>
</pre>

# <!-- {% include disqus_comments.html %} -->





