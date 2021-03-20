---
layout: post
mathjax: true
title:  "Understanding Artificial Neural Networks with Hands-on Experience - Part 4. Optimization and Optimizers with Custom Implementations and A Case Study"
date:   2021-03-03 21:00:00 -0700
categories: github pages
author: Xiyun Song, PhD
---

<p>This post is the 4<sup>th</sup> part of the series.</p>


<ul>
	<li><a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2020/09/22/matrixmultiplication.html">Matrix Multiplication, Its Gradients and Custom Implementations</a></li>
	<li><a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html">Convolution, Its Gradients and Custom Implementations </a></li>
	<li><a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2021/02/18/transposed_convolution.html">Transposed Convolution and Custom Implementations </a></li>
	<li><a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2021/03/04/optimization.html">Optimization and optimizers with Custom Implementations and A Case Study (this post)</a> </li>
	<li><a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2021/03/20/learningrate_schedulers.html">Hyper-Parameter Learning Rate and Schedulers (this post)</a></li>
</ul>

<p>In the previous posts, we have been continuously focused on gradients and backpropagation, which is the most important and also most difficult part of understanding neural network principles. With that being covered, we are moving to another important topic – optimization. More specifically, we will talk about fundamentals of optimization and several widely used optimizers, and then use a case study to demonstrate how to implement and use them in a network. For your convenience, this post is organized into the following sections. </p>

<ul>
	<li><a href="#_Optimization">Optimization in neural networks</a></li>
	<li><a href="#_Optimizers">Widely used optimizers</a></li>
    <ul>
        <li><a href="#_GD">Gradient descent algorithm </a></li>
        <li><a href="#_SGD">Stochastic gradient descent algorithm </a></li>
        <li><a href="#_Momentum">Gradient descent with momentum algorithm </a></li>
        <li><a href="#_RMSprop">RMSprop optimizer algorithm</a></li>
        <li><a href="#_Adam">Adam optimizer algorithm  </a></li>	
    </ul>
	<li><a href="#_CaseStudy">A case study – Regression of Earth-to-Mars transfer orbit</a></li>
	<li><a href="#_Implementations">Implementations of the network and optimizers </a></li>
    <li><a href="#_HyperParameters">Hyper-parameters and learn rate scheduler*</a></li>
    <li><a href="#_Summary">Summary</a></li>
	<li><a href="#_References">References</a></li>
</ul>

<h3><a name="_Optimization"></a><span style="color:darkblue">1. Optimization in neural networks</span></h3> 

<p>Artificial neural networks are really about training learnable parameters, but how? Figure 1 shows typical steps in an iteration of the training process.</p>

<ul>
	<li><strong>Step 1 (forward-propagation)</strong>: Take a new batch of input data, run it through the entire network with current learnable parameters, and compute the output.</li>
	<li><strong>Step 2 (loss)</strong>: Compute loss using the pre-defined loss function. The whole point of loss is to quantify the errors of the output from the network model against the labels (ground truth) and how well the so-far learnt parameters are doing on the training set. The loss function tells what types of errors you care about and what types of constraints you want to apply to the training. We can have a separate talk about this topic.</li>
	<li><strong>Step 3 (back-propagation)</strong>: Use the chain rule to compute gradients of the loss with respect to every learnable parameter, for example, the weights and bias used in convolution, transposed convolution, batch-norm as we already covered before.</li>
	<li><strong>Step 4 (optimization)</strong>: based on the gradients, apply the optimization algorithms to change the learnable parameters a little bit so that the new set of parameters will lead to a decrease in loss (or errors).</li>
</ul>


<p align="center">
 <img src="{{ "/assets/images/Part4_Fig1_Four_steps_of_networks1.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 1 Illustration of the 4 core steps in one iteration of artificial neural networks training. 
</p> 

<p>Repeat Steps 1 through 4 for another batch, then another one. Eventually, you can reach the minimum of the loss function and it means you have optimized the network. This process is also referred to as minimization of the loss function. </p>

<h3><a name="_Optimizers"></a><span style="color:darkblue">2. Widely used optimizers</span></h3> 

<p>Various algorithms have been developed for the minimization task. In this section, we will discuss 5 popular optimizers (namely <strong>Gradient descent</strong>, <strong>Stochastic gradient descent</strong>, <strong>Gradient descent with momentum</strong>, <strong>RMSprop</strong> and<strong> Adam</strong>) and see how these algorithms are used to train the learnable parameters on the training set. </p>

<h4><a name="_GD"></a><span style="color:darkblue">2.1.	Gradient descent  </span></h4>

<p>Gradient descent<sup>[<a href="#_Reference1">1</a>]</sup> represents the iterative algorithm that changes the learnable parameters by taking a small step in the opposite direction of the gradient of the loss function at the current point (of the learnable parameters). Since gradient points to the direction of steepest increase in the loss, the opposite direction gives the steepest decrease in the loss. That’s why it is called gradient descent. </p>

<p>The basic idea of gradient descent is illustrated in Figure 2 for a contour plot of loss function, \(L\), as a function of two learnable parameters, \({W_a}\) and \({W_b}\). In this plot, the redder, the greater loss; the bluer, the less loss. The minimum loss is indicated with the yellow dot, located at \(\left( { {W_a} = 1.26,{W_b} = 1.23 } \right)\). The initial random estimate of \(\left( { {W_a}^{\left( 0 \right)},{W_b}^{\left( 0 \right)} } \right)\) is indicated by the blue dot close to the top-left corner. The green arrows form a path from the initial point to the minimum loss point. Each arrow represents a training step, which points to the opposite direction of the gradients at the location of that step. For example, consider the 1st step at the blue dot, the gradient of the loss at that location actually points to the left, therefore, the training step takes the direction to the right, towards to the local minimum of loss. The same logic applies to the other steps and it ends up reaching the minimum loss at the yellow dot. </p>

<p>To help understand the intuition behind gradient descent, people often imagine this scenario: a person gets lost in a foggy mountain and is trying to get down to the village in the valley. Because of the fog, he cannot see a direct downhill path to the valley. One solution is to evaluate slope at each step and go with the direction of the steepest descent at the current location. Repeatedly carrying out the same strategy will eventually bring him down to the valley. </p>

<p align="center">
 <img src="{{ "/assets/images/Part4_Fig2_Loss_contour_ith_Arrows.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 2 Illustration of gradient descent in an example of loss function with 2 learnable parameters. 
</p> 

<p>Mathematically, let’s consider a regression case: the training dataset contains \(N\) separate samples, where the input of the \(i\)-th sample is \({X_i}\) and its target is \({Y_i}\),  and the network has \(M\) learnable parameters \(\left( { {w_1}, \cdots ,{w_M} } \right)\). The predicted output \({\hat Y_i}\) from the network for each sample can be written as  </p>

<div class="alert alert-secondary equation">
	<span>\({\hat Y_i} = f\left( { {X_i}{\rm{|} }{w_1}, \cdots ,{w_M} } \right)\) </span><span class="ref-num"> (1)</span>
</div>

<p>A typical loss function can be defined as Mean Squared Error</p>
<div class="alert alert-secondary equation">
	<span> \(L\left( { {w_1}, \cdots ,{w_M} } \right) = \frac{1} {N} \mathop \sum \limits_{i = 1}^N {\left( { { {\hat Y}_i} - {Y_i} } \right)^2}\)  </span><span class="ref-num"> (2)</span>
</div>

<p>Also assume that the gradient (or partial derivative) of the loss function \(L\left( { {w_1}, \cdots ,{w_M} } \right)\) w.r.t. individual parameter  \({w_m}\) has already been computed from back-propagation using the current values of the parameters \(\left( { {w_1}, \cdots ,{w_M} } \right)\), and labelled as \({g_m}\)</p>

<div class="alert alert-secondary equation">
	<span> \({g_m} \equiv \frac{ {\partial L} }{ {\partial {w_m} } }\)  </span><span class="ref-num"> (3)</span>
</div>


<p>Then, the iterative algorithm of gradient descent to update the learnable parameters can be written as </p>

<div class="alert alert-secondary equation">
	<span> 	\({w_m}^{\left( {t + 1} \right)} = {w_m}^{\left( t \right)} - \alpha  \cdot {g_m}^{\left( t \right)}\)  </span><span class="ref-num"> (4)</span>
</div>

<p>where \(t\) denotes the \(t\)-th iteration of update, \(\alpha \) is a positive number and denotes the learning rate and controls how big a step to take on each iteration.</p>


<p>There are a few points to make about Equation (4). </p>

<ul>
	<li>The term \( - \alpha  \cdot {g_m}\) is the amount to adjust \({w_n}\). The negative sign in the equation indicates taking the opposite direction of the gradient.</li>
	<li>If the gradient \({g_m}\) is positive, it means that an increase in \({w_n}\) will lead to an increase in loss \(L\). In this case, the adjustment amount \( - \alpha  \cdot {g_m}\) will be negative, so the new \({w_n}\) will be smaller, and therefore \({w_m}\) will be smaller, which in turn will lead the loss to be smaller. Vice versa.</li>
	<li>To differentiate from the learnable parameters \(w\), the learning rate \(\alpha \) is called hyper-parameter and typically requires users to tune up instead of learning from training. In other words, users need to try different \(\alpha \) in order to find an optimal one to use.</li>
	<li>Gradient descent typically uses the entire dataset as a batch to pass through the network, then performs one update of the learnable parameters. This is referred to as one “epoch” in training.</li>
</ul>


<h4><a name="_SGD"></a><span style="color:darkblue">2.2.	Stochastic gradient descent  </span></h4>

<p>Stochastic gradient descent (SGD) <sup>[<a href="#_Reference2">2</a>]</sup> is a variant of gradient descent and share the same basic idea and equations with GD. The only major difference is that, unlike GD that updates the parameters only once in an epoch based on the entire training dataset, SGD performs an update after each individual training sample passes through the network. Typically, the training dataset is randomly shuffled, and thus the update is based on a randomly picked sample, that’s why it is called Stochastic gradient descent - Stochastic means “random”. </p>

<p>The tradition gradient descent can be thought as to have a batch containing the entire training samples, while SGD can be thought as to have a batch contain only one single sample for each update.  What’s the impact of this difference? In general, SGD is more aggressive in updating the parameters and it provides the chance to converge faster for the same number of epochs, at least in the early epochs. However, due to the noise or errors in training dataset, update based on a single sample can have big bias. For example, even for the same set of parameters \(\left( { {w_1}, \cdots ,{w_M} } \right)\), two different training samples can lead to opposite directions in how to change the parameters. The fortunately thing is that, while some samples might cause the parameter to go with “wrong way”, other samples might cancel it out, so the overall trend still goes to the right way. On the contrary, gradient descent update the parameters at a much slower pace, but a much smoother way to find the optimal solution. It’s worth mentioning one burden about GD: if the entire dataset is passed through the network, it requires significantly more computing resources such as GPU memory. </p>

<p>A tradeoff between GD and SGD is introduced to neural networks by using “mini-batch” that takes a randomly-shuffled subset of the dataset instead of the entire dataset for each update. People usually choose a mini-batch size of power of 2. Now traditional GD and SGD can be thought of as two special and extreme cases of a more general mini-batch gradient descent.</p>

<p>Figure 3 compares the behaviors of 4 different mini-batch configurations by plotting loss vs epochs during training. In this example, the dataset contains 512 samples. The 4 configurations diffs only in the size of mini-batch: Mini-batch size=1, so 512 updates per epoch (SGD); Mini-batch size=32, so 16 updates per epoch; Mini-batch size=512, so 1 update per epoch (GD); Hybrid, mini-batch size of 32 for the first 8 epochs and size of 512 for the rest. Each point on the plots represents one evaluation of loss upon one update. For small min-batch size (such as 1), there are too many points, which would make the plots too busy, therefore it only plots 1 out of every 32 points for the case of mini-batch size=1 and 1 out of every 2 points for the case of mini-batch size=32. </p>

<p align="center">
 <img src="{{ "/assets/images/Part4_Fig3_SGD_Loss_vs_BatchSize_Plot.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 3 Comparison of the impact of batch size on gradient descent. 
</p> 

<p>From the plots, we can see that \(batchsize = 512\) has a slower converge at the early epochs, but the converge is quite smooth and stable. On the contrary, \(batchsize = 32\) has a faster converge during the early epoch, but suffers from noise. That’s why the hybrid configuration is investigated as well, trying to combine the advantages from both configurations. </p>


<h4><a name="_Momentum"></a><span style="color:darkblue">2.3.	Gradient Descent with momentut  </span></h4>

<p>Gradient descent with momentum<sup>[<a href="#_Reference3">3</a>]</sup> is a GD-based algorithm developed to speed up training by accelerating updates in the “right” direction and dampen oscillations in other directions. The basic idea is to compute an exponentially weighted average of the gradients from both the current step and the previous steps and then use that average to update the learnable parameters. Mathematically, it can be written as </p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({v_m}^{\left( t \right)} = \beta  \cdot {v_m}^{\left( {t - 1} \right)} + \left( {1 - \beta } \right) \cdot {g_m}^{\left( t \right)}\) </span><span class="ref-num"> (5) </span></p>
		
	<p class="equation"><span> \({w_m}^{\left( {t + 1} \right)} = {w_m}^{\left( t \right)} - \alpha  \cdot {v_m}^{\left( t \right)}\) </span><span class="ref-num"> (6)	</span></p>
</div>



<p>with the initial \({v_m}^{\left( 0 \right)}\)=0 and \(\beta  \in \left[ {0,1} \right)\). Let’s understand the meaning of Equation (5). If we substitute \({v_m}^{\left( {t - 1} \right)}\) with \({v_m}^{\left( {t - 2} \right)}\) and \({g_m}^{\left( {t - 1} \right)}\) and keep doing it recursively, it will end up with</p>

<div class="alert alert-secondary equation">
	<span> 	\({v_m}^{\left( t \right)} = \left( {1 - \beta } \right)\mathop \sum \limits_{k = 1}^t {\beta ^{t - k}}{g_m}^{\left( k \right)}\)  </span><span class="ref-num"> (7)</span>
</div>

<p>Equation (7) indicates that \({v_m}^{\left( t \right)}\) is a sum of past gradients \({g_m}^{\left( k \right)}\) weighted by the exponential factor of \({\beta ^{t - k}}\). That’s why it is called exponentially weighted average of the gradients. Because \(\beta \) is smaller than \(1\), an older gradient has less contribution to \({v_m}^{\left( t \right)}\) than a newer gradient. </p>

<p>The weighted average can be thought of as the momentum of a moving object in physics that has an inertia to main the motion in the original direction. The concept can be illustrated in Figure 4. When the “history” gradients are not taken into consideration in the standard GD algorithm, update is purely based on the “current” location alone and the path could look like the oscillating red arrows. By using the exponentially weighted average of the “history” in gradient descent with momentum, the components in the horizontal direction are likely to cancel out partially. As a result, the oscillations is reduced and the net effect is that the learning is making more progress in the vertical direction (the yellow arrow).</p>

<p align="center">
 <img src="{{ "/assets/images/Part4_Fig4_With_Momentum.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 4 Illustration of the concept of gradient descent with momentum. 
</p> 

<p> Please note that, if \({v_m}^{\left( 0 \right)}\) is set to zero, \({v_m}^{\left( t \right)}\) from Equation (5) will have a bias for early steps. To compensate for this bias, \({v_m}^{\left( t \right)}\) is adjusted with a correction factor before being used in Equation (6): </p>

<div class="alert alert-secondary equation">
	<span> 	\({v_m}^{\left( t \right)} := \frac{ { {v_m}^{\left( t \right)} } }{ {1 - {\beta ^t} } }\)  </span><span class="ref-num"> (8)</span>
</div>

<p><p>However, the bias becomes less important when \(t\) becomes bigger. Therefore, this correction is ignored in some implementations. </p>
 
A commonly used value for \(\beta \) is \(0.9\). When \(\beta  = 0\), this algorithm returns back to the standard GD algorithm. </p>

<h4><a name="_RMSprop"></a><span style="color:darkblue">2.4.	RMSprop  </span></h4>

<p> RMSprop stands for root mean square propagation<sup>[<a href="#_Reference4">4</a>]</sup>. It is another algorithm to dampen undesired oscillations by making the learning rate adaptive to the exponentially weighted average of the squared gradients from the past steps. </p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({s_m}^{\left( t \right)} = \beta  \cdot {s_m}^{\left( {t - 1} \right)} + \left( {1 - \beta } \right) \cdot {\left( { {g_m}^{\left( t \right)} } \right)^2}\) </span><span class="ref-num"> (9) </span></p>
		
	<p class="equation"><span> \({w_m}^{\left( {t + 1} \right)} = {w_m}^{\left( t \right)} - \frac{\alpha }{ {\sqrt { {s_m}^{\left( t \right)} }  + \epsilon} } \cdot {g_m}^{\left( t \right)}\) </span><span class="ref-num"> (10)	</span></p>
</div>

<p> The difference between Equations (9) and (5) is that it is the squared gradients, instead of the gradients, used to compute the weighted average for RMSprop. Comparing Equation (10) to Equation (4), we can see the only difference is that the learning rate \(\alpha \) is adjusted as \(\frac{\alpha }{ {\sqrt { {s_m}^{\left( t \right)} }  + \epsilon } }\), referred to as effective learning rate. A tiny number \( \epsilon \) is used to avoid the divided-by-zero exception where \({s_m}^{\left( t \right)} = 0\). A commonly used value for \(\beta \) is \(0.99\). </p>

<p> The intuition behind RMSprop is illustrated in Figure 5 below, a contour plot of loss as a function of two learnable parameters, \({w_h}\) in horizontal direction and \({w_v}\) in vertical direction. The starting point is A and the target point is M, the minimum location of the loss. In this example, we want to slow down the learning in the vertical direction and (relatively) speed up learning in the horizontal direction. Let’s see how the RMSprop algorithm help accomplish this objective. At point A, the magnitude of the gradient in the vertical direction, \({g_v}\), is much larger than that in the horizontal direction, \({g_h}\) (as you can see from the contour, the change in loss at point A is much more dramatic in the vertical direction). As a result, \({s_v}\) is much larger than \({s_h}\), so \(\frac{1}{ {\sqrt { {s_v} } } }\) is much smaller than \(\frac{1}{ {\sqrt { {s_h} } } }\). Therefore, the effective learning rate in the vertical, \(\frac{\alpha }{ {\sqrt { {s_v} } } }\), is smaller than that in the horizontal direction, \(\frac{\alpha }{ {\sqrt { {s_h} } } }\). The net effect is that the oscillation in the vertical direction is reduced and the training update will end up looking more like the yellow curve. </p>


<p align="center">
 <img src="{{ "/assets/images/Part4_Fig5_RMSprop.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 5 Illustration of the concept of RMSprop. 
</p> 


<h4><a name="_Adam"></a><span style="color:darkblue">2.5.	Adam  </span></h4>

<p>Adam<sup>[<a href="#_Reference5">5</a>]</sup> is another optimization algorithm that has been shown to work very well on a wide variety of applications. Adam stands for adaptive moment estimation. It combines the advantage of the gradient descent with momentum algorithm together with that of the RMSprop algorithm </p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({v_m}^{\left( t \right)} = {\beta _1} \cdot {v_m}^{\left( {t - 1} \right)} + \left( {1 - {\beta _1} } \right) \cdot {g_m}^{\left( t \right)}\) </span><span class="ref-num"> (11) </span></p>
		
	<p class="equation"><span>\({v_m}^{\left( t \right)} := \frac{ { {v_m}^{\left( t \right)} } }{ {1 - {\beta _1}^t} }\) </span><span class="ref-num"> (12)	</span></p>

    <p class="equation"><span> \({s_m}^{\left( t \right)} = {\beta _2} \cdot {s_m}^{\left( {t - 1} \right)} + \left( {1 - {\beta _2} } \right) \cdot {\left( { {g_m}^{\left( t \right)} } \right)^2}\) </span><span class="ref-num"> (13) </span></p>
		
	<p class="equation"><span> \({s_m}^{\left( t \right)} := \frac{ { {s_m}^{\left( t \right)} } }{ {1 - {\beta _2}^t} }\) </span><span class="ref-num"> (14)	</span></p>

    <p class="equation"><span> \({w_m}^{\left( {t + 1} \right)} = {w_m}^{\left( t \right)} - \frac{\alpha }{ {\sqrt { {s_m}^{\left( t \right)} }  + \epsilon} } \cdot {v_m}^{\left( t \right)}\) </span><span class="ref-num"> (15)	</span></p>
</div>


<p>Comparing these equations with those in the previous sections, we can see that Equations (11), (12) and the term \({v_m}^{\left( t \right)}\) in Equation (15) correspond to the momentum algorithm, whereas Equations (13), (14) and the adaptive learning rate \(\frac{\alpha }{ {\sqrt { {s_m}^{\left( t \right)} }  + \epsilon} }\) in Equation (15) correspond to the RMSprop algorithm. </p>

<p>There are 3 hyper-parameters in Adam algorithm. \({\beta _1}\) and \({\beta _2}\) are for the exponential weighted average of gradient and squared gradient, respectively. Typical default values are \(0.9\) for \({\beta _1}\) and \(0.99\) for \({\beta _2}\) and works well in most scenarios. On the other hand, the learning rate \(\alpha \) typically requires tune-up to find the optimal value for individual applications. </p>


<h3><a name="_CaseStudy"></a><span style="color:darkblue">3. A case study – Regression of Earth-to-Mars transfer orbit </span></h3>

<p>Just about two weeks ago, NASA’s Perseverance Rover successfully landed on Mars after its half a year and \(292.5\) million-mile journey from our planet, the Earth. Inspired by this great accomplishment, we will use a simplified model of the Earth-to-Mars transfer orbit as an example to demonstrate how to implement the optimization algorithms and how to incorporate them into neural network to help solve regress tasks. </p>

<p>With approximation (please see Figure 6), let’s assume both Earth (on blue circle) and Mars (on red circle) are moving around the Sun in circular orbits. Their radii are \(1.0\) \(AU\) and \(1.5237\) \(AU\), respectively (\(AU\) is astronomical unit, roughly the distance from Earth to the Sun and equal to about \(150\) million kilometers). In this story (somewhat different from the Perseverance rover), a spacecraft is launch from Earth at the point marked by the blue dot, then travels along Hofmann Transfer Orbit, and finally reaches Mars orbit at the point marked by the red dot. The Hofmann Transfer Orbit (green solid curve in the figure) is an elliptical orbit around the Sun and requires least amount of propellant to reach Mars. BTW, the actual orbit of Perseverance Rover is a little bit different from the Hofmann Transfer Orbit and takes less time to reach Mars. We will write a separate post to animate the Hofmann Transfer Orbit. </p>

<p>Taking the center of the Sun as the origin of the coordinate system, the Hofmann Transfer Orbit can be described by the elliptical equation</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ { { {\left( {x + 0.26185\;} \right)}^2} } }{ { {A^2} } } + \frac{ { {y^2} } }{ { {B^2} } } = 1\) </span><span class="ref-num"> (16)</span>
</div>


<p>where \(A = 1.261845\) is the length of semi-major axis and \(B = 1.234378\) is the length of semi-minor axis in unit of \(AU\). The offset of \(0.26185\) in x-axis is half of the distance between the two foci. This offset shows up in Equation (16) because the origin of the coordinate system is chosen to be at the center of the Sun (one of the two foci) instead of the center of the ellipse. </p>

<p align="center">
 <img src="{{ "/assets/images/Part4_Fig6_illustrate_orbits.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 6 Illustration of the Earth-to-Mars Transfer orbit. 
</p> 

<p>Using Equation (16), we sampled \(512\) \(\left( {N = 512} \right)\) points \(\left( { {X_i},{Y_i} } \right)\) of the Hohmann Transfer Orbit and then added random noise to the samples. The \(512\) samples are shown as the green dots in Figure 6.</p> 

<p>Now the task is, given the \(512\) samples and Equation (16) of an ellipse with unknown parameters \(A\) and \(B\), to find out what the values of \(A\) and \(B\) should be. This is a typical regress problem and can be solved using various mathematical ways. In this post, we will use neural networks with the discussed optimization algorithms to solve the problem. For the purpose of demonstration, we will assume the parameters \(A\) and \(B\) are independent, so the networks will have two learnable parameters, one for \(A\) and the other one for \(B\).</p> 


<h3><a name="_Implementations"></a><span style="color:darkblue">4. Implementations of the network and optimizers </span></h3>
	
<p> Recall that a typical network consists of 4 key steps during one iteration: forward propagation, calculation of loss, backpropagation and optimization. We will manually build each of these components. In this example, the learnable parameters are named \({w_A}\) and \({w_B}\), representing \(A\) and \(B\) respectively. They can be initialized with random numbers or with specific values for experiments. </p>

<h4><span style="color:darkblue">4.1.	Custom Dataset and Dataloader  </span></h4>

<p> To prepare the data for training, we write code to build our custom dataset by subclassing torch.utils.data.Dataset and implementing 3 member functions: </p>

<pre class="pre-scrollable">
	<code class="python">
class EllipseDataset(torch.utils.data.Dataset):    
    def __init__(self, N, A, B, noise_scale=0.1, transform=None):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
	</code>
</pre>

<p>Inside the <code class="python">__init__</code> constructor, it calls <code class="python">generate_training_data(N, a, b, noise_scale=0.1, plot_data=True)</code> to generate the training dataset of \(512\) samples of \(\left( { {X_i},{Y_i} } \right)\). In addition, transform can be very convenient when some operations such as transformation, scaling, image crop, etc. need to be applied to the samples. In this example, there is no such a need, so it is set to None. Please see the detailed source code <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Ellipse_Dataset.py">here on GitHub</a>. </p>

<p>Here are the two lines to instantiate <code class="python">EllipseDataset</code> and <code class="python">Dataloader</code>, and then fetch the mini-batch data from the <code class="python">Dataloader</code> during the training loop. </p>

<pre class="pre-scrollable">
	<code class="python">
xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1, transform =None)
xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
...
for t in range(epoch):
    for i_batch, sample_batched in enumerate(xy_dataloader):
        x, y = sample_batched['x'], sample_batched['y']
        ...
	</code>
</pre>

<p>The argument <code class="python">batch_size</code> can be any integer between \(1\) and \(N\) inclusive, but is suggested to choose a power of \(2\). The vectored \(x\) represents the input features (actually the \(x\)-coordinates of the sampled points from the Hofmann Transfer Orbit) and the vectored \(y\) represents the corresponding targets (the simulated \(y\) -coordinates of the sampled points). </p>

<h4><span style="color:darkblue">4.2.	Forward pass  </span></h4>

<p>The forward pass is to calculate the predicted \(\hat y\) for the given input \(x\)  based on the following equation, which is simply re-arranged form Equation (16) above. </p>

<div class="alert alert-secondary equation">
	<span>\({\hat y_i} = {w_B} \cdot \sqrt {\left( {1 - \frac{ { { {\left( { {x_i} + 0.26185\;} \right)}^2} } }{ { {w_A}^2} } } \right)} \) </span><span class="ref-num"> (17)</span>
</div>

<p>One thing to note is that, due to the noise introduced to the samples, the subtraction operation can result in negative values that is invalid to take square root. For those samples running into this issue, their out \({\hat y_i}\) are set to zero, and excluded from the loss and gradient calculations. </p>


<h4><span style="color:darkblue">4.3.	Loss function  </span></h4>

<p>In this example, loss is defined as the mean squared error between the predicted \({\hat y_i}\) and the given target \({y_i}\). </p>

<div class="alert alert-secondary equation">
	<span>\(L\left( { {w_A},{w_B} } \right) = \frac{1}{ {N'} }\mathop \sum \limits_{i = 1}^{N'} {\left( { { {\hat y}_i} - {y_i} } \right)^2}\) </span><span class="ref-num"> (18)</span>
</div>

<p>where \(N'\) denotes the number of all samples in the current mini-batch that don’t run into the square root issue. </p>

<h4><span style="color:darkblue">4.4.	Back-propagation to calculate gradients  </span></h4>

<p> The gradients of the loss function w.r.t. the learnable parameters \({w_A}\) and \({w_B}\) are derived using the chain rule that have been extensively discussed in the previous posts. Here we directly give out the derived results. </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial {w_A} } } = \frac{1}{ {N'} }\mathop \sum \limits_{i = 1}^{N'} 2\left( { { {\hat y}_i} - {y_i} } \right) \cdot \frac{ { { {\left( { {x_i} + 0.26185} \right)}^2} } }{ { { {\hat y}_i} } } \cdot \frac{ { {w_B}^2} }{ { {w_A}^3} }\)</span></p>

		<p><span> \(\frac{ {\partial L} }{ {\partial {w_B} } } = \frac{1}{ {N'} }\mathop \sum \limits_{i = 1}^{N'} 2\left( { { {\hat y}_i} - {y_i} } \right) \cdot \frac{ { { {\hat y}_i} } }{ { {w_B} } }\) </span></p>		
	</span>
	<span class="ref-num">(29)</span>
</div>

<h4><span style="color:darkblue">4.5.	Update of learnable parameters using the optimizers  </span></h4>

<p>The gradient descent, SGD, GD with momentum, RMSprop and Adam are implemented following the equations described in Section 2. In order validate these custom implementations, we also build another block of codes to do the same job using Torch built-in function<sup>[<a href="#_Reference6">6</a>]</sup> for comparison. For example, the following snippet shows the implementation of the Adam algorithm. </p>

<pre class="pre-scrollable">
	<code class="python">
if flag_manual_implement: 
    # Step 3: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
    dWa_via_yi = 2.0 * (y_pred - y) * ((x + c) ** 2) * (Wb ** 2) / (Wa ** 3) / y_pred
    dWa = dWa_via_yi[~negativeloc].mean()  # gradient w.r.t Wa

    dWb_via_yi = (2.0 * (y_pred - y) * y_pred / Wb)
    dWb = dWb_via_yi[~negativeloc].mean()  # gradient w.r.t Wb

    # Step 4: Update weights using the Adam algorithm.
    with torch.no_grad():
        beta1_to_pow_t *= beta1
        beta1_correction = 1.0 - beta1_to_pow_t
        beta2_to_pow_t *= beta2
        beta2_correction = 1.0 - beta2_to_pow_t

        VdWa = beta1 * VdWa + (1.0 - beta1) * dWa
        SdWa = beta2 * SdWa + (1.0 - beta2) * dWa * dWa
        Wa -= learning_rate * (VdWa / beta1_correction) / (torch.sqrt(SdWa) / math.sqrt(beta2_correction) + eps)

        VdWb = beta1 * VdWb + (1.0 - beta1) * dWb
        SdWb = beta2 * SdWb + (1.0 - beta2) * dWb * dWb
        Wb -= learning_rate * (VdWb / beta1_correction) / (torch.sqrt(SdWb) / math.sqrt(beta2_correction) + eps)

else:  # do the same job using Torch built-in autograd and optim
    # reset the gradients of the learnable parameters to zero. Otherwise, it will be accumulated. 
    optimizer.zero_grad()
    # Step 3: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
    loss.backward()
    # Step 4: Update weights using Adam algorithm.
    optimizer.step()
	</code>
</pre>

<p>Please see the complete source code for <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Ellipse_Optimizer_SGD_implement.py">GD</a>, <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Ellipse_Optimizer_SGD_w_Momentum_implement.py">GD with momentum</a>, <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Ellipse_Optimizer_RMSprop_implement.py">RMSprop</a> and <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Ellipse_Optimizer_Adam_implement.py">Adam</a> on GitHub, respectively. You can also try the custom implementation or the Torch version by setting <code class="python">flag_manual_implement</code> to <code class="python">True</code> or <code class="python">False</code>.  </p>

<p>Figures 7 through 10 shows the results from using the 4 optimizers, respectively. The starting point is \({A_{start} } = 0.1\) and \({B_{start} } = 1.8\). You can see from the figures, all of the 4 final results are very close to the ground truth (\({A_{truth} } = 1.261845\) and \({B_{truth} } = 1.234378\)) that were used to generate the noisy samples. However, each optimizer has its own searching path. Figure 11 shows an animation of the 4 optimizers’ update paths from the starting point to the minimum loss. Please note that this video is not an apple-to-apple comparison between these algorithms because different learning rates and adjustment were used. Why not use exactly the same learning rates and adjustment? It is because a learning rate working for one algorithm could cause failure for another one. We will talk about learning rate in a separate post. The animation is presented here to provide some general ideas. Just in case if the video cannot be played in your browser, you can download the <a href="{{ "/assets/images/Part4_Video1.mp4" | relative_url }}">mp4 video here </a>. You can also find the <a href="https://github.com/coolgpu/Demo_Optimizers/blob/main/Animation_of_loss_vs_TrainingUpdates.py">source code of the animation here</a>. </p>

<p align="center">
 <img src="{{ "/assets/images/Part4_Fig7_SGD_Result_plot.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 7 Results of the SGD optimizer. 
</p> 
<br>
<p align="center">
 <img src="{{ "/assets/images/Part4_Fig8_Momentum_Result_plot.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 8 Results of the GD with Momentum optimizer. 
</p> 
<br>
<p align="center">
 <img src="{{ "/assets/images/Part4_Fig9_RMSprop_Result_plot.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 9 Results of the RMSprop optimizer. 
</p> 
<br>
<p align="center">
 <img src="{{ "/assets/images/Part4_Fig10_Adam_Result_plot.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 10 Results of the Adam  optimizer. 
</p> 

<p align="center">
<video width="640" height="640" controls>
<source src="{{ "/assets/images/Part4_Video1.mp4" | relative_url }}" type="video/mp4">
</video> 
<br>Figure 11 Animation of the 4 optimizers’ update paths from the starting point to the minimum loss. 
</p>
<br>

<h3><a name="_HyperParameters"></a><span style="color:darkblue">5. Hyper-parameters and learn rate scheduler</span></h3>

<p>This post has been quite long and the hyper-parameters and learning rate schedulers seem to deserve a separate post, so we will stop here and will cover those topics separately.</p>


<h3><a name="_Summary"></a><span style="color:darkblue">6. Summary </span></h3> 
<p>In this post, we went through the fundamentals of optimization in artificial neural networks and discussed 4 widely used optimizers (gradient descent / stochastic gradient descent and its variants to speed up the training using GD with momentum, RMSprop and Adam algorithms). We also demonstrated how to implement them with a network to solve a regression task in the case study of Hohmann Earth-to-Mars Transfer Orbit. </p>

 <p>We hope, by walking through these topics and hands-on implementations, we can gain in-depth understanding of how neural network works. Those terminologies, such as gradients, back-propagation, loss function, optimization, etc. will hopefully no longer seem to be a secret.   
</p>



<h3><a name="_References"></a><span style="color:darkblue">7. References</span></h3> 
<ul>
    <li><a name="_Reference1"></a>[1] <a href="https://en.wikipedia.org/wiki/Gradient_descent"> Gradient descent on Wikipedia. </a></li>

    <li><a name="_Reference2"></a>[2] Léon Bottou (1998). <a href="https://leon.bottou.org/publications/pdf/online-1998.pdf">Online Algorithms and Stochastic Approximations.</a></li>

    <li><a name="_Reference3"></a>[3] Ning Qian (1999). <a href="https://web.archive.org/web/20140508200053/http://brahms.cpmc.columbia.edu/publications/momentum.pdf">On the momentum term in gradient descent learning algorithms. Neural networks : the official journal of the International Neural Network Society, 12(1):145–151.</a></li>

    <li><a name="_Reference4"></a>[4] Geoffrey Hinton. <a href="http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf">Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude. (Page 26)</a></li>   

	<li><a name="_Reference5"></a>[5] Diederik P. Kingma and Jimmy Lei Ba (2014). <a href="https://arxiv.org/abs/1412.6980">Adam : A method for stochastic optimization.</a></li>

	<li><a name="_Reference6"></a>[6] <a href="https://pytorch.org/docs/stable/optim.html"> torch.optim - PyTorch documentation. </a></li>
</ul>

<br>

<div id="HCB_comment_box">
<a href="http://www.htmlcommentbox.com"></a> is loading comments...
</div>
<link rel="stylesheet" type="text/css" href="https://www.htmlcommentbox.com/static/skins/bootstrap/twitter-bootstrap.css?v=0" />
<script type="text/javascript" id="hcb"> 
if(!window.hcb_user){hcb_user={};} 
(function(){
	var s=document.createElement("script"), l=hcb_user.PAGE || (""+window.location).replace(/'/g,"%27"), h="https://www.htmlcommentbox.com";
	s.setAttribute("type","text/javascript");
	s.setAttribute("src", h+"/jread?page="+encodeURIComponent(l).replace("+","%2B")+"&opts=16862&num=10&ts=1599873842684");
	if (typeof s!="undefined") document.getElementsByTagName("head")[0].appendChild(s);})(); 
</script>