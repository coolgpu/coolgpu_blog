---
layout: post
mathjax: true
title:  "Understanding Conv2d and ConvTrans2d in Neural Networks -- Part 1. Matrix Multiplication and Its Gradients"
date:   2020-09-22 07:46:41 -0700
categories: github pages
author: Xiyun Song
---

<p>Convolution is the most widely used and important layer in deep learning neural networks for image classification or regression tasks. Its counterpart, transpose convolution or typically named as ConvTranspose, is also widely used in networks (e.g. UNet) to convert data back to the original image size so that they can be added to or concatenated with the original data to form skip layers. Due to the complexity involved in the forward and backward computation, both convolution and ConvTranspose do not seem as straightforward as other modules. Therefore, we plan to use the following 4 posts to explain the fundamentals of convolution and ConvTranspose with examples of custom implementations and hope to help clarify these concepts. </p>

<ul>
	<li><a href="">Matrix multiplication and its custom implementation (this post)</a></li>
	<li>Conv2d and its custom implementation</li>
	<li>ConvTranpose2d and its custom implementation </li>
	<li>Application of Conv2d and ConvTranpose2d in Neural Networks</li>
</ul>


<p>We take the 1st part to talk about matrix multiplication because it can be used in Convolution and ConvTranspose operations to make things  simpler. In this post, we will focus on derivation of the gradients of matrix multiplication. While the derivation process may seem complex, the final results will be in a pretty simple form and are easy to remember. This post is organized into the following sections:</p>

<ul>
	<li><a href="#_The_simple_network">Matrix multiplication</a></li>
	<li><a href="#_Derivation_of_the_gradients">Derivation of gradients using backpropagation chain rule</a></li>
	<li><a href="#_Custom_implementations_and_validation">Custom implementations and validation </a></li>
	<li><a href="#_Summary">Summary</a></li>
	<li><a href="#_References">References</a></li>
</ul>

<h3><a name="_Matrix_multiplication"></a>1.	Matrix multiplication</h3>  

<p>The definition of matrix multiplication can be found in every linear algebra book. Let’s use the definition from <a href="https://en.wikipedia.org/wiki/Matrix_multiplication">Wikipedia</a>. Given a \(m \times k\) matrix \(\boldsymbol {A}\) and a \(k \times n\) matrix \(\boldsymbol {B}\)</p>

<div class="alert alert-secondary equation">
	<span>\(\boldsymbol {A}  = \left[ {\begin{array}{*{20}{c} }{ {a_{11} } }&{ {a_{12} } }& \ldots &{ {a_{1k} } }\\{ {a_{21} } }&{ {a_{22} } }& \ldots &{ {a_{2k} } }\\ \vdots & \vdots & \ddots & \vdots \\{ {a_{m1} } }&{ {a_{m2} } }& \ldots &{ {a_{mk} } }\end{array} } \right]\) and \(\boldsymbol {B}  = \left[ {\begin{array}{*{20}{c} }{ {b_{11} } }&{ {b_{12} } }& \ldots &{ {b_{1n} } }\\{ {b_{21} } }&{ {b_{22} } }& \ldots &{ {b_{2n} } }\\ \vdots & \vdots & \ddots & \vdots \\{ {b_{k1} } }&{ {b_{k2} } }& \ldots &{ {b_{kn} } }\end{array} } \right]\),	</span><span class="ref-num"> (1) </span>
</div>

<p>their matrix product \(\boldsymbol {C}  = AB\) is defined as </p>

<div class="alert alert-secondary equation">
	<span>\(\boldsymbol {C}  = \left[ {\begin{array}{*{20}{c} }{ {c_{11} } }&{ {c_{12} } }& \ldots &{ {c_{1n} } }\\{ {c_{21} } }&{ {c_{22} } }& \ldots &{ {c_{2n} } }\\ \vdots & \vdots &{ {c_{ij} } }& \vdots \\{ {c_{m1} } }&{ {c_{m2} } }& \ldots &{ {c_{mn} } }\end{array} } \right]\),	</span><span class="ref-num"> (2) </span>
</div>

<p>where its element \({c_{ij}}\) is given by </p>

<div class="alert alert-secondary equation">
	<span>\({c_{ij} } = \mathop \sum \limits_{t = 1}^k {a_{it} }{b_{tj} }\),	</span><span class="ref-num"> (3) </span>
</div>

<p>for \(i = 1, \ldots ,m\) and \(j = 1, \ldots ,n\). In other words, \({c_{ij}}\) is the dot product of the \(i\)th row of \(\boldsymbol {A} \) and the \(j\)th column of \(\boldsymbol {B} \). </p>

<h3><a name="_Derivation_of_the_gradients"></a>2. Derivation of the gradients</h3>
<h4>2.1. Dimensions of the gradients</h4>
<p>If we are considering an isolated matrix multiplication, the partial derivative matrix \(\boldsymbol {C} \) with respect to either matrix \(\boldsymbol {A} \) and matrix \(\boldsymbol {B} \) would be a 4-D hyper-space relationship, referred to as Jacobian Matrix. You will also find that there will be many zeros in the 4-D Jacobian Matrix because, as shown in Equation (3), \({c_{ij} }\) is a function of only the \(i\)th row of \(\boldsymbol {A} \) and the \(j\)th column of \(\boldsymbol {B} \), and independent of other rows of \(\boldsymbol {A} \) and other columns of \(\boldsymbol {B} \). </p>

<p>What we are considering here is not an isolated matrix multiplication. Instead, we are talking about matrix multiplication inside a neural network that will have a scalar loss function. For example, consider a simple case where the loss \(L\) is the mean of matrix \(\boldsymbol {C} \):</p>

<div class="alert alert-secondary equation">
	<span>\(L = \frac{1}{ {m \times n} }\mathop \sum \limits_{i = 1}^m \mathop \sum \limits_{j = 1}^n {c_{ij} }\),	</span><span class="ref-num"> (4) </span>
</div>

<p>our focus is the partial derivatives of scalar \(L\) w.r.t. the input matrix \(\boldsymbol {A} \) and \(\boldsymbol {B} \),  \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) and \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\), respectively. Therefore, \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) has the same dimension as \(\boldsymbol {A} \), which is another \(m \times k\) matrix, and  \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\) has the same dimension as \(\boldsymbol {B} \), which is another \(k \times n\) matrix.</p>

<h4>2.2 The chain rule</h4>
<p>We will use the chain rule to do backpropagation of gradients. For such an important tool in neural networks, it doesn’t hurt to briefly summarize the chain rule just like in <a href="https://coolgpu.github.io/coolgpu_blog/github/pages/2020/09/14/backpropagation.html">the previous post</a> for one more time. Given a function \(L\left( { {x_1},{x_2}, \ldots {x_N} } \right)\) as</p>

<div class="alert alert-secondary equation">
	<span> \(L\left( { {x_1}, \ldots {x_N} } \right) = L\left( { {f_1}\left( { {x_1}, \ldots {x_N} } \right),{f_2}\left( { {x_1}, \ldots {x_N} } \right), \ldots ,{f_M}\left( { {x_1}, \ldots {x_N} } \right)} \right)\)</span><span class="ref-num">(5)</span>
</div>

<p>Then the gradient of \(L\) w.r.t \({x_i}\) can be computed as </p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial {x_i} } } = \frac{ {\partial L} }{ {\partial {f_1} } }\frac{ {\partial {f_1} } }{ {\partial {x_i} } } + \frac{ {\partial L} }{ {\partial {f_2} } }\frac{ {\partial {f_2} } }{ {\partial {x_i} } } +\cdots + \frac{ {\partial L} }{ {\partial {f_M} } }\frac{ {\partial {f_M} } }{ {\partial {x_i} } } = \mathop \sum \limits_{m = 1}^M \frac{ {\partial L} }{ {\partial {f_m} } }\frac{ {\partial {f_m} } }{ {\partial {x_i} } }\)</span><span class="ref-num">(6)</span>
</div>

<p>Equation (6) can be understood from two perspectives:</p>
<ul>
<li>Summation means that all possible paths through which \({x_i}\) contributes to \(L\) should be included</li>
<li>Product means that, along each path \(m\), the output gradient equals the upstream passed in, \(\frac{ {\partial L} }{ {\partial {f_m} } }\), times the local gradient, \(\frac{ {\partial {f_m} } }{ {\partial {x_i} } }\). </li>
</ul>


<h4>2.3 Derivation of the gradient \(\frac{ {\partial L} }{ {\partial  \boldsymbol {\boldsymbol {A} } } }\)  </h4> 
<p>In this section, we will use a \(2 \times 4\) matrix \(\boldsymbol {A} \) and a \(4 \times 3\) matrix \(\boldsymbol {B} \) as an example to step-by-step derive the partial derivative of \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\). Please note that the same derivation can be performed on a general \(m \times k\) matrix \(\boldsymbol {A} \) and \(k \times n\) matrix \(\boldsymbol {B} \). A specific example is used here purely for the purpose of making it more straightforward. </p>

<p>Let’s start with writing the matrix \(\boldsymbol {A} \), \(\boldsymbol {B} \) and their matrix product \(\boldsymbol {C}  = AB\) in expanded format.</p>

<div class="alert alert-secondary equation">
	<span>\(\boldsymbol {A}  = \left[ {\begin{array}{*{20}{c} }{ {a_{11} } }&{ {a_{12} } }&{ {a_{13} } }&{ {a_{14} } }\\{ {a_{21} } }&{ {a_{22} } }&{ {a_{23} } }&{ {a_{24} } }\end{array} } \right]\) and \(\boldsymbol {B}  = \left[ {\begin{array}{*{20}{c} }{ {b_{11} } }&{ {b_{12} } }&{ {b_{13} } }\\{ {b_{21} } }&{ {b_{22} } }&{ {b_{23} } }\\{ {b_{31} } }&{ {b_{32} } }&{ {b_{33} } }\\{ {b_{41} } }&{ {b_{42} } }&{ {b_{43} } }\end{array} } \right]\),	</span><span class="ref-num"> (7) </span>
</div>


<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\boldsymbol {C}  = \left[ {\begin{array}{*{20}{c} }{ {c_{11} } }&{ {c_{12} } }&{ {c_{13} } }\\{ {c_{21} } }&{ {c_{22} } }&{ {c_{23} } }\end{array} } \right] = \left[ {\begin{array}{*{20}{c} }{ {a_{11} } }&{ {a_{12} } }&{ {a_{13} } }&{ {a_{14} } }\\{ {a_{21} } }&{ {a_{22} } }&{ {a_{23} } }&{ {a_{24} } }\end{array} } \right]\left[ {\begin{array}{*{20}{c} }{ {b_{11} } }&{ {b_{12} } }&{ {b_{13} } }\\{ {b_{21} } }&{ {b_{22} } }&{ {b_{23} } }\\{ {b_{31} } }&{ {b_{32} } }&{ {b_{33} } }\\{ {b_{41} } }&{ {b_{42} } }&{ {b_{43} } }\end{array} } \right]\)</span></p>

		<p><span> \(= \left[ {\begin{array}{*{20}{c} }{ { {a_{11} }{b_{11} } + {a_{12} }{b_{21} } + {a_{13} }{b_{31} } + {a_{14} }{b_{41} } } }&{ { {a_{11} }{b_{12} } + {a_{12} }{b_{22} } + {a_{13} }{b_{32} } + {a_{14} }{b_{42} } } }&{ { {a_{11} }{b_{13} } + {a_{12} }{b_{23} } + {a_{13} }{b_{33} } + {a_{14} }{b_{43} } } }\\{ { {a_{21} }{b_{11} } + {a_{22} }{b_{21} } + {a_{23} }{b_{31} } + {a_{24} }{b_{41} } } }&{ { {a_{21} }{b_{12} } + {a_{22} }{b_{22} } + {a_{23} }{b_{32} } + {a_{24} }{b_{42} } } }&{ { {a_{21} }{b_{13} } + {a_{22} }{b_{23} } + {a_{23} }{b_{33} } + {a_{24} }{b_{43} } } }\end{array} } \right] \) </span></p>	
	</span>
	<span class="ref-num">(8)</span>
</div>

<p> Consider an arbitrary element of \(\boldsymbol {A} \), for example \({a_{23} }\), we have the local partial derivative of \(\boldsymbol {C} \)  w.r.t. \({a_{23} }\) based on Equation (8). </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial {c_{11} } } }{ {\partial {a_{23} } } } = 0\)	</span></p>

		<p><span> \(\frac{ {\partial {c_{12} } } }{ {\partial {a_{23} } } } = 0\) </span></p>

		<p><span> \(\frac{ {\partial {c_{13} } } }{ {\partial {a_{23} } } } = 0\) </span></p>

		<p><span> \(\frac{ {\partial {c_{21} } } }{ {\partial {a_{23} } } } = \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{21} }{b_{11} } + {a_{22} }{b_{21} } + {a_{23} }{b_{31} } + {a_{24} }{b_{41} } } \right) = 0 + 0 + \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{23} }{b_{31} } } \right) + 0 = {b_{31} }\)</span></p>

        <p><span> \(\frac{ {\partial {c_{22} } } }{ {\partial {a_{23} } } } = \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{21} }{b_{12} } + {a_{22} }{b_{22} } + {a_{23} }{b_{32} } + {a_{24} }{b_{42} } } \right) = 0 + 0 + \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{23} }{b_{32} } } \right) + 0 = {b_{32} }\) </span></p>

        <p><span> \(\frac{ {\partial {c_{23} } } }{ {\partial {a_{23} } } } = \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{21} }{b_{13} } + {a_{22} }{b_{23} } + {a_{23} }{b_{33} } + {a_{24} }{b_{43} } } \right) = 0 + 0 + \frac{\partial }{ {\partial {a_{23} } } }\left( { {a_{23} }{b_{33} } } \right) + 0 = {b_{33} }\) </span></p>
	</span>
	<span class="ref-num">(9)</span>
</div>

<p> Using the chain rule, we have the partial derivative of the loss \(L\) w.r.t. \({a_{23}}\) </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \( \frac{ {\partial L} }{ {\partial {a_{23} } } } = \frac{ {\partial L} }{ {\partial {c_{11} } } }\frac{ {\partial {c_{11} } } }{ {\partial {a_{23} } } } + \frac{ {\partial L} }{ {\partial {c_{12} } } }\frac{ {\partial {c_{12} } } }{ {\partial {a_{23} } } } + \frac{ {\partial L} }{ {\partial {c_{13} } } }\frac{ {\partial {c_{13} } } }{ {\partial {a_{23} } } } + \frac{ {\partial L} }{ {\partial {c_{21} } } }\frac{ {\partial {c_{21} } } }{ {\partial {a_{23} } } } + \frac{ {\partial L} }{ {\partial {c_{22} } } }\frac{ {\partial {c_{22} } } }{ {\partial {a_{23} } } } + \frac{ {\partial L} }{ {\partial {c_{23} } } }\frac{ {\partial {c_{23} } } }{ {\partial {a_{23} } } }\)	</span></p>

        <p><span>\( = 0 + 0 + 0 + \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{31} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{32} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{33} }\)</span></p>

		<p><span>\( = \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{31} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{32} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{33} }\)</span></p>

		
	</span>
	<span class="ref-num">(10)</span>
</div>


<p> Following a similar manner, we can derive the other elements of \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) as below </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } } = \left[ {\begin{array}{*{20}{c} }{\frac{ {\partial L} }{ {\partial {a_{11} }} }}&{\frac{ {\partial L} }{ {\partial {a_{12} }} }}&{\frac{ {\partial L} }{ {\partial {a_{13} }} }}&{\frac{ {\partial L} }{ {\partial {a_{14} }} }}\\{\frac{ {\partial L} }{ {\partial {a_{21} }} }}&{\frac{ {\partial L} }{ {\partial {a_{22} }} }}&{\frac{ {\partial L} }{ {\partial {a_{23} }} }}&{\frac{ {\partial L} }{ {\partial {a_{24} }} }}\end{array} } \right]\)	</span></p>

        <p><span> \( = \left[ {\begin{array}{*{20}{c} }{ {\frac{ {\partial L} }{ {\partial {c_{11} } } }{b_{11} } + \frac{ {\partial L} }{ {\partial {c_{12} } } }{b_{12} } + \frac{ {\partial L} }{ {\partial {c_{13} } } }{b_{13} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{11} } } }{b_{21} } + \frac{ {\partial L} }{ {\partial {c_{12} } } }{b_{22} } + \frac{ {\partial L} }{ {\partial {c_{13} } } }{b_{23} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{11} } } }{b_{31} } + \frac{ {\partial L} }{ {\partial {c_{12} } } }{b_{32} } + \frac{ {\partial L} }{ {\partial {c_{13} } } }{b_{33} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{11} } } }{b_{41} } + \frac{ {\partial L} }{ {\partial {c_{12} } } }{b_{42} } + \frac{ {\partial L} }{ {\partial {c_{13} } } }{b_{43} } } }\\{ { \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{11} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{12} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{13} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{21} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{22} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{23} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{31} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{32} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{33} } } }&{ { \frac{ {\partial L} }{ {\partial {c_{21} } } }{b_{41} } + \frac{ {\partial L} }{ {\partial {c_{22} } } }{b_{42} } + \frac{ {\partial L} }{ {\partial {c_{23} } } }{b_{43} } } }\end{array} } \right]\)  </span></p>
		
	</span>
	<span class="ref-num">(11)</span>
</div>

<p>Equation (11) can be equivalently rewritten as a matrix product.</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } } = \left[ {\begin{array}{*{20}{c} }{\frac{ {\partial L} }{ {\partial {c_{11} } } } }&{\frac{ {\partial L} }{ {\partial {c_{12} } } } }&{\frac{ {\partial L} }{ {\partial {c_{13} } } } }\\{\frac{ {\partial L} }{ {\partial {c_{21} } } } }&{\frac{ {\partial L} }{ {\partial {c_{22} } } } }&{\frac{ {\partial L} }{ {\partial {c_{23} } } } }\end{array} } \right]\left[ {\begin{array}{*{20}{c} }{ {b_{11} } }&{ {b_{21} } }&{ {b_{31} } }&{ {b_{41} } }\\{ {b_{12} } }&{ {b_{22} } }&{ {b_{32} } }&{ {b_{42} } }\\{ {b_{13} } }&{ {b_{23} } }&{ {b_{33} } }&{ {b_{43} } }\end{array} } \right]\)</span><span class="ref-num">(12)</span>
</div>

<p> In fact, the first matrix is the upstream derivative \(\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\) and the second matrix is the transpose of \(\boldsymbol {B} \). Then we have </p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } } = \frac{ {\partial L} }{ {\partial \boldsymbol {C} } }{\boldsymbol {B} ^T}\)</span><span class="ref-num">(13)</span>
</div>

<p> Let’s check the dimensions. On the left hand side of Equation (13), \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) has a dimension of \(m \times k\). On the right hand side,  \(\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\) has a dimension of \(m \times n\) and \({\boldsymbol {B} ^T}\) has a dimension of \(n \times k\); thereforef their matrix product has a dimension of \(m \times k\) and matches that of \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\). </p>


<h4>2.4 Derivation of the gradient \(\frac{ {\partial L} }{ {\partial  \boldsymbol {\boldsymbol {B} } } }\)  </h4> 
<p>Similarly, for \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\), let’s consider an arbitrary element of \(\boldsymbol {B} \), for example \({b_{12} }\), we have the local partial derivative of \(\boldsymbol {C} \)  w.r.t. \({b_{12} }\) based on Equation (8) above. </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial {c_{11} } } }{ {\partial {b_{12} } } } = 0\)	</span></p>

		<p><span> \(\frac{ {\partial {c_{12} } } }{ {\partial {b_{12} } } } = \frac{\partial }{ {\partial {b_{12} } } }\left( { {a_{11} }{b_{12} } + {a_{12} }{b_{22} } + {a_{13} }{b_{32} } + {a_{14} }{b_{42} } } \right) = {a_{11} }\) </span></p>

		<p><span> \(\frac{ {\partial {c_{13} } } }{ {\partial {b_{12} } } } = 0\) </span></p>

		<p><span> \(\frac{ {\partial {c_{21} } } }{ {\partial {b_{12} } } } = 0\) </span></p>

        <p><span> \(\frac{ {\partial {c_{22} } } }{ {\partial {b_{12} } } } = \frac{\partial }{ {\partial {b_{12} } } }\left( { {a_{21} }{b_{12} } + {a_{22} }{b_{22} } + {a_{23} }{b_{32} } + {a_{24} }{b_{42} } } \right) = {a_{21} }\) </span></p>

        <p><span> \(\frac{ {\partial {c_{23} } } }{ {\partial {b_{12} } } } = 0\) </span></p>
	</span>
	<span class="ref-num">(13)</span>
</div>

<p> Using the chain rule, we have the partial derivative of the loss \(L\) w.r.t. \({b_{12} }\) </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \( \frac{ {\partial L} }{ {\partial {b_{12} } } } == \frac{ {\partial L} }{ {\partial {c_{11} } } }\frac{ {\partial {c_{11} } } }{ {\partial {b_{12} } } } + \frac{ {\partial L} }{ {\partial {c_{12} } } }\frac{ {\partial {c_{12} } } }{ {\partial {b_{12} } } } + \frac{ {\partial L} }{ {\partial {c_{13} } } }\frac{ {\partial {c_{13} } } }{ {\partial {b_{12} } } } + \frac{ {\partial L} }{ {\partial {c_{21} } } }\frac{ {\partial {c_{21} } } }{ {\partial {b_{12} } } } + \frac{ {\partial L} }{ {\partial {c_{22} } } }\frac{ {\partial {c_{22} } } }{ {\partial {b_{12} } } } + \frac{ {\partial L} }{ {\partial {c_{23} } } }\frac{ {\partial {c_{23} } } }{ {\partial {b_{12} } } }\)	</span></p>

        <p><span> \( = {a_{11} }\frac{ {\partial L} }{ {\partial {c_{12} } } } + {a_{21} }\frac{ {\partial L} }{ {\partial {c_{22} } } }\) </span></p>

		
	</span>
	<span class="ref-num">(14)</span>
</div>

<p> Following a similar manner again, we can derive the other elements of \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\) as shown below </p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } } = \left[ {\begin{array}{*{20}{c} }{\frac{ {\partial L} }{ {\partial {b_{11} } } } }&{\frac{ {\partial L} }{ {\partial {b_{12} } } } }&{\frac{ {\partial L} }{ {\partial {b_{13} } } } }\\{\frac{ {\partial L} }{ {\partial {b_{21} } } } }&{\frac{ {\partial L} }{ {\partial {b_{22} } } } }&{\frac{ {\partial L} }{ {\partial {b_{23} } } } }\\{\frac{ {\partial L} }{ {\partial {b_{31} } } } }&{\frac{ {\partial L} }{ {\partial {b_{32} } } } }&{\frac{ {\partial L} }{ {\partial {b_{33} } } } }\\{\frac{ {\partial L} }{ {\partial {b_{41} } } } }&{\frac{ {\partial L} }{ {\partial {b_{42} } } } }&{\frac{ {\partial L} }{ {\partial {b_{43} } } } }\end{array} } \right]\)	</span></p>

        <p><span> \(= \left[ {\begin{array}{*{20}{c} }{ { {a_{11} }\frac{ {\partial L} }{ {\partial {c_{11} } } } + {a_{21} }\frac{ {\partial L} }{ {\partial {c_{21} } } } } }&{ { {a_{11} }\frac{ {\partial L} }{ {\partial {c_{12} } } } + {a_{21} }\frac{ {\partial L} }{ {\partial {c_{22} } } } } }&{ { {a_{11} }\frac{ {\partial L} }{ {\partial {c_{13} } } } + {a_{21} }\frac{ {\partial L} }{ {\partial {c_{23} } } } } }\\{ { {a_{12} }\frac{ {\partial L} }{ {\partial {c_{11} } } } + {a_{22} }\frac{ {\partial L} }{ {\partial {c_{21} } } } } }&{ { {a_{12} }\frac{ {\partial L} }{ {\partial {c_{12} } } } + {a_{22} }\frac{ {\partial L} }{ {\partial {c_{22} } } } } }&{ { {a_{12} }\frac{ {\partial L} }{ {\partial {c_{13} } } } + {a_{22} }\frac{ {\partial L} }{ {\partial {c_{23} } } } } }\\{           { {a_{13} }\frac{ {\partial L} }{ {\partial {c_{11} } } } + {a_{23} }\frac{ {\partial L} }{ {\partial {c_{21} } } } } }&{ { {a_{13} }\frac{ {\partial L} }{ {\partial {c_{12} } } } + {a_{23} }\frac{ {\partial L} }{ {\partial {c_{22} } } } } }&{ { {a_{13} }\frac{ {\partial L} }{ {\partial {c_{13} } } } + {a_{23} }\frac{ {\partial L} }{ {\partial {c_{23} } } } } }\\{              { {a_{14} }\frac{ {\partial L} }{ {\partial {c_{11} } } } + {a_{24} }\frac{ {\partial L} }{ {\partial {c_{21} } } } } }&{ { {a_{14} }\frac{ {\partial L} }{ {\partial {c_{12} } } } + {a_{24} }\frac{ {\partial L} }{ {\partial {c_{22} } } } } }&{ { {a_{14} }\frac{ {\partial L} }{ {\partial {c_{13} } } } + {a_{24} }\frac{ {\partial L} }{ {\partial {c_{23} } } } } }\end{array} } \right]\)  </span></p>
		
	</span>
	<span class="ref-num">(15)</span>
</div>

<p>This can be rewritten as a matrix product.</p>

<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } } = \left[ {\begin{array}{*{20}{c} }{ {a_{11} } }&{ {a_{21} } }\\{ {a_{12} } }&{ {a_{22} } }\\{ {a_{13} } }&{ {a_{23} } }\\{ {a_{14} } }&{ {a_{24} } }\end{array} } \right]\left[ {\begin{array}{*{20}{c} }{\frac{ {\partial L} }{ {\partial {c_{11} } } } }&{\frac{ {\partial L} }{ {\partial {c_{12} } } } }&{\frac{ {\partial L} }{ {\partial {c_{13} } } } }\\{\frac{ {\partial L} }{ {\partial {c_{21} } } } }&{\frac{ {\partial L} }{ {\partial {c_{22} } } } }&{\frac{ {\partial L} }{ {\partial {c_{23} } } } }\end{array} } \right]\)</span><span class="ref-num">(16)</span>
</div>

<p> In fact, the first matrix is the transpose of \(\boldsymbol {A} \) and the second matrix is the upstream derivative \(\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\). Then we have  </p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } } = {\boldsymbol {A} ^T}\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\) </span><span class="ref-num">(17)</span>
</div>

<p> Let’s check the dimensions once more. On the left hand side of Equation (17), \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\) has a dimension of \(k \times n\), the same as \(\boldsymbol {B} \). On the right hand side, \({\boldsymbol {A} ^T}\) has a dimension of \(k \times m\) and \(\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\) has a dimension of \(m \times n\); therefore, their matrix product has a dimension of \(k \times n\) and matches that of \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\). </p>

<p> Again, the above derivations can be generalized to any matrix multiplication. If you have time, you can derive it by yourself, just make sure the subscript indices are correct. </p>

<h3><a name="_Custom_implementations_and_validation"></a>3. Custom implementations and validation  </h3>
<p>With the derived Equations (13) and (17), it is in fact pretty easy to implement the backward pass of matrix multiplication. Please see <a href="https://github.com/coolgpu/Demo_Matrix_Multiplication_backward/blob/master/Demo_MatrixMultiplication_backward.py">the example implementation on GitHub</a> for a network that simply takes the mean of the matrix product \(\boldsymbol {C}  = \boldsymbol {A}  \boldsymbol {B} \) as the loss. The core part is just a 3-line code as demonstrated below.  </p>

<pre class="pre-scrollable">
	<code class="python">
		
        grad_C_manual = (torch.ones(C.shape, dtype=torch.float64)/C.numel())
        grad_A_manual = grad_C_manual.mm(B.t())
        grad_B_manual = A.t().mm(grad_C_manual)

	</code>
</pre>

<p>The first line calculate the derivative of the loss w.r.t \(\boldsymbol {C} \) for the mean operation, which serves as the upstream gradient for \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) and \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\). </p>

<p>The second and third lines compute \(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } }\) and \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } }\) using the chain rule based on Equations (13) and (17), respectively. The function <b><i>t()</i></b> is a matrix transpose operation. </p>

<p>To validate our derivations and implementation, we compared these results with those from Torch built-in implementation via <b><i>loss.backward()</i></b> and they matched.  </p>


<h3><a name="_Summary"></a>4. Summary </h3>
<p>In this post, we demonstrated how to derive the gradients of matrix multiplication in neural networks. While the derivation steps seem complex, the final equations of the gradients are pretty simple and easy to implement:  </p>



<div class="alert alert-secondary equation">
	<span>\(\frac{ {\partial L} }{ {\partial \boldsymbol {A} } } = \frac{ {\partial L} }{ {\partial \boldsymbol {C} } }{\boldsymbol {B} ^T}\) and \(\frac{ {\partial L} }{ {\partial \boldsymbol {B} } } = {\boldsymbol {A} ^T}\frac{ {\partial L} }{ {\partial \boldsymbol {C} } }\)	</span><span class="ref-num"> </span>
</div>

<p> With matrix multiplication covered, we will move to Convolution in the next post. Please stay tuned. </p>



