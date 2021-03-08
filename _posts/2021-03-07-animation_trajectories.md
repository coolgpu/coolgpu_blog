---
layout: post
mathjax: true
title:  "Animation of Earth-to-Mars Spacecraft Trajectory -- from Physics to Math to Programming"
date:   2021-03-07 21:00:00 -0700
categories: github pages
author: Xiyun Song, PhD
---

<p>Inspired by the great accomplishment of NASA’s Perseverance Rover landing on Mars<sup>[<a href="#_Reference1">1</a>]</sup>, we used Hohmann Transfer Orbit as the regress case study in the last post to demonstrate the fundamentals of artificial neural networks. In this post, we will build an animation of motions of the Earth and Mars, together with the trajectory of the spacecraft that is launching from the Earth, travelling along the Hohmann Transfer orbit and finally meeting Mars. This is a topic off neural network, but an extension from the previous post for fun. This post is organized into the following sections: assumption, physics, math, and implementation using Python and Matplotlib.</p>

<p>For your convenience, this post is organized into the following sections. </p>

<ul>
	<li><a href="#_Background">Background</a></li>
    <li><a href="#_Task">Task – Animation of the trajectories</a></li>
	<li><a href="#_Physics">Physics</a></li>
	<li><a href="#_Mathematics">Mathematics</a></li>
	<li><a href="#_Implementations">Implementations </a></li>
    <li><a href="#_Summary">Summary</a></li>
	<li><a href="#_References">References</a></li>
</ul>

<h3><a name="_Background"></a><span style="color:darkblue">1. Background</span></h3> 

<p>With the development in science and technology, there have been increased interest in Mars exploration by sending spacecraft there. During the Earth-to-Mars flight windows that open up every two years, such a spacecraft can be launched from the Earth with fast rocket to obtain enough speed so that it can escape from the pull of Earth’s gravity and becomes a tiny “planet” orbiting around the Sun. After traveling over about five hundred million kilometers along an elliptical orbit, the spacecraft will finally meet up with its target, Mars, if Mars moves to that location at the same time. At arrival, the spacecraft will slow down its speed relative to Mars by firing onboard rockets or with the help of martial atmosphere in order to be captured by the gravity of Mars. If not slowed down enough, the spacecraft can escape and miss Mars. </p>

<p>Due to the presence of gravity and the facts that both Earth and Mars are moving, the spacecraft is not aiming at where Mars is at launch time and fly in a straight line. Instead, it aims at a point ahead of the current location of Mars, so that when the spacecraft arrives that point, Mars also arrives at the exactly right time. </p>

<p>With different launch speed, the spacecraft can fly along a variety of elliptical paths and meet the orbit of Mars at different location. Among these single-elliptical orbit, Hohmann transfer orbit the most efficient one that requires the least launch fuel. Hohmann Transfer Orbit refers to the elliptical orbit that touches the Earth orbit at Hohmann perihelion at launch time and touches the Mars orbit at Hohmann aphelion at arrival time, as shown in Figure 1. </p>

<p align="center">
 <img src="{{ "/assets/images/Post6_Figure1_Static_orbits.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 1 Illustration of the rotating orbits of Earth, Mars and the spacecraft launch to Mars. 
</p> 

<p>Assumptions</p>

<ul>
	<li>Assume both Earth and Mars are rotating around the Sun in circular orbits with constant speeds. We use their individual average distances from the Sun as the radii \({R_{Earth}} = 1\;AU\) and \({R_{Mars} } = 1.52\;AU\)(Astronomical Unit, roughly the distance from Earth to the Sun, \(1.495978707 \times {10^{11} }\) m). Please note, in reality, their orbits are slightly elliptical, especially for Mars.</li>
	<li>Assume the orbits of Earth and Mars are in the same plane. </li>
	<li>Rocket acceleration at launch and deceleration at arrival are not taken into account in this animation.</li>
</ul>

<p>These assumptions are not exactly true, but quite close and reasonable. The purpose is to simplify calculation for the animation. </p>



<h3><a name="_Task"></a><span style="color:darkblue">2. Task – Animation of the trajectories</span></h3> 

<p>The animation task includes </p>
<ul>
	<li>Simulate the trajectory of the spacecraft that takes the Hohmann orbit.</li>
	<li>Also simulate the motion of both Earth and Mar at the same time. </li>
	<li>Also display information of the lapsed time, speed of the spacecraft, the distance from the spacecraft to Earth, Mars and Sun, respectively.</li>
</ul>


<h3><a name="_Physics"></a><span style="color:darkblue">3. Physics</span></h3> 

<p>A realistic animation depends on physics behinds the story. The most important ones that will be used in this example are Kepler's laws of planetary motion <sup>[<a href="#_Reference2">2</a>]</sup>. </p>

<ul>
	<li><strong>Kepler’s first law: The orbit of a planet is an ellipse with the Sun at one of the two foci. </strong>Based on this law, all orbits of the Earth, Mars and spacecraft are ellipses. However, as discussed in the assumptions, the orbit of the spacecraft will be exactly handled as ellipse in this animation, whereas the orbits of Earth and Mars are approximated to be circular to make the problem simpler. </li>
	<li><strong>Kepler’s second law: A line segment joining a planet and the Sun sweeps out equal areas during equal intervals of time.</strong> Because the time duration of each frame in our animation is the same, the second law means that, during every animation frame, the area swept by the line segment joining the spacecraft and the Sun is the same. This relationship provides us a relatively easy way to determine where the spacecraft should be located at each frame of the animation. </li>
	<li><strong>Kepler’s third law: The square of a planet's orbital period is proportional to the cube of the length of the semi-major axis of its orbit.</strong> This law provides the relationship of orbit radius and period among the three “planets” (Earth, Mars and the spacecraft). This is critical because we need this information to calculate when to launch and what’s the correction relative location between Mars and Earth so that the spacecraft and Mars will arrive at the same location at the same time. For example, this relative location is illustrated as the φ angle in Figure 1.</li>
</ul>

<p>Another import one is the orbital-energy-invariance law, also known as the vis-viva equation <sup>[<a href="#_Reference3">3</a>]</sup>, that models the motion of orbiting bodies. It is derived from the principle of conservation of mechanical energy and gives the relationship between the speed of the orbiting body (such as the spacecraft) and its distance to the Sun. We will skip the derivation, instead, just give the equation as follows:</p>

<div class="alert alert-secondary equation">
	<span> \({v^2} = GM\left( {\frac{2}{r} - \frac{1}{a} } \right)\)</span><span class="ref-num"> (1)</span>
</div>

<p>where \(v\) is the instant speed, \(r\) is the distance between the spacecraft and the Sun, \(a\) is the length of the semi-major axis of the elliptical orbit, \(G\) is the universal gravitational constant and \(M\) is the mass of the central object. For solar system, Because the product \(GM\) can be measured much more accurately than either factor alone, people often substitute the term \(GM\) in Equation (2) with their product \(\mu  = GM\). </p>

<div class="alert alert-secondary equation">
	<span> \(v = \sqrt {\mu \left( {\frac{2}{r} - \frac{1}{a} } \right)} \)</span><span class="ref-num"> (2)</span>
</div>

<p>For solar system, \({\mu _{Sun} } = 1.32712440018 \times {10^{20} }\;{m^3} \cdot {s^{ - 2} }\) <sup>[<a href="#_Reference4">4</a>]</sup>. This equation will be used to calculate the speed of the spacecraft during the flight. </p>


<h3><a name="_Mathematics"></a><span style="color:darkblue">4. Mathematics</span></h3> 

<h4><a name="_Physics_to_Math"></a><span style="color:darkblue">4.1.	Physics to Mathematics  </span></h4>

<p>Let’s translate these physics laws into math equations and then solve the problem using mathematic ways. </p>

<p>Based on Kepler's first law, the Hohmann Transfer orbit is an ellipse. So, the standard equation of its orbit as shown in Figure 1 is given by</p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ { { {\left( {x + C\;} \right)}^2} } }{ { {A^2} } } + \frac{ { {y^2} } }{ { {B^2} } } = 1\)</span><span class="ref-num"> (3)</span>
</div>

<p>where \(A = 1.261845\;AU\) is the length of the Hohmann semi-major axis and \(B = 1.234378\;AU\) is length of the Hohmann semi-minor axis, \(C = \sqrt { {A^2} - {B^2} } \) is half of the distance between the two foci. </p>

<p>Equation (3) is the expression in the \(x\)-\(y\) Cartesian coordinate system. The exactly equivalent expression of Equation (3) in the polar coordinate system is given by </p>

<div class="alert alert-secondary equation">
	<span> \(r\left( \theta  \right) = A \cdot \frac{ {1 - {e^2} } }{ {1 + e\cos \theta } }\) </span><span class="ref-num"> (4)</span>
</div>

<p> where \(r\) is the distance from a point on the ellipse (spacecraft) to the origin (Sun); \(\theta \) is the polar angle of the ray (from the origin to this point), defined to have 0° at the 3 o’clock direction (Hohmann perihelion), and to increase for rotations in counterclockwise orientation; \(e\) is the eccentricity of the ellipse and defined to be \(e = \frac{C}{A}\). Equation (4) gives the relationship between the distance and the polar angle. In fact, Equation (4) makes the calculation much easier than using Equation (3) for this animation task. </p>

<p>When the spacecraft flies along the elliptical orbit from its perihelion to its aphelion, its distance to the Sun is keep increasing. Based on the vis-viva equation, we know its linear speed is keeping decreasing, so is the angular speed. This means that we cannot simply move the spacecraft by the same amount of angular increment or linear increment for each equal-duration frame of the animation. Instead, we have to figure out where the spacecraft is at any specific frame (or equivalently, time). But how? Use Kepler's second law. </p>

<h4><a name="_Derivation_Ellipse"></a><span style="color:darkblue">4.2.	Math derivation of the elliptical trajectory of spacecraft  </span></h4>

<p>Let’s revisit the spacecraft’s flight from its perihelion to its aphelion. From Figure 1, we can see that 1) it is exactly the upper half ellipse; 2) the area swept by the line segment connecting the Sun (origin) and the spacecraft is half of the total area of the ellipse, \({S_{Half}} = \frac{1}{2}\pi AB\). </p>

<p>Animation is a collection of dynamic frames with each frame showing the objects at a location. When these frames played continuous, human visual system will interpret the objects to be moving, so it is called animation. </p>

<p>Based on Kepler’s second law: the areas  swept by the Sun-spacecraft ray between the time moments of two adjacent frames are all the same because the time intervals between all frames are the same. This means that the area of the half ellipse is equally shared by all frames. If we denote the number of frames as \(N\), we have 
</p>
	 

<div class="alert alert-secondary equation">
	<span> \({\Delta}{S_i} \equiv {\Delta} S = \frac{ { {S_{Half} } } }{N} = \frac{ {\pi AB} }{ {2N} }\)</span><span class="ref-num"> (5)</span>
</div>


<p align="center">
 <img src="{{ "/assets/images/Post6_Figure2_Hohmann_orbit.png" | relative_url }}" style="border:solid; color:gray" width="600"> 
<br>Figure 1 Illustration of the ellipse in polar coordinate system. 
</p> 

<p>One of such a small area is illustrated in Figure 2. This area is actually a fan formed by the radius \(r\left( { {\theta _i} } \right)\) at the \(i\)-th frame, the increment arc \(r \cdot \Delta{\theta _i}\), and the radius \(r\left( { {\theta _{i + 1} } } \right) \equiv r\left( { {\theta _i} + {\Delta \theta _i} } \right)\) at the next frame. When \(N\) is chosen to be big (e.g. \(N = 2070\)), the tiny arc can be treated as a line segment and the fan can be treated as a triangle. Therefore, we can calculate the area \(S\) using triangle area equation as follows:</p>

<div class="alert alert-secondary equation">
	<span> \(\Delta S = \frac{1}{2}r\left( { {\theta _i} } \right) \cdot r\left( { {\theta _i} + {\Delta \theta _i} } \right) \cdot \sin {\theta _i}\)</span><span class="ref-num"> (6)</span>
</div>

<p> Comparing Equations (5) and (6), we have </p>

<div class="alert alert-secondary equation">
	<span> \( \frac{1}{2}r\left( { {\theta _i} } \right) \cdot r\left( { {\theta _i} + {\Delta \theta _i} } \right) \cdot \sin {\theta _i} = \frac{ {\pi AB} }{ {2N} }\)</span><span class="ref-num"> (7)</span>
</div>

<p> Substituting \(r\left( { {\theta _i} } \right)\) with Equation (4) of ellipse in polar coordinates, we have </p>

<div class="alert alert-secondary equation">
	<span> \(\frac{1}{2}A\frac{ {1 - {e^2} } }{ {1 + e\cos {\theta _i} } } \cdot A\frac{ {1 - {e^2} } }{ {1 + e\cos \left( { {\theta _i} + {\Delta \theta _i} } \right)} } \cdot \sin {\Delta \theta _i} = \frac{ {\pi AB} }{ {2N} }\) </span><span class="ref-num"> (8)</span>
</div>

<p>Let’s simply the equation by dropping the subscript \(i\) (because \(\theta \) can be any polar angle) and some simply re-arrangement of the equation, then we have</p>

<div class="alert alert-secondary equation">
	<span> \(NA{\left( {1 - {e^2}} \right)^2}\sin \Delta \theta  = \pi B\left( {1 + e\cos \theta } \right)\left[ {1 + e\cos \left( {\theta  + \Delta \theta } \right)} \right]\) </span><span class="ref-num"> (9)</span>
</div>

<p> Using Trigonometric Addition Formulas to expand \(\cos \left( {\theta  + \Delta \theta } \right)\), we have </p>

<div class="alert alert-secondary equation">
	<span> \(NA{\left( {1 - {e^2} } \right)^2}\sin \Delta \theta  = \pi B\left( {1 + e\cos \theta } \right)\left[ {1 + e\left( {\cos \theta \cos \Delta \theta  - \sin \theta \sin \Delta \theta } \right)} \right]\) </span><span class="ref-num"> (10)</span>
</div>

<p> What’s the purpose to do all of these derivations? We want to find, given the current  polar angle \(\theta \), what the angular increment \(\Delta \theta \) should be. If we have the relationship between \(\Delta \theta \) and \(\theta \), we will know how much increment we should add to the angle of the current frame to obtain the angle of the next frame. </p>

<p>Almost there, let continue to solve Equation (9) for \(\Delta \theta \). In fact, Equation (9) can be solved exactly using trigonometry and then the solution of quadratic equation. However, that’s a little bit overkilling this task. For the precision of this animation, we can make a reasonable approximation to greatly simplify that process. </p>

<p>Taylor series expansions for the cosine and sine functions are</p>

<div class="alert alert-secondary">
	<p class="equation"><span> \(\cos \Delta \theta  = 1 - \frac{ { {\Delta \theta ^2} } }{ {2!} } +  \cdots \) </span><span class="ref-num"> (11) </span></p>
		
	<p class="equation"><span> \(\sin \Delta \theta  = \Delta \theta  - \frac{ { {\Delta \theta ^3} } }{ {3!} } +  \cdots \) </span><span class="ref-num"> (12)	</span></p>
</div>


Please note that \(\Delta \theta \) is very small, so we can ignore all terms of \({\Delta \theta ^2}\) and higher order, then Equations (11) and (12) become very simple as follows

<div class="alert alert-secondary">
	<p class="equation"><span> \(\cos \Delta \theta  = 1\) </span><span class="ref-num"> (13) </span></p>
		
	<p class="equation"><span> \(\sin \Delta \theta  = \Delta \theta \) </span><span class="ref-num"> (14)	</span></p>
</div>



Substituting Equations (13) and (14) back into Equation (10), we have 


<div class="alert alert-secondary equation">
	<span> \(NA{\left( {1 - {e^2}} \right)^2} \cdot \Delta \theta  = \pi B\left( {1 + e\cos \theta } \right)\left[ {1 + e\left( {\cos \theta  - \sin \theta \cdot \Delta \theta } \right)} \right]\) </span><span class="ref-num"> (15)</span>
</div>

<p>Equation (15) is just a simple linear equation of \(\Delta \theta \). We can solve it for \(\Delta \theta \) easily</p>

<div class="alert alert-secondary equation">
	<span> \(\Delta \theta  = \frac{ {\pi B{ {\left( {1 + e\cos \theta } \right)}^2} } }{ {NA{ {\left( {1 - {e^2} } \right)}^2} + \pi B\left( {1 + e\cos \theta } \right) \cdot e \cdot \sin \theta } }\) </span><span class="ref-num"> (16)</span>
</div>

<p>Once \(\Delta \theta \) for the current angular position has been obtained, the new angular position of the next frame can be simply calculated as follows </p>

<div class="alert alert-secondary equation">
	<span> \({\theta _{Next}} = {\theta _{Current}} + \Delta \theta \) </span><span class="ref-num"> (17)</span>
</div>

<p>The x- and y-coordinates can be easily calculated as follows</p>

<div class="alert alert-secondary">
	<p class="equation"><span> \({x_{Next}} = r\left( { {\theta _{Next} } } \right)\cos {\theta _{Next} }\) </span><span class="ref-num"> (18) </span></p>
		
	<p class="equation"><span> \({y_{Next} } = r\left( { {\theta _{Next} } } \right)\sin {\theta _{Next} }\) </span><span class="ref-num"> (19)	</span></p>
</div>

<p>Equations (16) through (19) allow us to animate the elliptical trajectory of the spacecraft from the beginning, then find the coordinates for the next frame, one followed by another one, and so on. </p>


<h4><a name="_Circular"></a><span style="color:darkblue">4.3.	Math derivation of the circular trajectory of Earth and Mars  </span></h4>


<p>For Earth and Mars, the derivation is much simpler because their orbits are circular and the speeds are constant. What we need to do is to find out their starting locations and the ending locations that correspond to the time duration of the spacecraft flight from Hofmann Perihelion to Hofmann Aphelion. This information can be derived based on Kepler’s third law:</p>

<div class="alert alert-secondary equation">
	<span> \(\frac{ { {T_E}^2} }{ { {R_E}^3} } = \frac{ { {T_M}^2} }{ { {R_M}^3} } = \frac{ { {T_S}^2} }{ { {R_S}^3} } \equiv Constan{t_{SolarSystem} }\) </span><span class="ref-num"> (20)</span>
</div>

<p>Where \(T\) denote the orbiting period, and \(R\) is the radius or semi-major axis of the orbits. The subscripts \(E\), \(M\) and \(S\) denote Earth, Mars and the spacecraft, respectively. </p>

<p>So if the angular range of the spacecraft flight is \({ {\rm{\Phi } }_S}\), the angular range that Earth travels during the same flight duration \({T_0}\)  is </p>

<div class="alert alert-secondary equation">
	<span> \({ {\rm{\Phi } }_E} = 2\pi \frac{ { {T_0} } }{ { {T_E} } } = 2\pi \frac{ { {T_0} } }{ { {T_S}\sqrt {\frac{ { {R_E}^3} }{ { {R_S}^3} } } } } = 2\pi \frac{ { {T_0} } }{ { {T_S} } }\sqrt {\frac{ { {R_S}^3} }{ { {R_E}^3} } } \) </span><span class="ref-num"> (21)</span>
</div>

<p>For the spacecraft, it flies exactly half of the elliptical orbit, so \(\frac{ { {T_0} } }{ { {T_S} } } = \frac{1}{2}\). Substituting this into Equation (20), we have </p>

<div class="alert alert-secondary equation">
	<span> \({ {\rm{\Psi } }_E} = \pi \sqrt {\frac{ { {R_S}^3} }{ { {R_E}^3} } } \) </span><span class="ref-num"> (22)</span>
</div>

<p>Similarly, the angular range that Mars travels during the same flight duration \({T_0}\)  is </p>

<div class="alert alert-secondary equation">
	<span> \({ {\rm{\Psi } }_M} = \pi \sqrt {\frac{ { {R_S}^3} }{ { {R_M}^3} } } \) </span><span class="ref-num"> (23)</span>
</div>

<p>Please note that, Earth share the same starting location with the spacecraft and whereas Mars share the same starting location with the spacecraft. Therefore, the starting and ending angular locations of Earth and Mars, are</p>

<div class="alert alert-secondary equation">
	<span>
		<p><span> \({\phi _{E,Start} } = 0\) </span></p>

		<p><span> \({\phi _{E,End} } = { {\rm{\Psi } }_E}\)	</span></p>

		<p><span> \({\phi _{M,Start} } = \pi  - { {\rm{\Psi } }_M}\) </span></p>

		<p><span> \({\phi _{M,End} } = \pi \)	</span></p>

	</span>
	<span class="ref-num">(24)</span>
</div>

<p>Using the approximated radii, Equation (24) gives  \({\phi _{M,Start} } = 44.3^\circ \).</p>

<p>Then animation frames can simply evenly samples between the starting and ending angular locations for Earth and Mars, respectively.</p>


<h4><a name="_Stats"></a><span style="color:darkblue">4.4.	Math derivation of the flight stats of spacecraft  </span></h4>

<p>For each animation frame, the “real-time” speed of the spacecraft \({v_S}\) can be calculated as follows:</p>

<div class="alert alert-secondary equation">
	<span> \({v_S} = \sqrt { {\mu _{Sun} }\left( {\frac{2}{ {r\left( \theta  \right)} } - \frac{1}{A} } \right)} \) </span><span class="ref-num"> (25)</span>
</div>

<p>The “real-time” travel distance of the spacecraft \({L_S}\) is approximated as an integration of the arc length over individual frames. </p>

<div class="alert alert-secondary equation">
	<span> \({L_{S,i + 1} } = {L_{S,i} } + {\Delta L_{S,i} } = {L_{S,i} } + \frac{1}{2}\left( {r\left( { {\theta _i} } \right) + r\left( { {\theta _i} + {\Delta \theta _i} }  \right)} \right) \cdot {\Delta \theta _i}\) </span><span class="ref-num"> (25)</span>
</div>

<p>where the subscript \(i\) denotes the \(i\)-th frame of the animation. </p>

<p>The “real-time” distances from the spacecraft to Earth, Mars and Sun can be simply calculated based on their current locations. </p>


<h3><a name="_Implementations"></a><span style="color:darkblue">5.	Implementations  </span></h3>

<p>The animation is implemented in Python using the Matplolib package <sup>[<a href="#_Reference5">5</a>]</sup>. The complete <a href="https://github.com/coolgpu/Animation_Earth_Mars_Hohmann_Orbits/blob/main/Animation_of_Earth_Mars_H_trajectories.py">source code can be found on GitHub</a>.</p>

<p>For convenience, the x- and y-coordinates of Earth, Mars and the spacecraft of all animation frames are pre-computed, instead of on-the-fly calculation at the update of each frame, using the Equations derived in the last section. It is equivalent, but avoid duplicated calculation when the animation repeats again and again. The travel length is also handled in a similar way. The interface of the function is </p>

<pre class="pre-scrollable">
	<code class="python">
def generate_animation_data(N=2070, R_Earth=1.0, Angle_Earth_0_deg = 0, R_Mars=1.5237, Angle_Mars_0_deg = None)
    ...
	</code>
</pre>

<p> \(N\) is the number of frames in the animation to generate. <code class="python">Angle_Mars_0_deg</code> is the \({\phi _{M,Start} }\) in Equation (24). If not passed in via the argument, it will be automatically calculated using Equation (24). This function returns the lists of the coordinates and the travel length values of the animation frames. </p>

<p>After data are generated, the script sets up the figure and axes, and plots some basic elements such as the static orbits in dashed lines as background, annotations of the orbits, etc. </p>

<p>Then, for those plotting artists that must be updated during each frame of the animation such as the dots representing the objects, their trajectories, and the status information texts, the script creates the artists with empty value so that they are invisible before the animation starts. </p>

<p>The <code class="python">init()</code> function is also defined to reset the data buffer and text contents to empty. It must return all plotting artists that need to be refreshed during animation. </p>

<p>The most important core part is defined as the function <code class="python">animate(i)</code>, where \(i\) is the argument to specify this is for the update of the \(i\)-th frame. The <code class="python">if i == 0:</code> block is to reset the data buffer so that the old plots will be cleared when the animation continue to play repeatly. Without this block, the new animation will be messed up by the old animation. </p>

<p>The rest lines in the <code class="python">animate(i)</code> function just fetch data from the pre-computed lists and update the data buffer for the plotting artists, or on-the-fly compute the information of the days, speed and distances. Again, those pre-computations can also be moved here to achieve the same effect equivalently. Just like the <code class="python">init()</code> function, the must return all plotting artists that need to be refreshed. Otherwise, the missed artists won’t be refreshed. </p>

<pre class="pre-scrollable">
	<code class="python">
# animation step function
def animate(i):
    if i == 0:  # clean up the previous plots
        xlist_Hohmann[:] = []
        ylist_Hohmann[:] = []
        xlist_Earth[:] = []
        ylist_Earth[:] = []
        xlist_Mars[:] = []
        ylist_Mars[:] = []

    x_Hohmann, y_Hohmann = Xs_Hohmann[i], Ys_Hohmann[i]
    xlist_Hohmann.append(x_Hohmann)
    ylist_Hohmann.append(y_Hohmann)
    point_Hohmann.set_data(x_Hohmann, y_Hohmann)
    line_Hohmann.set_data(xlist_Hohmann, ylist_Hohmann)

    x_Earth, y_Earth = Xs_Earth[i], Ys_Earth[i]
    xlist_Earth.append(x_Earth)
    ylist_Earth.append(y_Earth)
    point_Earth.set_data(x_Earth, y_Earth)
    line_Earth.set_data(xlist_Earth, ylist_Earth)

    x_Mars, y_Mars = Xs_Mars[i], Ys_Mars[i]
    xlist_Mars.append(x_Mars)
    ylist_Mars.append(y_Mars)
    point_Mars.set_data(x_Mars, y_Mars)
    line_Mars.set_data(xlist_Mars, ylist_Mars)

    day = i // 4
    text_day.set_text('Days = {:3d}'.format(day))
    travel_length_Hohmann = int(TravelLen_Hohmann[i] * AU)

    dist_2_Earth = int(math.sqrt((x_Hohmann-x_Earth)**2+(y_Hohmann-y_Earth)**2) * AU)
    text_dist_2_Earth.set_text('Distance to Earth = {:,d} km'.format(dist_2_Earth))
    dist_2_Mars = int(math.sqrt((x_Hohmann - x_Mars) ** 2 + (y_Hohmann - y_Mars) ** 2) * AU)
    text_dist_2_Mars.set_text('Distance to Mars = {:,d} km'.format(dist_2_Mars))
    dist_2_Sun = int(math.sqrt((x_Hohmann) ** 2 + (y_Hohmann) ** 2) * AU)
    text_dist_2_Sun.set_text('Distance to Sun = {:,d} km'.format(dist_2_Sun))
    uSun = 1.32712440018E20 # m^3/s^2
    A = (R_Mars + R_Earth) / 2 * AU
    speed = math.sqrt(uSun * (2/dist_2_Sun/1000 - 1/A/1000) )  # m/s
    speed /= 1000  # km/s
    text_spacecraft_speed.set_text('Spacecraft\'s speed = {:.3f} km/s'.format(speed))
    text_spacecraft_travellen.set_text('Spacecraft has traveled $ \\approx $ {:,d} km'.format(travel_length_Hohmann))

    return point_Hohmann, line_Hohmann, point_Earth, line_Earth, point_Mars, line_Mars, \
           text_day, text_dist_2_Mars, text_dist_2_Sun, text_dist_2_Earth, text_spacecraft_speed, text_spacecraft_travellen
	</code>
</pre>

<p>After everything has been properly set up, simply call the following code to start the animation. </p>

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, interval=intervalms, blit=True)

<p>Among the arguments, init and animate are the initialization and update functions we just discussed above. <code class="python">Frames=N</code> tells the animation engines to automatically trigger the animate function with \(i = 0,1, \cdots ,N - 1\), sequentially. Alternatively, you can also pass a list of integer instead of a single integer \(N\) to Frames, in which case the animate function will be triggered with each of the element in the list. The <code class="python">interval</code> argument specifies the delay (in milliseconds) between frames. This impacts the animation display on the screen only, but won’t impact the frames-per-second when saving to a video file. </p>

<p>Last, if we want to save the animation to external files (e.g. mp4), we uncomment the following line</p>

<pre class="pre-scrollable">
	<code class="python">
anim.save('Animation_Of_Planet_Orbits.mp4', fps=30, bitrate=1800, extra_args=['-vcodec', 'h264']) </p>
	</code>
</pre>


<p>More <code class="python">extra_args</code> can be passed in to specify the format of the output video. If not specified, the default settings will be used. To support the save feature, ffmpeg <sup>[<a href="#_Reference6">6</a>]</sup> need to be installed and can be found here. </p>

<p>The video below shows the result of this animation project. Enjoy. Just in case the video cannot be played in your browser, please try it with another browser such as firefox, Chrome or IE. You can also download the <a href="{{ "/assets/images/Post6_video1_Animation_Of_Planet_Orbits.mp4" | relative_url }}">mp4 video here and then play it. </p> 


<p align="center">
<video width="640" height="640" controls>
<source src="{{ "/assets/images/Post6_video1_Animation_Of_Planet_Orbits.mp4" | relative_url }}" type="video/mp4">
</video> 
<br>Figure 3: Animation of the trajectories of Earth, Mars and the spacecraft taking Hohmann Tranfer Orbit. 
</p>
<br>


<h3><a name="_Summary"></a><span style="color:darkblue">6. Summary </span></h3> 
<p>In this post, we demonstrated how to animate the trajectories of Earth, Mars and the spacecraft launched from Earth to Mars via the Hohmann Transfer Orbit. This project involves interesting knowledge in physics behind the story, math for solving the problem, and programming to implement the solution. So we went over the details and steps from physics to math to programming. In the end, the objects “really fly in the animation” as designed and expected. Cheers! </p>

<p>And, our salute to those pioneers, scientists, engineering, workers and all people who worked hard to make those accomplishments happen, including the Perseverance Rover and a lot more. 
</p>


<h3><a name="_References"></a><span style="color:darkblue">7. References</span></h3> 
<ul>
    <li><a name="_Reference1"></a>[1] <a href="https://www.nasa.gov/perseverance"> NASA Perseverance Rover landing on Mars. </a></li>

    <li><a name="_Reference2"></a>[2] <a href="https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion">wikipedia Kepler's laws of planetary motion.</a></li>

    <li><a name="_Reference3"></a>[3] <a href="https://en.wikipedia.org/wiki/Vis-viva_equation ">Wikipedia Vis-viva equation.</a></li>

    <li><a name="_Reference4"></a>[4] <a href="https://en.wikipedia.org/wiki/Gravitational_constant#Value_and_uncertainty">Wiki Gravitational constant</a></li>   

	<li><a name="_Reference5"></a>[5] <a href="https://matplotlib.org/">Matplotlib</a></li>

	<li><a name="_Reference6"></a>[6] <a href="https://www.ffmpeg.org/"> ffmpeg </a></li>
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
