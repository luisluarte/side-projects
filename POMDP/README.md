# **Hierarchical Bayesian Reinforcement Learning (Uncertainty-Driven Exploration)**

This repository contains a hierarchical Bayesian reinforcement learning model. It takes the same basic principles from a standard Rescorla-Wagner model but increases the complexity a bit to directly "catch" uncertainty-driven exploration, along with motor perseveration (stickiness) and the dynamic value of waiting.\
Basically, the pipeline takes raw lickometer data, discretizes it into 25ms chunks without losing detail, and feeds it into a custom physics engine written in Stan.

## **Pipeline**

The analysis runs through three main scripts sequentially. Here is what each one does:

### **1. create_discretization.R**

Turn raw event data into a uniform 25ms time-series grid.

-   Raw lickometer data just gives you timestamps for when a sensor is triggered. To run a continuous-time model, we need a grid. This script chops the timeline into 25ms steps and assigns a state (Idle, Nosepoking, Armed, Reward) and an action (Wait, Lick1, Lick2) to every single tick.\
-   **Input Files:** \* ../metadata/lickometer_metadata.csv (Drug groups, contexts, etc.)
    -   Raw CSV data inside ../data/lickometer_raw/\
-   **Output:** ../data/processed/discrete_data.rds

### **2. run_analysis.R**

Deal with optimization-related stuff, prep the data for Stan, and run the sampler.

-   This script takes the discrete data and maps the states/actions to numeric IDs. The big trick here is **Wait-Step Compression**. Otherwise, the model would be looking at a huge amount of "waiting" without any reason, drastically slowing down the CPU. Consecutive wait steps are rolled together and assigned a weight based on how long the animal waited. After compressing the data, it compiles the C++ model and runs Pathfinder (for fast initial estimates) followed by NUTS MCMC (the actual sampler).\
-   **Input Files:** discrete_data.rds and the .stan model file.\
-   **Output:** The final Stan data object, plus the fit_stabilized_final.rds posterior draws.

### **3. pomdp_model.stan** 

Compute the likelihood of the animal's actions given the parameters.

-   **The Logic:** This is where the hot loop is. The Bayesian sampler works by making a proposal "step" for a parameter and looking at the likelihood of that step given the data. Doing this over all animals natively is very inefficient, so the script uses reduce_sum to compute partial likelihoods in parallel (getting about \~2 animals per thread on a good CPU). It steps through the trials, updates the expected values, and evaluates the competing Q-values for each action.

## **Stan parts**

1.  **data**: the arrays of licks and waits.\
2.  **parameters**: global parameters, plus 'traits' which are just the random effects each animal has as a starting point.\
3.  **transformed parameters**: Where we map parameters to the experimental design (Baseline + Vehicle Shift + Drug Delta).\
4.  **model**: The priors (logical boundaries for the math) and the likelihood function.

## **The Reinforcement Learning Model**

At its core, the model asks at every 25ms tick: *"Based on what the rat knows about the world, what is the most valuable action it can take right now?"*

### **1. Value and Uncertainty (Beta-Bernoulli)**

Instead of a fixed learning rate tracking a single value, the model explicitly tracks uncertainty using a Beta distribution. $\\alpha$ tracks rewards, $\\beta$ tracks non-rewards.\
The Expected Value (EV) is straightforward—nothing fancy here:

$$EV\_i \= \\frac{\\alpha\_i}{\\alpha\_i \+ \\beta\_i}$$\
But because we know the distribution, we can formally calculate the **Shannon Entropy (**$U$**)**, which represents how flat or uncertain the environment is:

$$U\_i \= \-p\_i \\log(p\_i) \- (1-p\_i) \\log(1-p\_i)$$

### **2. The Q-Values (The Competing Drives)**

When the animal just licked Spout 1, it has to decide whether to stay or switch.\
What drives the animal to stick to the same spout is motor perseveration or "stickiness" ($\\phi$).

$$Q\_{\\text{stay}} \= EV\_1 \\cdot \\gamma \+ \\phi$$\
What drives the animal to change spouts is the uncertainty of the *other* spout, scaled by its drive to explore ($\\kappa$).

$$Q\_{\\text{switch}} \= EV\_2 \\cdot \\gamma \+ \\kappa \\cdot U\_2$$\
*(Note:* $\\gamma$ *is just a fixed scalar to keep the objective EV and subjective parameters in a nice neighborhood for the optimizer).*

### **3. The Value of Waiting (Continuous Time)**

The animal is not head-fixed; it roams around. So we model the value of waiting/roaming. This value starts at a baseline ($\\beta\_{\\text{base}}$) and changes over time based on an impatience slope ($\\beta\_{\\text{slope}}$):

$$Q\_{\\text{wait}} \= \\beta\_{\\text{base}} \+ \\beta\_{\\text{slope}} \\cdot \\ln(1 \+ t)$$\
**The Calculus Trick:** If an animal waits for 2 seconds, that's \~80 steps. Computing the softmax for every single step is insane computation. To avoid this, if the animal waits a long time, the script uses Simpson's 1/3 Rule. It takes the start, mid, and end points of the wait period and integrates the area under the curve to get the likelihood in $\\mathcal{O}(1)$ time:

$$\\sum\_{k=1}^{W} P(Wait\_k) \\approx \\frac{W}{6} \\left\[ P(Wait\_{\\text{start}}) \+ 4 P(Wait\_{\\text{mid}}) \+ P(Wait\_{\\text{end}}) \\right\]$$

### **4. Action Selection & Global Noise (**$\\epsilon$)

The Q-values are turned into action probabilities using a standard Softmax:

$$P(A\_i) \= \\frac{e^{Q\_i}}{\\sum e^{Q\_j}}$$\
However, I added a global randomness parameter ($\\epsilon$). This prevents super weird results when both spouts pay out at 100%. When an animal switches spouts in a 100/100 context, we ask: is it doing some complex calculation, or was it just a random slip?\
The final probability is a mixture of pure random chance (epsilon) and the softmax output:

$$P(\\text{Action}) \= \\epsilon \\cdot P\_{\\text{random}} \+ (1 \- \\epsilon) \\cdot P\_{\\text{softmax}}$$\
This allows the model to treat silly mistakes as $\\epsilon$-related noise rather than inventing a stupidly high $\\kappa$ value to explain it!
