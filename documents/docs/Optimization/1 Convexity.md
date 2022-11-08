# Convex Optimization

## Optimization Problems

Many problems in science and engineering can be formulated as an *optimization* problem:
$$
\text { minimize } f(\mathbf{x}) \quad \text { subject to } \quad \mathbf{x} \in \mathcal{C} \text {, }
$$
where $f$ is a cost function and $\mathcal{C}$ is a set. Due to $\max _{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})=\min _{\mathbf{x} \in \mathcal{C}}-f(\mathbf{x})$, this formulation includes the problem of maximizing a cost function. The function $f$ can model goodness of fit in machine learning (we'll see many examples later), utility, or cost. The set $\mathcal{C}$ can incorporate constraints such as a limited budget $(\|\mathbf{x}\| \leq B)$ or priors ( $\mathbf{x}$ is non-negative).

We say that $\mathbf{x}^*$ is a (global) solution to the optimization problem (1) if $\mathbf{x}^* \in \mathcal{C}$, and $f\left(\mathbf{x}^*\right) \leq f(\mathbf{x})$ for all $\mathrm{x} \in \mathcal{C}$. We say that $\mathrm{x}^*$ is a local solution to the optimization problem if in a neighborhood $\mathcal{N}$ around $\mathbf{x}^*, f\left(\mathbf{x}^*\right) \leq f(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{C} \cap \mathcal{N}$. Optimality can be very hard to check, even if $f$ is differentiable and $\mathcal{C}=\mathbb{R}^n$.

## Convexity

An important class of optimization problems for which we can check optimality rather easily are convex optimization problems. A standard reference for convex optimization problems is the book [BV04].

**Definition 1.** A convex set $\mathcal{C}$ is any set such that for all $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ and all $\theta \in(0,1)$,
$$
\theta \mathbf{x}+(1-\theta) \mathbf{y} \in \mathcal{C}
$$

![](https://i.imgur.com/zQzqblO.png)

<mark>连接set内任意两个点, 这两个点的连线上任意一点都在set内</mark>

Figure 1 shows examples of convex and non-convex sets. Intersections of convex sets are convex. Important examples of convex sets are the following:

- Subspaces $\left\{\mathbf{x}=\mathbf{U a}: \mathbf{a} \in \mathbb{R}^d\right\}$,
- Affines spaces $\left\{\mathbf{x}=\mathbf{U a}+\mathbf{b}: \mathbf{a} \in \mathbb{R}^d\right\}$
- Half-spaces $\{\mathbf{x}:\langle\mathbf{a}, \mathbf{x}\rangle \leq b\}$

**Definition 2.** for a set of points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^n$, a convex combination of them is defined as the set

$$
\left\{\mathbf{x}=\sum_{i=1}^n \theta_i \mathbf{x}_i: \sum_{i=1}^n \theta_i=1, \theta_i \geq 0\right\}
$$

<mark> A set $\mathcal{C}$ is convex if and only if it contains all convex combinations of its elements. </mark>

For any given set $\mathcal{S}$, the convex hull of $\mathcal{S}$ is defined as the set of all convex combinations of points in $\mathcal{S}$. Intuitively, it is the smallest convex set that contains $\mathcal{S}$. As an example, consider the non-convex set $\left\{\mathbf{x}:\|\mathbf{x}\|_0 \leq 1,\|\mathbf{x}\|_1 \leq 1\right\}$. Its convex hull is the $\ell_1$-norm ball $\left\{\mathbf{x}\right.$ : $\left.\|\mathbf{x}\|_1 \leq 1\right\}$.
We are now ready for one of the most important definitions in this class.

**Definition 3.** A function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is convex if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, and all $\theta \in(0,1)$,
$$
\theta f(\mathbf{x})+(1-\theta) f(\mathbf{y}) \geq f(\theta \mathbf{x}+(1-\theta) \mathbf{y}) .
$$
A function is strictly convex, if the inequality above is strict whenever $\mathbf{x} \neq \mathbf{y}$. 

A function $f$ is concave if $-f$ is convex.
Common examples of convex functions are:

- Linear functions: $f(\mathbf{x})=\langle\mathbf{a}, \mathbf{x}\rangle+b$,
- Quadratics: $f(\mathbf{x})=\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}+\mathbf{b}^T \mathbf{x}+c$, where $\mathbf{Q}$ is positive semidefinite. The function $f$ is convex if and only if $\mathbf{Q}$ is positive semidefinite.
- Any norm $f(\mathbf{x})=\|\mathbf{x}\|$ is convex. This follows from the triangle inequality and homogeneity/scalability.

<u>An important property of convex functions is that local minima are global.</u>

**Proposition 1.** Any local minimum of a convex function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is also a global minimum.
We now turn to first order optimality conditions. To this end, we first state a proposition stating that a function is convex if and only if it is lower bounded by its first order Taylor expansion at any point.

**Proposition 2.** A differentiable function $f$ is convex if and only if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,
$$
f(\mathbf{y}) \geq f(\mathbf{x})+\langle\mathbf{y}-\mathbf{x}, \nabla f(\mathbf{x})\rangle .
$$
It is strictly convex if and only if the inequality holds strictly for all $\mathbf{x} \neq \mathbf{y}$.
An important consequence is the following optimality condition.

**Corollary 1.** If a differentiable function $f$ is convex and $\nabla f\left(\mathbf{x}^*\right)=0$, then $\mathbf{x}^*$ is a global minimizer of $f$.

## Computational aspects of optimization algorithms
Suppose we want to minimize a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ over $\mathbb{R}^n$ :
$$
\underset{\mathbf{x} \in \mathbb{R}^n}{\operatorname{minimize}} f(\mathbf{x}) .
$$
For simplicity assume that $f(\mathbf{x})=\|\mathbf{A} \mathbf{x}-\mathbf{y}\|_2^2$, where $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a matrix with full column rank. So the optimization problem above is a standard least squares problem. If the columns are linearly independent, we can simply obtain a closed form solution as $\mathbf{x}^*=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{y}$. However, for many practically relevant functions $f$, we cannot compute a closed form solution. Even if we can, we might not want to because computing the closed form solution can be computationally expensive, e.g., for $\mathbf{x}^*=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{y}$ we have to compute the inverse of a possibly large matrix. In the first part of this course we consider algorithms for finding approximate solutions of the minimization problem above. In practice we are essentially always content with an approximate solution (a computer can only give us a solution up to machine precision anyways). We assume that we have oracle access to $f$:

- A zeroth order oracle returns $f(\mathbf{x})$ for a given $\mathbf{x}$.
- A first order oracle returns $\{f(\mathbf{x}), \nabla f(\mathbf{x})\}$ for a given $\mathbf{x}$.
- A second order oracle returns $\left\{f(\mathbf{x}), \nabla f(\mathbf{x}), \nabla^2 f(\mathbf{x})\right\}$ for a given $\mathbf{x}$.

Ultimately, we are interested in understanding the computational complexity of convex optimization algorithms, i.e., the number of flops to obtain a $\epsilon$-accurate solution (an $\epsilon$-accurate solution can be a vector $\mathbf{x}^k$ obeying $\left\|\mathbf{x}^k-\mathbf{x}^*\right\| \leq \epsilon$, or $\left.f\left(\mathbf{x}^k\right)-f\left(\mathbf{x}^*\right) \leq \epsilon\right)$. To this end, we study the oracle complexity of iterative algorithms to obtain approximate solutions to convex optimization problems. The number of flops can then be obtained as the number of oracle queries required to obtain an $\epsilon$-accurate solution times the complexity of the oracle.

A word of caution: Other properties than the oracle complexity can affect the runtime of algorithms in practice. For example, standard gradient descent has a worse convergence rate than the accelerated gradient method, but might be more sensitive to noise.


## Takeaway

**Convex Set**

1. 连接set内任意两个点, 这两个点的连线上任意一点都在set内

2. A set C is convex if and only if it contains all convex combinations of its elements.



**Convex Combination**

for a set of points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^n$, a convex combination of them is defined as the set


$$
\left\{\mathbf{x}=\sum_{i=1}^n \theta_i \mathbf{x}_i: \sum_{i=1}^n \theta_i=1, \theta_i \geq 0\right\}
$$



**Convex Hull of S**

For any given set S, the convex hull of S is defined as the set of all convex combinations of points in S. Intuitively, it is the smallest convex set that contains S

**Differentiable Function** 

A differentiable function $f$ is convex if and only if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,
$$
f(\mathbf{y}) \geq f(\mathbf{x})+\langle\mathbf{y}-\mathbf{x}, \nabla f(\mathbf{x})\rangle .
$$

