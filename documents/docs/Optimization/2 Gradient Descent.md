# Gradient Descent

Gradient descent is a simple iterative algorithm for minimizing a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ on $\mathbb{R}^n$. Starting from an initial point $\mathbf{x}_0$, gradient descent iterates:
$$
\mathbf{x}^{k+1}=\mathbf{x}^k-\alpha_k \nabla f\left(\mathbf{x}^k\right),
$$
where $\alpha_k$ is a step size parameter. Gradient descent converges to a local minimum, and provided that $f$ is convex, it converges to a global minimum. The idea behind this algorithm is to make small steps in the direction that minimizes the local first order approximation of $f$.

## Convergence for quadratic functions
In order to understand the gradient method better, let us start with a simple class of functions, namely quadratic functions:
$$
f(\mathbf{x})=\frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x}-\mathbf{b}^T \mathbf{x},
$$
where $\mathbf{Q}$ is symmetric and strictly positive definite. A closed form solution to the minimization problem minimize $\mathbf{f}(\mathbf{x})$ is $\mathbf{x}^*=\mathbf{Q}^{-1} \mathbf{b}$.

We want to understand what gradient descent yields for this problem. We consider gradient descent with a fixed stepsize:
$$
\mathbf{x}^{k+1}=\mathbf{x}^k-\alpha \nabla f\left(\mathbf{x}^k\right) .
$$
Using that the gradient is given by $\nabla f(\mathbf{x})=\mathbf{Q} \mathbf{x}-\mathbf{b}$ and that the optimal solution obeys $\mathbf{Q x}^*=\mathbf{b}$, the difference of the $(k+1)$-st iteration to the optimum is
![](https://i.imgur.com/Hh5bGNU.png)



It follows that

$$
\left\|\mathbf{x}^{k+1}-\mathbf{x}^*\right\|_2 \leq\|\mathbf{I}-\alpha \mathbf{Q}\|\left\|\mathbf{x}^k-\mathbf{x}^*\right\|_2
$$

Since $\mathbf{I}-\alpha \mathbf{Q}$ is symmetric (the first equality below can be checked by taking the singular value decomposition of the matrix)
$$
\|\mathbf{I}-\alpha \mathbf{Q}\|=\max \left(\lambda_{\max }(\mathbf{I}-\alpha \mathbf{Q}),-\lambda_{\min }(\mathbf{I}-\alpha \mathbf{Q})\right)=\max (\alpha M-1,1-\alpha m),
$$
where $M$ and $m$ are the largest and smallest singular values of the matrix $\mathbf{Q}$. For the second equality, we used that, the eigenvalues of $\mathbf{I}-\alpha \mathbf{Q}$ and $\mathbf{Q}$ are related as $\lambda_i(\mathbf{I}-\alpha \mathbf{Q})=1-\alpha \lambda_i(\mathbf{Q})^1$. The right hand side above is minimized by $\alpha=\frac{2}{M+m}$. For this choice, $\|\mathbf{I}-\alpha \mathbf{Q}\|=\frac{1-1 / \kappa}{1+1 / \kappa}<1$, where $\kappa=M / m$ is the condition number of the matrix $\mathbf{Q}$.
To summarize, we have proven the following proposition:

![](https://i.imgur.com/IXCyNUT.png)


## References
[BV04] S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.