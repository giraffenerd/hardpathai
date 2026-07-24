7/5/2026

project time

logistic regression

$$
z^{(i)} = w^T x^{(i)} + b
$$

$$
\hat{y}^{(i)} = a^{(i)} = \operatorname{sigmoid}(z^{(i)})
$$

$$
\mathcal{L}(a^{(i)}, y^{(i)})
= -y^{(i)} \log(a^{(i)})
- (1 - y^{(i)}) \log(1 - a^{(i)})
$$

The cost is then computed by summing over all training examples:

$$
J = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(a^{(i)}, y^{(i)})
$$
