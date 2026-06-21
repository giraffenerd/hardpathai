6/20/2026

2 days later, I am back, week2 of the deep learning course!

logistic regression, sigmod, why sigmod,

Because the model’s final score is linear addition:

z = w1*x1 + w2*x2 + b

A linear layer naturally adds evidence.

But odds naturally multiply.

So we use log because:

log turns multiplication into addition

That is the key.

Maybe feature 1 says:

pointy ears → cat is 3x more likely

Feature 2 says:

whiskers → cat is 4x more likely

Together, the odds should multiply:

3 × 4 = 12x more likely cat

But neural network linear layer wants to add:

w1*x1 + w2*x2

So we take log:

log(3 × 4) = log(3) + log(4)

features produce additive evidence
z = w\*x + b
z = log odds
sigmoid(z) = probability

Why log?

Because:

probability ratio / odds are multiplicative
linear model is additive
log converts multiplicative evidence into additive evidence

Lost function is single one like
Cost Function

# 7 Common Loss Functions and Cost Functions

## Loss Function vs Cost Function

A **loss function** measures error for **one training example**.

$$
L_i = L(y_i, \hat{y}_i)
$$

A **cost function** measures the average loss over the **whole training dataset**.

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m} L(y_i,\hat{y}_i)
$$

Where:

$$
m = \text{number of training examples}
$$

$$
y_i = \text{true label / true value}
$$

$$
\hat{y}_i = \text{model prediction}
$$

$$
\theta = \text{model parameters, such as } w,b
$$

Sometimes we add regularization:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}L(y_i,\hat{y}_i)+\lambda R(\theta)
$$

---

# 1. Mean Squared Error, MSE

Used for **regression**, for example house price prediction.

## Single-example loss

$$
L(y,\hat{y})=(y-\hat{y})^2
$$

## Cost function

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

## Explanation

MSE squares the prediction error.

Example:

$$
y=10,\quad \hat{y}=7
$$

$$
L=(10-7)^2=9
$$

Because the error is squared, large mistakes are punished more heavily.

---

# 2. Mean Absolute Error, MAE

Used for **regression**.

## Single-example loss

$$
L(y,\hat{y})=|y-\hat{y}|
$$

## Cost function

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

## Explanation

MAE measures the direct distance between prediction and truth.

Example:

$$
y=10,\quad \hat{y}=7
$$

$$
L=|10-7|=3
$$

Compared with MSE, MAE is less sensitive to very large errors.

---

# 3. Huber Loss

Used for **regression with outliers**.

Huber loss behaves like MSE for small errors and MAE for large errors.

Let:

$$
r=y-\hat{y}
$$

where \(r\) is the residual, or prediction error.

## Single-example loss

$$
L_{\delta}(y,\hat{y})=
\begin{cases}
\frac{1}{2}r^2, & \text{if } |r|\leq \delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & \text{if } |r|>\delta
\end{cases}
$$

## Cost function

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}L_{\delta}(y_i,\hat{y}_i)
$$

## Explanation

For small errors:

$$
\text{Huber uses squared error}
$$

For large errors:

$$
\text{Huber uses absolute-error-like behavior}
$$

This makes Huber more stable when the dataset has outliers.

---

# 4. Binary Cross-Entropy, BCE

Used for **binary classification**, for example:

$$
\text{cat}=1,\quad \text{not cat}=0
$$

The model output is usually:

$$
p=P(y=1|x)
$$

## Single-example loss

$$
L(y,p)=-\left[y\log(p)+(1-y)\log(1-p)\right]
$$

## Cost function

$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
$$

## Explanation

If the true label is:

$$
y=1
$$

then the loss becomes:

$$
L=-\log(p)
$$

If the model predicts:

$$
p=0.9
$$

then:

$$
L=-\log(0.9)\approx 0.105
$$

Good prediction, small loss.

If the model predicts:

$$
p=0.01
$$

then:

$$
L=-\log(0.01)\approx 4.605
$$

Bad prediction, large loss.

---

# 5. Categorical Cross-Entropy

Used for **multi-class classification**, for example:

$$
\text{cat, dog, bird}
$$

Usually used with **softmax**.

Let:

$$
K = \text{number of classes}
$$

$$
y_k = \text{true label for class } k
$$

$$
p_k = \text{predicted probability for class } k
$$

For one-hot labels:

$$
y=[1,0,0]
$$

means class 1 is correct.

## Single-example loss

$$
L(y,p)=-\sum_{k=1}^{K}y_k\log(p_k)
$$

Because only the correct class has:

$$
y_k=1
$$

this becomes:

$$
L=-\log(p_{\text{correct class}})
$$

## Cost function

$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}
\sum_{k=1}^{K}y_{ik}\log(p_{ik})
$$

## Explanation

Example:

True label is cat:

$$
y=[1,0,0]
$$

Model predicts:

$$
p=[0.8,0.15,0.05]
$$

Loss:

$$
L=-\log(0.8)\approx 0.223
$$

If model predicts:

$$
p=[0.05,0.9,0.05]
$$

Loss:

$$
L=-\log(0.05)\approx 2.996
$$

The model is punished when it gives low probability to the correct class.

---

# 6. Negative Log-Likelihood, NLL

Used for **probabilistic models**, classification, and language models.

## Single-example loss

$$
L(y,p)=-\log P(y|x)
$$

## Cost function

$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\log P(y_i|x_i;\theta)
$$

## Explanation

NLL punishes the model when it assigns low probability to the correct answer.

For classification, if the correct class is \(c\):

$$
L=-\log(p_c)
$$

For language models, if the correct next token is \(w_t\):

$$
L_t=-\log P(w_t|w_1,w_2,\ldots,w_{t-1})
$$

Meaning:

$$
\text{If the model gives high probability to the correct token, loss is small.}
$$

$$
\text{If the model gives low probability to the correct token, loss is large.}
$$

---

# 7. Hinge Loss

Used for **SVM-style classification**.

Labels are usually:

$$
y\in\{-1,+1\}
$$

The model gives a raw score:

$$
s=f(x)
$$

Prediction rule:

$$
s>0 \Rightarrow +1
$$

$$
s<0 \Rightarrow -1
$$

## Single-example loss

$$
L(y,s)=\max(0,1-ys)
$$

## Cost function

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\max(0,1-y_is_i)
$$

## Explanation

If the prediction is correct and confident:

$$
ys\geq 1
$$

then:

$$
L=0
$$

If the prediction is wrong or not confident enough:

$$
ys<1
$$

then the model is punished.

Example:

$$
y=+1,\quad s=2
$$

$$
L=\max(0,1-1\cdot 2)=0
$$

Good prediction.

But:

$$
y=+1,\quad s=0.2
$$

$$
L=\max(0,1-0.2)=0.8
$$

Correct direction, but not confident enough.

---

# Summary Table

| Loss Function             | Formula                                   | Common Use                 |
| ------------------------- | ----------------------------------------- | -------------------------- | -------------------------- | ---------- |
| Mean Squared Error, MSE   | $$(y-\hat{y})^2$$                         | Regression                 |
| Mean Absolute Error, MAE  | $$                                        | y-\hat{y}                  | $$                         | Regression |
| Huber Loss                | Piecewise MSE / MAE                       | Regression with outliers   |
| Binary Cross-Entropy      | $$-\left[y\log(p)+(1-y)\log(1-p)\right]$$ | Binary classification      |
| Categorical Cross-Entropy | $$-\sum_{k=1}^{K}y_k\log(p_k)$$           | Multi-class classification |
| Negative Log-Likelihood   | $$-\log P(y                               | x)$$                       | Probabilistic models, LLMs |
| Hinge Loss                | $$\max(0,1-ys)$$                          | SVM classification         |

---

# Most Important Ones for Neural Networks

## Regression

$$
\text{Regression} \Rightarrow \text{MSE}
$$

## Binary Classification

$$
\text{Binary classification} \Rightarrow \text{Binary Cross-Entropy}
$$

## Multi-Class Classification

$$
\text{Multi-class classification} \Rightarrow \text{Categorical Cross-Entropy}
$$

## Language Model

$$
\text{Language model} \Rightarrow \text{Negative Log-Likelihood / Cross-Entropy}
$$
