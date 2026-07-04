7/1/2026

1.Logistic regression with gradient decent example

2.vectorization

# Matrix Operations Cheat Sheet for Machine Learning

## 1. Matrix Shape

A matrix is a 2D array of numbers.

$$
A \in \mathbb{R}^{m \times n}
$$

means:

- \(m\) rows
- \(n\) columns

Example:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

Shape:

$$
A \in \mathbb{R}^{2 \times 3}
$$

---

## 2. Vector Shape

A column vector:

$$
x \in \mathbb{R}^{n \times 1}
$$

Example:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
$$

A row vector:

$$
x^T \in \mathbb{R}^{1 \times n}
$$

Example:

$$
x^T =
\begin{bmatrix}
x_1 & x_2 & x_3
\end{bmatrix}
$$

The symbol \(T\) means **transpose**.

---

## 3. Matrix Addition

Two matrices can be added only if they have the same shape.

If:

$$
A, B \in \mathbb{R}^{m \times n}
$$

then:

$$
C = A + B
$$

and:

$$
C_{ij} = A_{ij} + B_{ij}
$$

Example:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
+
\begin{bmatrix}
10 & 20 \\
30 & 40
\end{bmatrix}
=
\begin{bmatrix}
11 & 22 \\
33 & 44
\end{bmatrix}
$$

Rule:

$$
(m \times n) + (m \times n) = m \times n
$$

---

## 4. Scalar Multiplication

A scalar is a single number.

If:

$$
c \in \mathbb{R}
$$

and:

$$
A \in \mathbb{R}^{m \times n}
$$

then:

$$
cA
$$

means multiply every element of \(A\) by \(c\).

Example:

$$
2
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
=
\begin{bmatrix}
2 & 4 \\
6 & 8
\end{bmatrix}
$$

---

## 5. Dot Product

Dot product takes two vectors and returns one number.

If:

$$
a, b \in \mathbb{R}^{n}
$$

then:

$$
a \cdot b = a^T b
$$

Formula:

$$
a^T b = \sum_{i=1}^{n} a_i b_i
$$

Example:

$$
a =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix},
\quad
b =
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

$$
a^T b = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6
$$

$$
a^T b = 4 + 10 + 18 = 32
$$

In ML, one neuron often computes:

$$
z = w^T x + b
$$

where:

- \(w\) = weights
- \(x\) = input features
- \(b\) = bias
- \(z\) = raw score

---

## 6. Matrix Multiplication

If:

$$
A \in \mathbb{R}^{m \times n}
$$

and:

$$
B \in \mathbb{R}^{n \times p}
$$

then:

$$
C = AB
$$

and:

$$
C \in \mathbb{R}^{m \times p}
$$

Shape rule:

$$
(m \times n)(n \times p) = m \times p
$$

The inner dimensions must match.

Each output element is:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

Meaning:

$$
C_{ij} = \text{row } i \text{ of } A \cdot \text{column } j \text{ of } B
$$

Example:

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\quad
B =
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
$$

$$
AB =
\begin{bmatrix}
1\cdot5 + 2\cdot7 & 1\cdot6 + 2\cdot8 \\
3\cdot5 + 4\cdot7 & 3\cdot6 + 4\cdot8
\end{bmatrix}
$$

$$
AB =
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

Important:

$$
AB \neq BA
$$

Matrix multiplication order matters.

---

## 7. Matrix-Vector Multiplication

If:

$$
W \in \mathbb{R}^{m \times n}
$$

and:

$$
x \in \mathbb{R}^{n \times 1}
$$

then:

$$
z = Wx
$$

and:

$$
z \in \mathbb{R}^{m \times 1}
$$

Example:

$$
W =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

$$
x =
\begin{bmatrix}
10 \\
20 \\
30
\end{bmatrix}
$$

$$
Wx =
\begin{bmatrix}
1\cdot10 + 2\cdot20 + 3\cdot30 \\
4\cdot10 + 5\cdot20 + 6\cdot30
\end{bmatrix}
$$

$$
Wx =
\begin{bmatrix}
140 \\
320
\end{bmatrix}
$$

In a neural network layer:

$$
z = Wx + b
$$

$$
a = f(z)
$$

where:

- \(W\) = weight matrix
- \(x\) = input vector
- \(b\) = bias vector
- \(f\) = activation function
- \(a\) = activation output

---

## 8. Transpose

Transpose flips rows and columns.

If:

$$
A \in \mathbb{R}^{m \times n}
$$

then:

$$
A^T \in \mathbb{R}^{n \times m}
$$

Example:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

$$
A^T =
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
$$

Useful rules:

$$
(A^T)^T = A
$$

$$
(A + B)^T = A^T + B^T
$$

$$
(AB)^T = B^T A^T
$$

Notice that the order reverses in:

$$
(AB)^T = B^T A^T
$$

---

## 9. Identity Matrix

The identity matrix is like the number \(1\) for matrix multiplication.

For a \(2 \times 2\) matrix:

$$
I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

For compatible matrix \(A\):

$$
AI = A
$$

$$
IA = A
$$

In general:

$$
I_n \in \mathbb{R}^{n \times n}
$$

---

## 10. Inverse Matrix

For a square matrix \(A\), the inverse is written:

$$
A^{-1}
$$

It satisfies:

$$
A^{-1}A = I
$$

and:

$$
AA^{-1} = I
$$

Only some square matrices have inverses.

If a matrix has no inverse, it is called **singular**.

In ML, we usually avoid directly computing inverse because it can be expensive or numerically unstable.

---

## 11. Element-Wise Multiplication

Element-wise multiplication is different from matrix multiplication.

It is also called the **Hadamard product**.

If:

$$
A, B \in \mathbb{R}^{m \times n}
$$

then:

$$
C = A \odot B
$$

where:

$$
C_{ij} = A_{ij}B_{ij}
$$

Example:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\odot
\begin{bmatrix}
10 & 20 \\
30 & 40
\end{bmatrix}
=
\begin{bmatrix}
10 & 40 \\
90 & 160
\end{bmatrix}
$$

In NumPy / PyTorch:

```python
A * B   # element-wise multiplication
A @ B   # matrix multiplication
```
