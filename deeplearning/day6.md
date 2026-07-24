7/21/2026

Paused for a while from working and travelling. Back to deep learning.

Today i need to review and finish the logistic regression project and understand everyline of it.

The most important part is

1. forward propagation
2. backword propagation

we want to know change w a little bit, how much it changes on L

after caculation for one individual derative
dl_dw = x_i \*(a-y)
dl_db = a-y

dZ = A- Y
dw = 1/m*np.dot(X, (A-Y).T)
db = 1/m*np.sum(A-Y) 3. Why sigmoid activation 4. Why Loss function is like this
