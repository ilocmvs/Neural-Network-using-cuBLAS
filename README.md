A simple learning model based on linear regression (realized using basic CUDA and cuBLAS functions):
  // Forward: z = XW + b, p = sigmoid(z)
  // Backward: e = (p - y) * (y > 0.5? 9:1) the weights here is for imbalanced data
  //           dW = (1/N) X^T e
  //           db = (1/N) sum(e)
  // Update:   W -= lr*dW; b -= lr*db

Training and test data is sourced from the famous https://www.cs.toronto.edu/~kriz/cifar.html
