import numpy as np
def transform_with_bias(X_train,X_test,X_val):
  print(X_train)
  print(X_train.shape)
  # Flatten X_train and X_test to 2D arrays (num_samples, num_features)
  X_train = X_train.reshape(X_train.shape[0], -1)
  X_test = X_test.reshape(X_test.shape[0], -1)
  X_val = X_val.reshape(X_val.shape[0], -1)

  X_train = X_train / 255.0
  X_test = X_test / 255.0
  X_val = X_val / 255.0

  print(X_train)
  print(X_train.shape)

  # Now apply the bias trick
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

  print(X_train)
  print(X_train.shape)
  print(X_test.shape)
  
  return X_train,X_test,X_val