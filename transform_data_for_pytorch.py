import numpy as np
import torch
def transform_data(X_train, y_train, X_test, y_test, X_val, y_val):
  # convert the training and test data to floating point
  X_train = X_train.astype(np.float32)
  X_test = X_test.astype(np.float32)
  X_val = X_val.astype(np.float32)

  X_train = X_train / 255.0
  X_test = X_test / 255.0
  X_val = X_val / 255.0

  # Reshape the training data such that we have one image per row
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))

  # pre-processing: subtract mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_test -= mean_image
  X_val -= mean_image

  # convert everything to tensors
  X_train, y_train, X_test, y_test, X_val, y_val = map(
      torch.tensor, (X_train, y_train, X_test, y_test, X_val, y_val)
  )

  X_train = X_train.float()
  X_test = X_test.float()
  X_val = X_val.float()

  return X_train, y_train, X_test, y_test, X_val, y_val
