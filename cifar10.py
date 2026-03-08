import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py", val_size=5000):
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])

    X_all = np.vstack(train_data)                          # (50000, 3072)
    y_all = np.array(train_labels)                         # (50000,)

    test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test  = test_dict[b'data']
    y_test  = np.array(test_dict[b'labels'])

    # ── Carve out validation set ─────────────────────────────────────────────
    X_val,   y_val   = X_all[:val_size],  y_all[:val_size]   # (5000,  3072)
    X_train, y_train = X_all[val_size:],  y_all[val_size:]   # (45000, 3072)

    # ── Reshape to (N, 32, 32, 3) ────────────────────────────────────────────
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_val   = X_val.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test  = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)