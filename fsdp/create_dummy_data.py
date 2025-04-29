import numpy as np
import os

# Create train data
os.makedirs('/data/train', exist_ok=True)
np.save('/data/train/features.npy', np.random.randn(10000, 1024).astype(np.float32))
np.save('/data/train/labels.npy', np.random.randint(0, 10, size=10000))

# Create validation data
os.makedirs('/data/val', exist_ok=True)
np.save('/data/val/features.npy', np.random.randn(2000, 1024).astype(np.float32))
np.save('/data/val/labels.npy', np.random.randint(0, 10, size=2000))

print("Dummy data created successfully!")
