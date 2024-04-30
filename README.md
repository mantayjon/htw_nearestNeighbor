# htw_nearestNeighbor Class for CIFAR-10 Image Processing

This Python script utilizes the NearestNeighbor class to process images from the CIFAR-10 dataset, visualize them, and perform nearest neighbor calculations using L1 distance. The class provides methods to unpickle data, retrieve image vectors, reshape them for visualization, and identify the nearest neighbors based on the calculated distances.

## Features

- **Unpickle Data**: Load CIFAR-10 dataset batches.
- **Image Vector Retrieval**: Extract image vectors from data batches.
- **Image Reshaping and Visualization**: Reshape image data into 32x32x3 arrays and visualize them.
- **Nearest Neighbor Calculation**: Calculate the nearest neighbors using L1 distance.
- **Label Retrieval**: Get labels for images based on batch indices.
- **Distance Calculation**: Compute L1 distance between two image vectors.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Setup and Execution

1. **Ensure Python 3.x is installed**: Python 3.x is required to run this script. You can download it from [python.org](https://www.python.org/downloads/).

2. **Install Required Libraries**:
   ```bash
   pip install numpy matplotlib
