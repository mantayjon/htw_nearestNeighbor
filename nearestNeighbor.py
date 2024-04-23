import numpy as np
import matplotlib.pyplot as plt


class NearestNeighbor:
    def __int__(self):
        pass

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_image_vector(xtr, batch_idx, image_idx):
        try:
            batch = xtr[batch_idx]
            images = batch[b'data']
            image_vector = images[image_idx]
            return image_vector
        except IndexError:
            print("batch_index or image_index not correct")
            return None

    def reshape_image(image_data):
        # Reshape the image data into a 32x32x3 array
        image_reshaped = np.array(image_data).reshape(3, 32, 32).transpose(1, 2, 0)  # transpose changes RGB to GBR

        return image_reshaped

    def visualize_image(image_reshaped):

        # Visualize the image
        plt.imshow(image_reshaped)
        plt.axis('off')  # Turn off axis
        plt.show()

    Xtr = []

    for x in range(1, 6):  # Loop from 1 to 5
        file_path = "cifar-10-batches-py/data_batch_" + str(x)
        data = unpickle(file_path)
        Xtr.append(data)

    #  print(Xtr)
    #  print(len(Xtr))

    y_path = "cifar-10-batches-py/test_batch"
    y = unpickle(y_path)

    image_vector = get_image_vector(Xtr, 0, 0)

    image = reshape_image(image_vector)

    print(image)


    #  visualize_image(image)
