import numpy as np
import matplotlib.pyplot as plt


class NearestNeighbor:
    def __init__(self):
        pass

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_image_vector(self,batch, image_idx):
        try:
            images = batch[b'data']
            image_vector = images[image_idx]
            return image_vector
        except IndexError:
            print("batch_index or image_index not correct")
            return None

    def get_all_image_vectors(self, Xtr):

        Xtr_vectors = []
        try:
            for x in range(0, len(Xtr)):
                batch = Xtr[x]
                images = batch[b'data']

                for image_vector in images:
                    Xtr_vectors.append(image_vector)
            return Xtr_vectors

        except IndexError:
            print("batch_index or image_index not correct")
            return None

    def reshape_image(self, image_data):
        # Reshape the image data into a 32x32x3 array
        image_reshaped = np.array(image_data).reshape(3, 32, 32).transpose(1, 2, 0)  # transpose changes RGB to GBR
        return image_reshaped

    def visualize_image(self, image_reshaped):
        # Visualize the image
        plt.imshow(image_reshaped)
        plt.axis('off')  # Turn off axis
        plt.show()

    def calc_l1_distance(self, image_vec1, image_vec2):
        distance = np.sum(np.abs(image_vec1 - image_vec2))
        return distance

    def k_nearest_neighbor(self, Xtr_vectors, test_image, k):
        distances = []
        for image_vector in Xtr_vectors:
            distance = self.calc_l1_distance(test_image, image_vector)
            distances.append(distance)

        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        return k_nearest_indices

    def get_batch_indice(self, indice, Xtr):
        batch = 0
        for x in range(0, len(Xtr)):
            batch_len = len(Xtr[x][b'data'])
            if batch_len < indice:
                indice = indice - batch_len
                batch = batch + 1

        return batch, indice

    def get_label(self, batch_index, indice):
        batch = Xtr[batch_index]
        label = batch[b'labels'][indice]
        return label



if __name__ == '__main__':

    nn = NearestNeighbor()

    Xtr = []

    for x in range(1, 6):
        file_path = "cifar-10-batches-py/data_batch_" + str(x)
        data = nn.unpickle(file_path)
        Xtr.append(data)

    y_path = "cifar-10-batches-py/test_batch"
    y = nn.unpickle(y_path)

    all_vectors = nn.get_all_image_vectors(Xtr)

    for i in range(0,10):
        single_image = nn.get_image_vector(y, i)

        nearest_indices = nn.k_nearest_neighbor(all_vectors, single_image, 7)

        single_image = nn.reshape_image(single_image)

        labels = []
        for image in nearest_indices:
            batch_index = nn.get_batch_indice(image, Xtr)
            batch = batch_index[0]
            indice = batch_index[1]
            # image_vector = nn.get_image_vector(Xtr[batch], indice)
            # image_vector = nn.reshape_image(image_vector)
            # nn.visualize_image(image_vector)
            labels.append(nn.get_label(batch, indice))

        counts = np.bincount(labels)
        # print(labels)
        print(np.argmax(counts))
        counts = np.bincount(labels)
        print(counts)
        nn.visualize_image(single_image)

