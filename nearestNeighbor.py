import numpy as np
import matplotlib.pyplot as plt
import pickle



class NearestNeighbor:
    def __init__(self):
        pass

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_image_vector(self, batch, image_idx):
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

    def visualize_image(self, image_reshaped, label):
        # Visualize the image
        plt.imshow(image_reshaped)
        plt.axis('off')
        plt.title(str(label))


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

    def get_label_Xtr(self, batch_index, indice):
        batch = Xtr[batch_index]
        label = batch[b'labels'][indice]
        return label

    def get_label_y(self, indice, b):
        label = b[b'labels'][indice]
        return label




if __name__ == '__main__':

    nn = NearestNeighbor()
    Xtr = []
    y_path = "cifar-10-batches-py/test_batch"
    y = nn.unpickle(y_path)
    plots = []
    acuracy = []
    num_columns = 5
    figsize = (12, 3)
    k = 1
    correct_predictions = 0
    n = 25


    for x in range(1, 6):
        file_path = "cifar-10-batches-py/data_batch_" + str(x)
        data = nn.unpickle(file_path)
        Xtr.append(data)

    all_vectors = nn.get_all_image_vectors(Xtr)


    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(0, n):
        single_image = nn.get_image_vector(y, i)
        label_actual = nn.get_label_y(i, y)

        nearest_indices = nn.k_nearest_neighbor(all_vectors, single_image, k)

        single_image = nn.reshape_image(single_image)

        labels = []
        for image in nearest_indices:
            batch_index = nn.get_batch_indice(image, Xtr)
            batch = batch_index[0]
            indice = batch_index[1]
            # image_vector = nn.get_image_vector(Xtr[batch], indice)
            # image_vector = nn.reshape_image(image_vector)
            label = nn.get_label_Xtr(batch, indice)
            # nn.visualize_image(image_vector, label)
            labels.append(label)

        counts = np.bincount(labels)

        label_i = np.argmax(counts)
        counts = np.bincount(labels)

        title = 'predicted: ' + class_names[label_i] + '\n' + 'actual: ' + class_names[label_actual]

        plots.append((single_image, title))
        acuracy.append((label_actual, label_i))

    num_images = len(plots)
    num_rows = int(np.ceil(num_images / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)

    axes = axes.flatten()

    # Iterate over each subplot and display image with text
    for i, (image, text) in enumerate(plots):
        ax = axes[i]
        ax.imshow(image)  # Plot the image on the subplot
        ax.set_title(text)  # Set the title of the subplot
        ax.axis('off')  # Hide the axes of the subplot

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')


    plt.tight_layout()

    # plt.savefig(f'images/k_{k}.png')
    # plt.show()

    # Iterate over the accuracy array
    for actual, predicted in acuracy:
        # If the actual and predicted labels match, increment the counter
        if actual == predicted:
            correct_predictions += 1

    # Calculate the accuracy
    accuracy = correct_predictions / n

    print(f"The accuracy is for k={k} with n={n} is: {accuracy}")

    '''
    The accuracy is for k=1 with n=10 is: 0.1
    The accuracy is for k=3 with n=10 is: 0.4
    The accuracy is for k=5 with n=10 is: 0.4
    The accuracy is for k=7 with n=10 is: 0.4
    
    The accuracy is for k=1 with n=25 is: 0.2 
    The accuracy is for k=3 with n=25 is: 0.32
    The accuracy is for k=5 with n=25 is: 0.36
    The accuracy is for k=7 with n=25 is: 0.4
    '''