
import os
import numpy as np
from scipy.misc import imread, imresize


class CUB200(object):
    """
    Util for loading the CUB-200 dataset.
    """
    def __init__(self, path, image_size=(224, 224)):
        self._path = path
        self._size = image_size

    def _classes(self):
        return os.listdir(self._path)

    def _load_image(self, category, im_name):
        return imresize(imread(os.path.join(self._path, category, im_name), mode="RGB"), self._size)

    def load_dataset(self, num_per_class=None):
        classes = self._classes()
        all_images = []
        all_labels = []
        for c in classes:
            class_images = os.listdir(os.path.join(self._path, c))

            if num_per_class is not None:
                class_images = np.random.choice(class_images, num_per_class)

            for image_name in class_images:
                all_images.append(self._load_image(c, image_name))
                all_labels.append(c)
        return np.array(all_images).astype(float), np.array(all_labels)


if __name__ == "__main__":
    """
    This is just a test to make sure all is good with the world. 
    """
    import matplotlib.pyplot as plt

    DATA_DIR = os.path.expanduser(os.path.join("~", "data", "blog"))
    CUB_DIR = os.path.join(DATA_DIR, "CUB_200_2011", "images")

    X, lbl = CUB200(CUB_DIR).load_dataset(num_per_class=2)
    n = X.shape[0]
    rnd_birds = np.vstack([np.hstack([X[np.random.choice(n)] for i in range(10)])
                           for j in range(10)])
    plt.figure(figsize=(10, 10))
    plt.imshow(rnd_birds / 255)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title("100 random birds...", fontsize=30)

    plt.show()
