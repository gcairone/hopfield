import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



class ImageMatrix:
    def __init__(self, image_filename, n):
        current_path = os.path.dirname(os.path.abspath(__file__)) + "\\img\\"
        image_path = os.path.join(current_path, image_filename)

        # Load image
        image = Image.open(image_path).convert('L')

        image = image.resize((n, n))
        self.matrix = np.array(image)
        treshold = 190
        self.matrix = np.vectorize(lambda x: -1 if x > treshold else 1)(self.matrix)

    def print_matrice(self):
        plt.imshow(self.matrix, cmap='binary', interpolation='nearest')
        plt.title("Matrice dell'immagine")
        plt.show()
"""
if __name__=="__main__":
    filename = ""  
    lato_matrice = 40  
    image_matrix = ImageMatrix(filename, lato_matrice)
    image_matrix.print_matrix()
"""