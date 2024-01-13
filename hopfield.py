import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame 

class Grid:
    def __init__(self, M) -> None:
        if M.shape[0] != M.shape[1]:
            print("Wrong dimension")
            return
        if not np.all((M == 1) | (M == -1)):
            print("Only zeros or ones allowed")
            return
        self.M = M
    def show_grid(self):
        fig, ax = plt.subplots()
        ax.imshow(self.M, cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    


class HopfieldNetwork:
    def __init__(self, W) -> None:
        self.W = W
    def compute(self, x):
        temp = self.W @ x
        return np.vectorize(lambda x: 1 if x >= 0 else -1)(temp)
    def iter(self, x, stop_if_stable=True, max_iter=20, verbose=True, energy=True):
        tprec = x
        i = 0
        while(True):
            t = self.compute(tprec)
            if verbose: 
                print(i, tprec)
                if energy:
                    print(self.energy(tprec))
            i = i+1
            if(stop_if_stable and np.array_equal(t, tprec)):
                if verbose:
                    print("Stable")
                    print(t)
                return t
            if(i==max_iter):
                if verbose:
                    print("Stop")
                return t
            tprec = t
    def energy(self, x):
        return -0.5 * (x @ self.W @ x.T)
    def explore(self, verbose):
        stable_states = set()
        for i in range(2**self.W.shape[0]):
            x = int_to_array(i, self.W.shape[0]) 
            stable_state = self.iter(x, verbose=verbose)
            if verbose:
                print("")
            stable_states.add(array_to_int(stable_state))
        return [int_to_array(num, self.W.shape[0]) for num in stable_states]
    def animate(self, x0):
        # x0 must be a matrix
        pygame.init()

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        n = x0.shape[0]

        PIXEL_DIM = 5
        N_ROWS = n
        N_COLUMNS = n

        matrice = x0

        window = pygame.display.set_mode((N_COLUMNS * PIXEL_DIM, N_ROWS * PIXEL_DIM))
        pygame.display.set_caption('Matrice Animata')


        def disegna_matrice():
            for i in range(N_ROWS):
                for j in range(N_COLUMNS):
                    colore = BLACK if matrice[i, j] == 1 else WHITE
                    pygame.draw.rect(window, colore, (j * PIXEL_DIM, i * PIXEL_DIM, PIXEL_DIM, PIXEL_DIM))

        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        matrice = self.compute(matrice.flatten()).reshape((n, n)) 

            window.fill(WHITE)
            disegna_matrice()
            pygame.display.flip()

            clock.tick(5)  
        pygame.quit()

        

# only one pattern
def create_matrix(m):
    return (np.outer(m, m) - np.eye(len(m)))

def create_matrix_patterns(l):
    return np.sum((np.outer(p, p) - np.eye(len(p))) for p in l)

def int_to_array(num, n):
    bin_repr = bin(num % (2**n))[2:].zfill(n)
    array_repr = [int(bit) for bit in bin_repr]
    for i in range(len(array_repr)):
        if array_repr[i] == 0:
            array_repr[i] = -1
    return np.array(array_repr)


def array_to_int(array_repr):
    array_repr[array_repr == -1] = 0
    bin_repr_str = ''.join(map(str, array_repr))
    return int(bin_repr_str, 2)



"""

pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, -1, -1, -1, 1, 1, 1, 1, 1], 
                    [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
                    [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
                    [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
                    [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
                    [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
                    [1, 1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

x0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
               [1, 1, 1, 1, 1, -1, 1, 1, -1, 1], 
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, -1, 1, 1],
               [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
               [1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
               [1, 1, -1, -1, -1, 1, 1, 1, 1, 1],
               [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1]])

"""
"""
pattern = immagine.ImageMatrix("Cattura.PNG", 40).matrix
x0 = np.ones(shape=(40, 40))


matrix = create_matrix(pattern.flatten())
h = HopfieldNetwork(matrix)
Grid(x0).show_grid()

result = h.iter(x0.flatten(), stop_if_stable=True, max_iter=20, verbose=True)

Grid(result.reshape(len(x0), len(x0))).show_grid()

"""






