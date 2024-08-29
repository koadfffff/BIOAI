import numpy as np
import cv2
import matplotlib.pyplot as plt


class Population:
    def __init__(self, population_size, genotype_length):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))

    def samplingMask(self, reference_image_array):
        gray_img = cv2.cvtColor(reference_image_array, cv2.COLOR_BGR2GRAY)
        gradx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=1)
        grady = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=1)
        gradient_magnitude = cv2.magnitude(gradx, grady)

        # Normalize the gradient magnitude to be between 0 and 1
        sampling_mask = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # Increase the contrast of the sampling mask to give more weight to high gradients
        sampling_mask = sampling_mask ** 2

        return sampling_mask

    def compute_min_rgb_vals(self, reference_image_array):
        r_min, g_min, b_min = 255, 255, 255
        r_max, g_max, b_max = 0, 0, 0
        for i in range(len(reference_image_array)):
            for j in range(len(reference_image_array[0])):
                pixel = reference_image_array[i][j]
                red = pixel[0]
                green = pixel[1]
                blue = pixel[2]
                r_min, r_max = min(r_min, red), max(r_max, red)
                g_min, g_max = min(g_min, green), max(g_max, green)
                b_min, b_max = min(b_min, blue), max(b_max, blue)
        return r_min, r_max, g_min, g_max, b_min, b_max

    def initialize(self, genebound, reference_image_array):
        r_min, r_max, g_min, g_max, b_min, b_max = self.compute_min_rgb_vals(reference_image_array)

        # Calculate the sampling mask based on the gradient magnitudes
        sampling_mask = self.samplingMask(reference_image_array)

        # Flatten the sampling mask and normalize it to sum to 1
        flat_sampling_mask = sampling_mask.flatten()
        flat_sampling_mask /= flat_sampling_mask.sum()

        # Get the image dimensions
        height, width = reference_image_array.shape[:2]

        # Convert the flat mask to cumulative distribution for weighted sampling
        cumulative_distribution = np.cumsum(flat_sampling_mask)

        genelength = self.genes.shape[1]

        for i in range(self.genes.shape[0]):
            info = np.zeros(genelength)
            for j in range(0, genelength, 5):
                # Sample a point based on the cumulative distribution
                pixel_index = np.searchsorted(cumulative_distribution, np.random.rand())
                y, x = divmod(pixel_index, width)

                pixel = reference_image_array[y, x]
                info[j] = x
                info[j + 1] = y

                if np.random.random() >= 0.5:  # Adjust the probability of sampling a pixel from image
                    info[j] = np.random.randint(low=genebound[j][0], high=genebound[j][1])
                    info[j + 1] = np.random.randint(low=genebound[j+1][0], high=genebound[j+1][1])
                    info[j + 2] = np.random.randint(low=r_min, high=r_max)
                    info[j + 3] = np.random.randint(low=g_min, high=g_max)
                    info[j + 4] = np.random.randint(low=b_min, high=b_max)
                else:
                    info[j + 2] = pixel[0]
                    info[j + 3] = pixel[1]
                    info[j + 4] = pixel[2]
            self.genes[i, :] = info


    def mix(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]


