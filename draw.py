import numpy as np
from PIL import Image
from imgcompare import image_diff
from multiprocess import Pool, cpu_count
from scipy.spatial import KDTree


imagepoint = []


def voronoiData(genotype, width, height, scale=1):
    sc_width = int(width * scale)
    sc_height = int(height * scale)
    num_points = int(len(genotype) / 5)

    coords = [(genotype[i * 5] * scale, genotype[i * 5 + 1] * scale) for i in range(num_points)]
    colors = [(genotype[i * 5 + 2], genotype[i * 5 + 3], genotype[i * 5 + 4]) for i in range(num_points)]

    voronoidata = KDTree(coords)

    if scale == 1:
        image_points = imagepoint
    else:
        image_points = np.array([(x, y) for x in range(sc_width) for y in range(sc_height)])

    image_points_regions = (voronoidata.query(image_points))[1]

    data = np.zeros((sc_height, sc_width, 3), dtype='uint8')
    i = 0
    for x in range(sc_width):
        for y in range(sc_height):
            for j in range(3):
                data[y, x, j] = colors[image_points_regions[i]][j]
            i += 1

    return data


def voronoiImage(genotype, img_width, img_height, scale=1) -> Image:
    data = voronoiData(genotype, img_width, img_height, scale)
    return Image.fromarray(data, 'RGB')


def imagediff(genotype, reference_image: Image):
    actual = voronoiData(genotype, reference_image.width, reference_image.height)
    return image_diff(Image.fromarray(actual, 'RGB'), reference_image)


def standard(inputs):
    return imagediff(inputs[0], inputs[1])


def fitnessCalc(genes, reference_image: Image):
    if len(imagepoint) == 0:
        imagepoint.extend([(x, y) for x in range(reference_image.width) for y in range(reference_image.height)])

    with Pool(cpu_count()-1) as p:
        fitness_values = list(p.map(standard, zip(genes, [reference_image] * genes.shape[0])))
    return np.array(fitness_values)
