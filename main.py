import csv
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize, AffineTransform, warp
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB, plot_matches
import matplotlib.pyplot as plt

MAX_FEATURES = 400

np.random.seed(seed=1)


def alignImages(im1, im2, output_file):
    # Convert images to grayscale
    im1Gray = rgb2gray(im1)
    im2Gray = rgb2gray(im2)

    # Detect ORB features and compute descriptors.
    extractor = ORB(n_keypoints=MAX_FEATURES, fast_threshold=0.05)
    extractor.detect_and_extract(im1Gray)
    keypoints1 = extractor.keypoints
    descriptors1 = extractor.descriptors

    extractor.detect_and_extract(im2Gray)
    keypoints2 = extractor.keypoints
    descriptors2 = extractor.descriptors

    # Match features.
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Draw matchs
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, im1, im2, keypoints1, keypoints2, matches)
    plt.savefig(output_file)

    src = []
    dst = []
    for i in range(len(matches)):
        src.append(keypoints1[matches[i][0]])
        dst.append(keypoints2[matches[i][1]])
    src = np.array(src)
    dst = np.array(dst)

    model, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=500)

    warped = warp(im1, model, output_shape=im2.shape)

    return (warped, model.params)


if __name__ == '__main__':

    file_pairs = [
        ('./data/A1-2_HE_line.jpg', './data/A1-2.csv', './data/upscale/A1-2_HE_line.png'),
        ('./data/A9-3_HE_line.jpg', './data/A9-3.csv', './data/upscale/A9-3_HE_line.png'),
        ('./data/B3-2_HE_line.jpg', './data/B3-2.csv', './data/upscale/B3-2_HE_line.png'),
        ('./data/B9-3_HE_line.jpg', './data/B9-3.csv', './data/upscale/B9-3_HE_line.png')
    ]

    for i, (file1, file2, file3) in enumerate(file_pairs):  # file1 is the reference from pathology slice
        try:
            img = imread(file1)
            img = np.rot90(img)
            with open(file2, newline='') as csvfile:
                reader = csv.reader(csvfile)
                width = 0
                data = np.array(list(reader), dtype=np.float)

            data = resize(data, img.shape[0:2], anti_aliasing=True)
            data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
            data = gray2rgb(data)
            data = img_as_ubyte(data)
            imsave(file3, data)

            imReg, h = alignImages(data, img, 'matches{}.png'.format(i))
            imsave('warped{}.png'.format(i), imReg)
        except Exception as e:
            raise e
