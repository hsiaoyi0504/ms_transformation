import cv2
import csv
import numpy as np

MAX_FEATURES = 700
GOOD_MATCH_PERCENT = 1


def alignImages(im1, im2, i):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches{}.jpg".format(i), imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':

    file_pairs = [
        ('./data/A1-2_HE_line.jpg', './data/A1-2.csv', './data/upscale/A1-2_HE_line.png'),
        ('./data/A9-3_HE_line.jpg', './data/A9-3.csv', './data/upscale/A9-3_HE_line.png'),
        ('./data/B3-2_HE_line.jpg', './data/B3-2.csv', './data/upscale/B3-2_HE_line.png'),
        ('./data/B9-3_HE_line.jpg', './data/B9-3.csv', './data/upscale/B9-3_HE_line.png')
    ]

    for i, (file1, file2, file3) in enumerate(file_pairs):  # file1 is the reference from pathology slice
        try:
            img = cv2.imread(file1)

            with open(file2, newline='') as csvfile:
                reader = csv.reader(csvfile)
                width = 0
                data = np.array(list(reader), dtype=np.float32)

            data = data / np.amax(data) * 255  # scale to float (0~255)
            data = cv2.merge((data, data, data))
            data = cv2.resize(data, img.shape[1::-1])
            data = data.astype(np.uint8)
            cv2.imwrite(file3, data)
            # print(data.shape)
            # cv2.imshow('image', data)
            # cv2.waitKey(0)

            imReg, h = alignImages(data, img, i)
        except Exception as e:
            # raise e
            pass
