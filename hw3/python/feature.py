import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
    
class SIFTExtractor:
    def __init__(self, vocabulary_size) -> None:
        self.sift = cv2.SIFT_create()
        self.kmeans = KMeans(n_clusters=vocabulary_size, n_init='auto')
        self.vocabulary = None
        

    # get sift features of each image
    def get_sift(self, dataset):
        all_features = []
        for image in tqdm(dataset):
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            if descriptors is not None:
                all_features.extend(descriptors)
        return all_features

    # build vocabulary -> find center of all descriptors by kmeans
    def build_vocabulary(self, dataset):
        all_features = self.get_sift(dataset)

        # clustering by kmeans
        self.kmeans.fit(all_features)
        self.vocabulary = self.kmeans.cluster_centers_

        # return self.vocabulary

    # turn an image into a histogram representation
    def image_to_histogram(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        # predict the words (clusters)
        nearest_clusters = self.kmeans.predict(descriptors.astype(np.double))

        # histogram
        histogram = np.zeros(len(self.vocabulary))
        for cluster in nearest_clusters:
            histogram[cluster] += 1

        return histogram
    
    # transform a whole dataset to histogram representation
    def dataset_to_histograms(self, dataset):
        histograms = []
        for image in tqdm(dataset):
            histograms.append(self.image_to_histogram(image))
        return histograms

    
# 22.68%
class TinyImageExtractor:
    def __init__(self, target_size=(16, 16)) -> None:
        self.target_size = target_size

    # turn a whole dataset to tiny image representations
    def dataset_to_tiny_images(self, dataset, normalized = True):
        tiny_images = []
        for i, (image) in enumerate(tqdm(dataset)):
            img_resized = cv2.resize(image, self.target_size)
            if normalized:
                img_resized = img_resized.flatten() - np.mean(img_resized.flatten())
                img_resized /= np.std(img_resized)
            else: 
                img_resized = img_resized.flatten()
            tiny_images.append(img_resized)
            
        
        return tiny_images
        
