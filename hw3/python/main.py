from dataset import load_data, makedirs
from feature import TinyImageExtractor, SIFTExtractor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def sift(data_train, label_train, data_test, vocabulary_size = 50):
    extractor = SIFTExtractor(vocabulary_size)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    
    print('building vocabulary...')
    extractor.build_vocabulary(data_train)

    # turn datasets into histograms
    print('transforming datasets...')
    histogram_train = extractor.dataset_to_histograms(data_train)
    histogram_test  = extractor.dataset_to_histograms(data_test)

    # train knn
    knn_classifier.fit(histogram_train, label_train)

    # predict
    predictions = knn_classifier.predict(histogram_test)

    return predictions 

def tiny_img(data_train, label_train, data_test, tiny_size = (16, 16)):
    extractor = TinyImageExtractor(tiny_size)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    
    # turn datasets into tiny images
    print('transforming datasets...')
    tiny_imgs_train = extractor.dataset_to_tiny_images(data_train)
    tiny_imgs_test  = extractor.dataset_to_tiny_images(data_test)
    
    # train knn
    knn_classifier.fit(tiny_imgs_train, label_train)

    # predict
    predictions = knn_classifier.predict(tiny_imgs_test)

    return predictions 


if __name__ == "__main__":
    data_train, label_train, data_test, label_test, labels = load_data()
    
    for size in [(8, 8), (16, 16), (24, 24), (32, 32), (64, 64)]:
        print(f'-------------------------------- {size} --------------------------------')
        predictions = tiny_img(data_train, label_train, data_test, size)
        print(classification_report(label_test, predictions, target_names=labels, digits=4))
    
    for size in [20, 50, 100, 200]:
        print(f'-------------------------------- {size} --------------------------------')
        
        predictions = sift(data_train, label_train, data_test, size)
        print(classification_report(label_test, predictions, target_names=labels, digits=4))
    
    
    