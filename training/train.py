import os
import time
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

# Not used for train_as_lib!
MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model. You can either give path to dataset or provide path '
                    'to pre-generated embeddings, labels and class_to_idx. You can pre-generate this with '
                    'util/generate_embeddings.py script.')
    parser.add_argument('-d', '--dataset-path', help='Path to folder with images.')
    parser.add_argument('-e', '--embeddings-path', help='Path to file with embeddings.')
    parser.add_argument('-l', '--labels-path', help='Path to file with labels.')
    parser.add_argument('-c', '--class-to-idx-path', help='Path to pickled class_to_idx dict.')
    parser.add_argument('--grid-search', action='store_true',
                        help='If this option is enabled, grid search will be performed to estimate C parameter of '
                             'Logistic Regression classifier. In order to use this option you have to have at least '
                             '3 examples of every class in your dataset. It is recommended to enable this option.')
    parser.add_argument('--svm', action='store_true', help='Use Support Vector Machine classifier. Slow for large'
                        ' number of embeddings.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(args, features_extractor):
    if args.embeddings_path:
        return np.loadtxt(args.embeddings_path), \
               np.loadtxt(args.labels_path, dtype='str').tolist(), \
               joblib.load(args.class_to_idx_path)

    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if args.grid_search and not args.svm:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        if args.svm and not args.grid_search:
            clf = svm.SVC(kernel = 'linear', probability = True)
        else:
            clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(args, features_extractor)
    clf = train(args, embeddings, labels)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('model', 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)

def train_as_lib(dataset_path, models_path):
    features_extractor = FaceFeaturesExtractor()
    # Time of folder-freeze for training: if something is deleted in the dataset, training could fail
    # depending on progress, and will not include change in case it is successful
    freeze_time = time.time()
    dataset = datasets.ImageFolder(dataset_path)
    # Embeddings Generation
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    # Classifier Training
    clf = svm.SVC(kernel = 'linear', probability = True)
    clf.fit(embeddings, labels)
    
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))
    
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    model_path = os.path.join(models_path, 'face_recogniser_'+str(freeze_time)+'.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)
    
def load_and_train_as_lib(dataset_path, models_path):
    features_extractor = FaceFeaturesExtractor()
    # Time of folder-freeze for training: if something is deleted in the dataset, training could fail
    # depending on progress, and will not include change in case it is successful
    freeze_time = time.time()
    # Generate Dataset
    dataset = datasets.ImageFolder(dataset_path)
    class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    # Load Embeddings and Labels
    embeddings = np.loadtxt(dataset_path+os.path.sep+"embeddings.txt")
    labels = np.loadtxt(dataset_path+os.path.sep+"labels.txt", dtype='str').tolist()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Classifier Training
    clf = svm.SVC(kernel = 'linear', probability = True)
    clf.fit(embeddings, labels)
    
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))
    # Save model
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    model_path = os.path.join(models_path, 'face_recogniser_'+str(freeze_time)+'.pkl')
    print ("Model saved in " + model_path)
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)

def normalise_string(string):
    return string.lower().replace(' ', '_')
def normalise_dict_keys(dictionary):
    new_dict = dict()
    for key in dictionary.keys():
        new_dict[normalise_string(key)] = dictionary[key]
    return new_dict

def add_embeddings_for_img(dataset_path, img_path, img_id):
    # Create Embeddings and Labels in case they are not present using
    embeddings_path = os.path.join(dataset_path, 'embeddings.txt')
    labels_path = os.path.join(dataset_path, 'labels.txt')
    if((not os.path.isfile(embeddings_path)) or (not os.path.isfile(labels_path))):
        dataset_to_embeddings_lib(dataset_path)
        return
    
    # Load Embeddings and Labels
    embeddings = np.loadtxt(embeddings_path).tolist()
    labels = np.loadtxt(labels_path, dtype='str').tolist()
    
    # Generate new embedding
    features_extractor = FaceFeaturesExtractor()
    print(img_path)
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
    if embedding is None:
        print("Could not find face on {}".format(img_path))
        return
    if embedding.shape[0] > 1:
        print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
        embedding = embedding[0, :]
    embeddings.append(embedding.flatten())
    # Just append, as the labels are already a list of strings and the ID is a string
    labels.append(img_id)
    # Store embeddings and labels
    np.savetxt(dataset_path+os.path.sep+"embeddings.txt", embeddings)
    np.savetxt(dataset_path+os.path.sep+"labels.txt", np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")
    
def dataset_to_embeddings_lib(dataset_path):
    features_extractor = FaceFeaturesExtractor()
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    dataset = datasets.ImageFolder(dataset_path)
    embeddings = []
    labels = []
    size = 0
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        size = size + 1
        embeddings.append(embedding.flatten())
        labels.append(label)
    # If there are less than two embeddings, do not save
    if(size < 2):
        return
    # Store embeddings and labels
    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))
    np.savetxt(dataset_path+os.path.sep+"embeddings.txt", embeddings)
    np.savetxt(dataset_path+os.path.sep+"labels.txt", np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")
    #joblib.dump(dataset.class_to_idx,dataset_path+os.path.sep+"class_to_idx.pkl")
    
def remove_embeddings_for_id(dataset_path,img_id):
    # Load Embeddings and Labels
    embeddings_path = os.path.join(dataset_path, 'embeddings.txt')
    labels_path = os.path.join(dataset_path, 'labels.txt')
    try:
        embeddings = np.loadtxt(embeddings_path).tolist()
        labels = np.loadtxt(labels_path, dtype='str').tolist()
    except:
        return
    for _ in range(labels.count(normalise_string(img_id))):
        location = labels.index(normalise_string(img_id))
        __ = labels.pop(location)
        __ = embeddings.pop(location)
    if(np.size(labels) < 2):
        os.remove(embeddings_path)
        os.remove(labels_path)
        return
    # Generate Dataset
    dataset = datasets.ImageFolder(dataset_path)
    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    # Save data
    np.savetxt(dataset_path+os.path.sep+"embeddings.txt", embeddings)
    np.savetxt(dataset_path+os.path.sep+"labels.txt", np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")
    #joblib.dump(dataset.class_to_idx,dataset_path+os.path.sep+"class_to_idx.pkl")

if __name__ == '__main__':
    main()
