from skimage import io, exposure, morphology, filters, measure
from sklearn import metrics, ensemble, model_selection
import matplotlib.pyplot as plt
import numpy as np

NUM_OF_TRAIN_SAMPLES = 1000000
SHOW_EVERY_PLOT = False

INPUT_FILES = ["Data/Images/" + str(x // 10) + str(x % 10) + "_h.jpg" for x in range(1, 16)]
OUTPUT_FILES = ["Data/Output/" + str(x // 10) + str(x % 10) + "_h.tif" for x in range(1, 16)]


def main():

    # --------------------
    # IMAGE PROCESSING
    # --------------------
    print('*' * 3, "Image Processing", '*' * 3)

    # Open image
    img = io.imread(INPUT_FILES[0])
    output = io.imread(OUTPUT_FILES[0]) // 255

    # Preprocessing
    preprocessed_img, mask = preprocess(img)

    # Blood vessels extraction from preprocessed image
    feature_extracted_img = feature_extraction(preprocessed_img)

    # Improve result and create binary output image
    binary_output = enhanced_binary_result(feature_extracted_img, mask)

    # Print confusion matrix and accuracy, sensitivity and specificity
    show_confusion_matrix(output.ravel(), binary_output.ravel())

    # --------------------
    # MACHINE LEARNING
    # --------------------
    print('*' * 3, "Machine learning", '*' * 3)

    # Images to learn from
    learning_image_set = [10, 11, 12, 13]
    samples_per_image = NUM_OF_TRAIN_SAMPLES // len(learning_image_set)

    # Training dataset
    train_set = []
    train_classes = []

    # Gather training data
    counter = 0
    for i in learning_image_set:

        counter += 1
        print("Gathering train data... (", counter, '/', len(learning_image_set), ')')

        training_img = io.imread(INPUT_FILES[i])
        training_output = io.imread(OUTPUT_FILES[i]) // 255
        preprocessed_img, _ = preprocess(training_img)

        # Random pixels in given image
        train_samples_x = np.random.randint(2, preprocessed_img.shape[0] - 4, samples_per_image)
        train_samples_y = np.random.randint(2, preprocessed_img.shape[1] - 4, samples_per_image)

        # Create train_set
        for j in range(samples_per_image):
            train_set.append(get_data(train_samples_x[j], train_samples_y[j], preprocessed_img))
            train_classes.append(training_output[train_samples_x[j], train_samples_y[j]])

    # Create classifier
    clf_forest = ensemble.RandomForestClassifier(n_jobs=-1)

    # Training
    print("Training...")
    clf_forest.fit(train_set, train_classes)

    # K-fold cross validation
    # print("k-fold cross validation...")
    # score = model_selection.cross_val_score(clf_forest, train_set, train_classes, cv=10)
    # print("k-fold cross validation: accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

    # Empty result image
    ml_output = np.zeros(preprocessed_img.shape)
    test_set = []

    # Line by line fitting
    print("Classifying...")
    for x in range(2, preprocessed_img.shape[0] - 2):
        test_set.clear()
        for y in range(2, preprocessed_img.shape[1] - 2):
            test_set.append(get_data(x, y, preprocessed_img))
        predicted_forest = clf_forest.predict(test_set)
        ml_output[x, 2:-2] = predicted_forest.ravel()

    # Show quality of result
    show_confusion_matrix(output.ravel(), ml_output.ravel())

    # Show original image
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")

    # Show image processing result
    plt.subplot(3, 2, 3)
    plt.imshow(binary_output, cmap='gray')
    plt.axis('off')
    plt.title("Image processing")

    # Show machine learning result
    plt.subplot(3, 2, 4)
    plt.imshow(ml_output, cmap='gray')
    plt.axis('off')
    plt.title("Machine learning")

    # Plot expert mask
    plt.subplot(3, 1, 3)
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.title("Expert mask")
    plt.show()


# Function extracts data for given pixel from neighbourhood
def get_data(x, y, img):

    # Create data and 5x5 image slice
    data = []
    new_image = img[x - 2:x + 2, y - 2:y + 2]

    # Append location of pixel
    data.append(x)
    data.append(y)

    # Append pixel values to data
    for pixel_val in new_image.ravel():
        data.append(pixel_val)

    # Append central moments to data
    central_moments = measure.moments_central(new_image)
    for central_moment in central_moments.ravel():
        data.append(central_moment)

    # Append hu moments to data
    hu_moments = measure.moments_hu(measure.moments_normalized(central_moments))
    for hu_moment in hu_moments:
        data.append(hu_moment)

    # Append variation of pixel values to data
    variation = np.var(new_image)
    data.append(variation)

    return data


def show_confusion_matrix(real_matrix, predicted_matrix):
    print("Checking quality of result...")

    # Get confusion matrix and calculate parameters
    tn, fp, fn, tp = metrics.confusion_matrix(real_matrix, predicted_matrix).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    g_mean1 = (precision * sensitivity) ** (1 / 2)
    g_mean2 = (sensitivity * specificity) ** (1 / 2)

    # Print result
    print('-' * 25)
    print("Confusion Matrix:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1-score:", f1_score)
    print("G-mean = sqrt(precision * sensitivity):", g_mean1)
    print("G-mean = sqrt(sensitivity * specificity):", g_mean2)
    print('-' * 25)


def preprocess(img):
    print("Preprocessing...")

    # Consider Green channel
    img_g = img[:, :, 1]

    # Set black pixels to zero
    mask = img[:, :, 0] > 10
    img_g = img_g * mask

    # Create binary mask for eye
    mask = morphology.binary_erosion(mask * 1, selem=np.ones((15, 15)))
    mask[:20, :] = 0
    mask[-20:, :] = 0

    # Contrast stretching
    p1, p99 = np.percentile(img_g[np.nonzero(img_g)], (1, 99))
    img_rescale = exposure.rescale_intensity(img_g, in_range=(p1, p99))

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img_rescale, clip_limit=0.07)

    if SHOW_EVERY_PLOT:
        plt.figure()
        plt.title("Preprocessing")
        plt.subplot(2, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title("Mask")
        plt.subplot(2, 2, 2)
        plt.imshow(img_rescale, cmap='gray')
        plt.axis('off')
        plt.title("Contrast stretching")
        plt.subplot(2, 1, 2)
        plt.imshow(img_adapteq, cmap='gray')
        plt.axis('off')
        plt.title("Adaptive Equalization 0.07")
        plt.show()

    return img_adapteq, mask


def feature_extraction(img):
    print("Extracting...")

    # Frangi filter
    frangi = filters.frangi(img)

    # Closing
    closing = morphology.closing(frangi)

    # Equalize Histogram
    eq_hist = exposure.equalize_hist(closing)

    # Erosion
    erosion = morphology.erosion(morphology.erosion(morphology.erosion(eq_hist)))

    if SHOW_EVERY_PLOT:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(frangi, cmap='gray')
        plt.axis('off')
        plt.title("Frangi filter")
        plt.subplot(2, 2, 2)
        plt.imshow(closing, cmap='gray')
        plt.axis('off')
        plt.title("Closing")
        plt.subplot(2, 2, 3)
        plt.imshow(eq_hist, cmap='gray')
        plt.axis('off')
        plt.title("Equalized histogram")
        plt.subplot(2, 2, 4)
        plt.imshow(erosion, cmap='gray')
        plt.axis('off')
        plt.title("Erosion")
        plt.show()

    return erosion


def enhanced_binary_result(img, mask):
    print("Enhancing...")

    # Threshold
    threshold = (img > 0.85) * 1

    # Apply binary mask from preprocessing
    masked = threshold * mask

    # Dilatation
    binary_output = morphology.binary_dilation(masked)

    if SHOW_EVERY_PLOT:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(threshold, cmap='gray')
        plt.axis('off')
        plt.title("Binary threshold")
        plt.subplot(2, 2, 2)
        plt.imshow(masked, cmap='gray')
        plt.axis('off')
        plt.title("Masked output")
        plt.subplot(2, 1, 2)
        plt.imshow(binary_output, cmap='gray')
        plt.axis('off')
        plt.title("Enhanced result")
        plt.show()

    return binary_output


if __name__ == '__main__':
    main()
