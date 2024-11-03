import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cityblock, euclidean, cosine
from scipy.special import softmax


def reduce_dimensionality(features, labels, n_components=None):
    """
    Reduces the dimensionality of feature vectors using Fisher Linear Discriminant Analysis.
    
    Args:
        features (np.ndarray): Array of original feature vectors (shape: [num_samples, num_features]).
        labels (np.ndarray): Array of labels corresponding to each feature vector.
        n_components (int): Number of dimensions for reduced feature vector space. Can't be larger than the number of irises

    Returns:
        np.ndarray, np.ndarray: The LDA model and a dictionary containing class centers
    """

    # Initialize the LDA model
    model = LinearDiscriminantAnalysis(n_components=n_components, solver='eigen', shrinkage='auto')
    model.fit(features, labels)

    if not n_components:
        # No dimension reduction if number of components is not specified
        reduced_features = features
        model = None
    else:
        # Else reduce the features
        reduced_features = model.transform(features)

    # Compute the class centers for the reduced features
    class_centers = {}
    for label in np.unique(labels):
        class_centers[label] = np.mean(reduced_features[labels == label], axis=0)

    return model, class_centers


def compute_nearest_center(reduced_feature, class_centers, distance_type='L2'):
    """
    Classifies a feature vector by finding the nearest class center and returns the probability.

    Args:
        feature (np.ndarray): Reduced feature vector of the unknown sample.
        class_centers (dict): Dictionary of class centers.
        distance_type (str): Type of distance metric ('L1', 'L2', or 'cosine').

    Returns:
        int: The predicted class label.
        float: The probability of the feature vector belonging to that class.
    """
    # Distance measure functions
    distance_funcs = {
        'L1': lambda f1, f2: cityblock(f1, f2),
        'L2': lambda f1, f2: euclidean(f1, f2),
        'cosine': lambda f1, f2: cosine(f1, f2)
    }
    distance_func = distance_funcs[distance_type]

    # Calculate distances to all class centers
    distances = {label: distance_func(reduced_feature, center) for label, center in class_centers.items()}

    # Convert distances to similarities
    similarities = {label: 1 / (dist + 1e-8) for label, dist in distances.items()} 

    # Use the distance similarities to calculate a softmax score, which we can use for the ROC curve
    labels, values = zip(*similarities.items())
    probabilities = softmax(np.array(values))

    # Find the predicted class and its probability
    best_index = np.argmax(probabilities)
    predicted_class = labels[best_index]
    similarity = values[best_index]

    return predicted_class, similarity


def match_iris(feature, class_centers, reduced_class_centers, model, 
               rotations=[-9, -6, -3, 0, 3, 6, 9], distance_type='L2'):
    """
    Matches the unknown feature vector to the closest class using rotated templates and returns class probability.

    Args:
        feature (np.ndarray): Original feature vector of the unknown iris.
        class_centers (dict): Dictionary of class centers in the original feature space.
        reduced_class_centers (dict): Dictionary of class centers in the reduced feature space.
        model (LinearDiscriminantAnalysis): LDA model for projecting features.
        rotations (list): List of rotation angles to test for rotation invariance.
        distance_type (str): Type of distance metric ('L1', 'L2', or 'cosine').

    Returns:
        np.ndarray: Best rotated original feature vector of the iris (1536, 1)
        int: The predicted class label after evaluating all original feature rotated templates.
        float: The probability of the feature vector belonging to the predicted class.
        np.ndarray: Best rotated reduced feature vector of the iris (n_components, 1)
        int: The predicted class label after evaluating all reduced rotated templates.
        float: The probability of the reduced feature vector belonging to the predicted class.
    """

    min_score = float('inf')
    reduced_min_score = float('inf')

    best_class = None
    reduced_best_class = None

    best_probability = 0
    reduced_best_probability = 0

    # Iterate through each rotation angle
    for angle in rotations:

        # Rotate the original feature and reduce it 
        rotated_feature = np.roll(feature, angle)
        reduced_rotated_feature = model.transform([rotated_feature])[0]

        # Perform nearest center for both the original and reduced rotations
        predicted_class, predicted_similarity = compute_nearest_center(
            rotated_feature, class_centers, distance_type
        )
        reduced_predicted_class, reduced_predicted_similarity = compute_nearest_center(
            reduced_rotated_feature, reduced_class_centers, distance_type
        )

        # Calculate the matching score (distance to the nearest center)
        distance = euclidean(rotated_feature, class_centers[predicted_class])
        reduced_distance = euclidean(reduced_rotated_feature, reduced_class_centers[reduced_predicted_class])

        # Update if this rotation yields a closer match
        if distance < min_score:
            min_score = distance
            best_class = predicted_class
            best_similarity = predicted_similarity
            best_rotated_feature = rotated_feature

        if reduced_distance < reduced_min_score:
            reduced_min_score = reduced_distance
            reduced_best_class = reduced_predicted_class
            reduced_best_similarity = reduced_predicted_similarity
            reduced_best_rotated_feature = reduced_rotated_feature

    return best_rotated_feature, best_class, best_similarity, reduced_best_rotated_feature, reduced_best_class, reduced_best_similarity


