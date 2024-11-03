import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cityblock, euclidean, cosine
from scipy.special import softmax


def reduce_dimensionality(features, labels, n_components=100):
    """
    Reduces the dimensionality of feature vectors using Fisher Linear Discriminant Analysis.
    
    Args:
        features (np.ndarray): Array of original feature vectors (shape: [num_samples, num_features]).
        labels (np.ndarray): Array of labels corresponding to each feature vector.
        n_components (int): Number of dimensions for reduced feature vector space. Can't be larger than the number of irises

    Returns:
        np.ndarray, np.ndarray: The LDA model and a dictionary containing class centers
    """

    # Fit the LDA model
    model = LinearDiscriminantAnalysis(n_components=n_components)
    model.fit(features, labels)
    reduced_features = model.transform(features)

    # Compute the class centers on the reduced features
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
    predicted_probability = probabilities[best_index]

    return predicted_class, predicted_probability


def match_iris(feature, class_centers, model, rotations=[-9, -6, -3, 0, 3, 6, 9], distance_type='L2'):
    """
    Matches the unknown feature vector to the closest class using rotated templates and returns class probability.

    Args:
        feature (np.ndarray): Original feature vector of the unknown iris.
        class_centers (dict): Dictionary of class centers in the reduced feature space.
        model (LinearDiscriminantAnalysis): LDA model for projecting features.
        rotations (list): List of rotation angles to test for rotation invariance.
        distance_type (str): Type of distance metric ('L1', 'L2', or 'cosine').

    Returns:
        np.ndarray: Original feature vector of the iris (1536, 1)
        np.ndarray: Best rotated feature vector of the iris (n_components, 1)
        int: The predicted class label after evaluating all rotated templates.
        float: The probability of the feature vector belonging to the predicted class.
    """

    min_score = float('inf')
    best_class = None
    best_probability = 0

    # Iterate through each rotation angle
    for angle in rotations:

        # Simulate the rotations in the paper by applying each angle offset
        rotated_feature = np.roll(feature, angle)

        # Project the rotated feature to the reduced space
        reduced_rotated_feature = model.transform([rotated_feature])[0]

        # Perform nearest center classification
        predicted_class, probability = compute_nearest_center(reduced_rotated_feature, class_centers, distance_type)

        # Calculate the matching score (distance to the nearest center)
        distance = euclidean(reduced_rotated_feature, class_centers[predicted_class])

        # Update if this rotation yields a closer match
        if distance < min_score:
            min_score = distance
            best_class = predicted_class
            best_probability = probability
            best_rotated_feature = reduced_rotated_feature

    return feature, best_rotated_feature, best_class, best_probability


