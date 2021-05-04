
from .base import _check_is_fitted
import numpy as np

def uncertainty(classifier, X) -> np.ndarray:
    """
    Classification uncertainty of the classifier for the provided samples.
    Args:
        classifier: The classifier for which the uncertainty is to be measured.
        X: array like
    Returns:
        Classifier uncertainty, which is 1 - max(P(prediction is correct)).
    """
    # calculate uncertainty for each point provided
    if _check_is_fitted(classifier, X):
        classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
    else:
        return np.ones(shape=(X.shape[0], ))

    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    return uncertainty
