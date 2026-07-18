from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def tune_xgb_hyperparameters(
    x_train,
    x_cv,
    y_train,
    y_cv,
    **search_space
):
    
    """
    Tune XGBoost hyperparameters by evaluating different candidate values
    and selecting the value that achieves the highest validation accuracy.
    
    This function performs a simple sequential hyperparameter search. Each
    hyperparameter is tested independently while keeping all other parameters
    at their default values. It does not evaluate combinations of multiple
    hyperparameters like sklearn.model_selection.GridSearchCV.
    
    Parameters
    ----------
    x_train : array-like
        Training feature data used to fit the XGBoost model.
    
    x_cv : array-like
        Validation feature data used to evaluate model performance.
    
    y_train : array-like
        Training target labels.
    
    y_cv : array-like
        Validation target labels.
    
    **search_space : dict
        Hyperparameter names mapped to lists of candidate values to test.
    
        Example:
            max_depth=[2, 4, 6, 8],
            subsample=[0.5, 0.7, 0.9]
    
        Each value in the list will be used to train and evaluate a separate
        XGBClassifier model.
    
    Returns
    -------
    best_params : dict
        Dictionary containing each hyperparameter and the candidate value that
        achieved the highest validation accuracy.
    
    train_scores : dict
        Dictionary containing training accuracy scores for each tested
        hyperparameter.
    
        Example:
            {
                "max_depth_train_accuracy": [0.91, 0.94, 0.96]
            }
    
    cv_scores : dict
        Dictionary containing validation accuracy scores for each tested
        hyperparameter.
    
        Example:
            {
                "max_depth_cv_accuracy": [0.80, 0.83, 0.82]
            }
    
    Notes
    -----
    - This function tunes one hyperparameter at a time.
    - Validation accuracy is used as the metric for selecting the best value.
    - A fixed random_state is used internally for reproducible results.
    - For a full grid search across all parameter combinations, use
      sklearn.model_selection.GridSearchCV.
    
    Examples
    --------
    >>> best_params, train_scores, cv_scores = tune_data(
    ...     x_train,
    ...     x_cv,
    ...     y_train,
    ...     y_cv,
    ...     max_depth=[2, 4, 6, 8],
    ...     subsample=[0.5, 0.7, 0.9]
    ... )
    
    >>> best_params
    {
        "max_depth": 6,
        "subsample": 0.9
    }
    """
    
    best_params = {}
    
    train_scores = {}
    cv_scores = {}

    best_cv_score = 0
    best_value = 0

    for parameter_name, values in search_space.items():

        for candidate_value in values:

            model_params = {
                parameter_name: candidate_value,
            }

            model = XGBClassifier(**model_params, random_state=36)

            model.fit(x_train, y_train)

            train_predictions = model.predict(x_train)
            cv_predictions = model.predict(x_cv)

            train_accuracy = accuracy_score(
                y_train,
                train_predictions
            )

            cv_accuracy = accuracy_score(
                y_cv,
                cv_predictions
            )

            train_key = f"{parameter_name}_train_accuracy"
            cv_key = f"{parameter_name}_cv_accuracy"

            if train_key not in train_scores:
                train_scores[train_key] = []

            if cv_key not in cv_scores:
                cv_scores[cv_key] = []

            train_scores[train_key].append(train_accuracy)
            cv_scores[cv_key].append(cv_accuracy)

            if cv_accuracy > best_cv_score:
                
                best_cv_score = cv_accuracy
                best_value = candidate_value

        best_params[parameter_name] = best_value

    return best_params, train_scores, cv_scores    
