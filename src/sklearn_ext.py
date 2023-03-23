import knime_extension as knext

main_category = knext.category(
    path="/community",
    level_id="sklearn_ext",
    name="Sklearn",
    description="Nodes for Sklearn library",
    icon="icons/sklearn-logo.png",
)

learners_category = knext.category(
    path=main_category,
    level_id="sklearn_learners",
    name="Sklearn Learners",
    description="Learner nodes for regression and classification",
    icon="icons/sklearn-logo.png",
)

partial_least_squares_category = knext.category(
    path=learners_category,
    level_id="pls",
    name="Partial Least Squares",
    description="Nodes for the Partial Least Squares family",
    icon="icons/sklearn-logo.png",
)

linear_models_category = knext.category(
    path=learners_category,
    level_id="linear_models",
    name="Linear Models",
    description="Nodes for linear models",
    icon="icons/sklearn-logo.png",
)

gaussian_processes_category = knext.category(
    path=learners_category,
    level_id="gaussian_processes",
    name="Gaussian Processes",
    description="Nodes for gaussian processes",
    icon="icons/sklearn-logo.png",
)

gpr_category = knext.category(
    path=gaussian_processes_category,
    level_id="gaussian_process_regression",
    name="Regression",
    description="Nodes for gaussian process regression",
    icon="icons/sklearn-logo.png",
)

gpc_category = knext.category(
    path=gaussian_processes_category,
    level_id="gaussian_process_classification",
    name="Classification",
    description="Nodes for gaussian process classification",
    icon="icons/sklearn-logo.png",
)

predictors_category = knext.category(
    path=main_category,
    level_id="sklearn_predictors",
    name="Sklearn Predictors",
    description="Predictor nodes for regression and classification",
    icon="icons/sklearn-logo.png",
)

# Node files
import nodes.learner_nodes.gaussian_process_regression
import nodes.learner_nodes.gaussian_process_classification
import nodes.learner_nodes.lasso_regression
import nodes.learner_nodes.partial_least_squares_regression
import nodes.predictor_nodes.predictor_nodes
