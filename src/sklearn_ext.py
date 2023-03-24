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
    level_id="learners",
    name="Sklearn Learners",
    description="Learner nodes",
    icon="icons/sklearn-logo.png",
)

regression_learners_category = knext.category(
    path=learners_category,
    level_id="regression",
    name="Regression",
    description="Nodes for sklearn regression algorithms",
    icon="icons/sklearn-logo.png",
)

classification_learners_category = knext.category(
    path=learners_category,
    level_id="classification",
    name="Classification",
    description="Nodes for sklearn classification algorithms",
    icon="icons/sklearn-logo.png",
)

predictors_category = knext.category(
    path=main_category,
    level_id="predictors",
    name="Sklearn Predictors",
    description="Predictor nodes for sklearn algorithms",
    icon="icons/sklearn-logo.png",
)

# Node files
import nodes.learner_nodes.gaussian_process_regression
import nodes.learner_nodes.gaussian_process_classification
import nodes.learner_nodes.lasso_regression
import nodes.learner_nodes.partial_least_squares_regression
import nodes.predictor_nodes.predictor_nodes
