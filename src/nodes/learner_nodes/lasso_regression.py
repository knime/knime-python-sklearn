import knime_extension as knext
import sklearn_ext
from util import utils
from sklearn.linear_model import Lasso


@knext.parameter_group(label="Algorithm Settings")
class LassoAlgorithmSettings:

    alpha = knext.DoubleParameter(
        "Alpha",
        "Constant that multiplies the L1 term, controlling regularization strength.",
        default_value=1.0,
        min_value=0,
    )
    fit_intercept = knext.BoolParameter(
        "Calculate the intercept",
        """Whether to calculate the intercept for the model. Specifies if a constant 
        (a.k.a. bias or intercept) should be added to the decision function. If not 
        selected, no intercept will be used in calculations meaning that the data is 
        expected to be centered.""",
        True,
    )
    max_iter = knext.IntParameter(
        "Maximum number of iterations",
        """The maximum number of iterations run by the coordinate descent solver to 
        converge (the algorithm used to fit the model is coordinate descent).""",
        default_value=1000,
        min_value=1,
    )


@knext.node(
    name="Lasso Regression Learner (sklearn)",
    node_type=knext.NodeType.LEARNER,
    category=sklearn_ext.linear_models_category,
    icon_path="icons/sklearn-logo.png",
)
@knext.input_table(
    name="Input table",
    description="""Numerical and nominal columns can be selected as feature columns
                    from this table, and the target column must be numerical.""",
)
@knext.output_port(
    "Trained model",
    "Trained Lasso regression model.",
    port_type=utils.regression_model_port_type,
)
class LassoLearner(knext.PythonNode):
    """Lasso Learner

    Learns Lasso Regression implemented by scikit-learn library.
    It is a linear regression model trained with L1 prior as regularizer.

    The model is trained with the selected numerical target column, and feature columns
    (can be numerical or nominal) from the input table. By default, the rightmost numerical
    column is selected as the target column and all the remaining numerical columns are
    selected as features.
    """

    general_settings = utils.RegressionGeneralSettings()
    algo_settings = LassoAlgorithmSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_table: knext.Schema,
    ):
        return self._create_spec(input_table)

    def _create_spec(self, input_table: knext.Schema):

        numerical_columns = [
            (c.name, c.ktype) for c in input_table if utils.is_numerical(c)
        ]

        nominal_numerical_columns = [
            (c.name, c.ktype) for c in input_table if utils.is_nominal_numerical(c)
        ]

        # Expected at least one numerical column and another numerical or nominal column
        if len(numerical_columns) == 0 or (
            len(numerical_columns) == 1 and len(nominal_numerical_columns) == 1
        ):
            raise knext.InvalidParametersError(
                f"""The number of numerical columns are {len(numerical_columns)}
                ({numerical_columns}), expected at least one numerical column
                and another numerical or nominal column."""
            )

        # Set the rightmost numerical column as the default target column
        if self.general_settings.target_column is None:
            self.general_settings.target_column = numerical_columns[-1][0]

        # Create schema from the target column(s)
        target_schema = input_table[[self.general_settings.target_column]]

        # Set all the remaining columns as feature columns
        if self.general_settings.feature_columns is None:
            self.general_settings.feature_columns = [
                i[0]
                for i in nominal_numerical_columns
                if i[0] != self.general_settings.target_column
            ]

        # Create schema from feature columns
        feature_schema = knext.Schema.from_columns(
            [c for c in input_table if c.name in self.general_settings.feature_columns]
        )

        # Check if feature column(s) have been specified
        if not self.general_settings.feature_columns:
            raise knext.InvalidParametersError(
                """Feature column(s) have not been specified."""
            )

        # Check if target column have been specified
        if not self.general_settings.target_column:
            raise knext.InvalidParametersError(
                """Target column has not been specified."""
            )

        if self.general_settings.target_column in self.general_settings.feature_columns:
            raise knext.InvalidParametersError(
                f"""Numerical column "{self.general_settings.target_column}"
                cannot be selected as both target and feature column."""
            )

        return utils.RegressionModelObjectSpec(feature_schema, target_schema)

    def execute(
        self,
        execution_context: knext.ExecutionContext,
        input_table: knext.Table,
    ):
        """
        During configuration, all selected column names are filtered from the input table.
        During execution, the model is fitted according to specified settings with the selected
        feature and target columns.

        Input: Input table.
        Output: Trained model as a binary object.
        """

        # Convert input table to pandas dataframes
        dfX = input_table.to_pandas()

        # Skip rows with missing values if "SkipRow" option is selected
        # or fail execution if "Fail" is selected and there are missing values
        missing_value_handling_setting = utils.MissingValueHandling[
            self.general_settings.missing_value_handling
        ]
        dfX = utils.handle_missing_values(dfX, missing_value_handling_setting)

        # Filter feature and target columns
        dfx = dfX.filter(items=self.general_settings.feature_columns)
        dfy = dfX[self.general_settings.target_column]

        # Encode feature columns with one-hot encoder
        dfx_encoded, one_hot_encoder = utils.encode_train_feature_columns(dfx)

        # Perform Lasso regression
        reg = Lasso(
            alpha=self.algo_settings.alpha,
            fit_intercept=self.algo_settings.fit_intercept,
            max_iter=self.algo_settings.max_iter,
        )

        # Fit Lasso model to data
        model = reg.fit(dfx_encoded, dfy)

        return utils.RegressionModelObject(
            self._create_spec(input_table.schema),
            model,
            one_hot_encoder,
            missing_value_handling_setting,
        )
