import knime_extension as knext
import sklearn_ext
from util import utils
from sklearn.gaussian_process import GaussianProcessClassifier


@knext.parameter_group(label="Input")
class ClassificationInputSettings:

    feature_columns = knext.MultiColumnParameter(
        "Feature columns",
        """Selection of columns used as feature columns. Columns with nominal and numerical 
        data are allowed.""",
        column_filter=utils.is_nominal_numerical,
    )

    target_column = knext.ColumnParameter(
        "Target column",
        """Selection of column used as the target column. Only columns with nominal data 
        are allowed.""",
        column_filter=utils.is_nominal,
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the features",
        """Define whether missing values in the input data should be skipped or whether the 
        node execution should fail on missing values.""",
        utils.MissingValueHandling.SkipRow.name,
        utils.MissingValueHandling,
    )


@knext.parameter_group(label="Algorithm Settings")
class GaussianProcessClassificationAlgorithmSettings:
    class KernelOptions(knext.EnumParameterOptions):
        Default = (
            "Default",
            "1.0 * RBF(1.0)",
        )
        RationalQuadratic = (
            "Rational quadratic kernel",
            "sklearn's RationalQuadratic kernel",
        )
        DotProduct = ("Dot-Product Kernel", "sklearn's DotProduct kernel")
        RBF = ("RBF", "sklearn's Radial basis function kernel")
        WhiteKernel = ("White Kernel", "sklearn's WhiteKernel")

    class MultiClassOptions(knext.EnumParameterOptions):
        OVR = (
            "One-vs-Rest",
            "One binary Gaussian process classifier is fitted for each class. This is the default.",
        )
        OVO = (
            "One-vs-One",
            "One binary Gaussian process classifier is fitted for each pair of classes",
        )

    kernel = knext.EnumParameter(
        "Kernel",
        """The kernel specifying the covariance function of the GP. The default kernel 
        is '1.0 * RBF(1.0)' is used as default.""",
        KernelOptions.Default.name,
        KernelOptions,
    )

    multi_class_method = knext.EnumParameter(
        "Multi-class classification method",
        "Multi-class classification method selection.",
        MultiClassOptions.OVR.name,
        MultiClassOptions,
    )


@knext.node(
    name="Gaussian Process Classification Learner (sklearn)",
    node_type=knext.NodeType.LEARNER,
    category=sklearn_ext.gpc_category,
    icon_path="icons/sklearn-logo.png",
)
@knext.input_table(
    name="Input table",
    description="""Nominal and numerical columns can be selected as feature columns
                    from this table, and the target column must be nominal.""",
)
@knext.output_port(
    "Trained model",
    "Trained Gaussian process classification model.",
    port_type=utils.classification_model_port_type,
)
class GaussianProcessClassificationLearner(knext.PythonNode):
    """Gaussian Process Classification Learner

    Learns Gaussian Process Classification based on Laplace approximation
    implemented by scikit-learn library. The implementation follows the
    algorithm in sections 3.1, 3.2 and 5.1 of the paper "Gaussian Processes
    for Machine Learning" by Carl E. Rasmussen and Christopher K.I. Williams (2006).

    The model is trained with the selected nominal target column, and feature columns
    (can be nominal or numerical) from the input table. By default, the rightmost nominal
    column is selected as the target column and all the remaining nominal and numerical
    columns are selected as features.
    """

    general_settings = ClassificationInputSettings()
    algorithm_settings = GaussianProcessClassificationAlgorithmSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_table: knext.Schema,
    ):
        return self._create_spec(input_table, class_probability_schema=None)

    def _create_spec(
        self,
        input_table: knext.Schema,
        class_probability_schema: knext.Schema,
    ):

        nominal_columns = [
            (c.name, c.ktype) for c in input_table if utils.is_nominal(c)
        ]

        nominal_numerical_columns = [
            (c.name, c.ktype) for c in input_table if utils.is_nominal_numerical(c)
        ]

        # Expected at least one nominal column and another numerical or nominal column
        if len(nominal_columns) == 0 or (
            len(nominal_columns) == 1 and len(nominal_numerical_columns) == 1
        ):
            raise knext.InvalidParametersError(
                f"""The number of nominal columns are {len(nominal_columns)}
                ({nominal_columns}), expected at least one nominal column
                and another numeric or nominal column."""
            )

        # Set the rightmost nominal column as the default target column
        if self.general_settings.target_column is None:
            self.general_settings.target_column = nominal_columns[-1][0]

        # Create schema from the target column
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

        # Check if the target column have been specified
        if not self.general_settings.target_column:
            raise knext.InvalidParametersError(
                """Target column has not been specified."""
            )

        # Check whether a column is selected as both a feature and a target
        if self.general_settings.target_column in self.general_settings.feature_columns:
            raise knext.InvalidParametersError(
                f"""Nominal column "{self.general_settings.target_column}"
                cannot be selected as both target and feature column."""
            )

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")

        return utils.ClassificationModelObjectSpec(
            feature_schema, target_schema, class_probability_schema
        )

    def execute(
        self, execution_context: knext.ExecutionContext, input_table: knext.Table
    ):
        """
        During configuration, all selected column names are filtered from the input table.
        During execution, the model is fitted according to specified settings with the selected
        feature and target columns.

        Input: Input table.
        Output: Trained model as a binary object.
        """

        # Convert input table to pandas dataframe
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

        # Get class names(=column names) for probability estimates
        prob_estimates_column_names = utils.pd.DataFrame(columns=dfy.unique())

        # Update the table schema with probability estimates column names
        prob_estimates_column_names = prob_estimates_column_names.reindex(
            sorted(prob_estimates_column_names.columns), axis=1
        )
        input_table_schema = input_table.schema
        for column_name in prob_estimates_column_names:
            input_table_schema = input_table_schema.append(
                knext.Column(
                    ktype=knext.double(),
                    name=f"P ({self.general_settings.target_column}_pred={column_name})",
                )
            ).get()

        # Encode feature columns with one-hot encoder
        dfx_encoded, one_hot_encoder = utils.encode_train_feature_columns(dfx)

        # Encode target column labels between 0..(n_classes-1)
        dfy_encoded, label_encoder = utils.encode_target_column(
            dfy,
            self.general_settings.target_column,
        )

        # Get the selected kernel object
        if self.algorithm_settings.kernel in utils.kernels:
            selected_kernel = utils.kernels[self.algorithm_settings.kernel]()
        else:
            selected_kernel = None

        # Get the selected multi classification method
        MCO = GaussianProcessClassificationAlgorithmSettings.MultiClassOptions
        multi_class_methods = {
            MCO.OVR.name: "one_vs_rest",
            MCO.OVO.name: "one_vs_one",
        }

        selected_mc_method = multi_class_methods[
            self.algorithm_settings.multi_class_method
        ]

        # Perform GP classification
        reg = GaussianProcessClassifier(
            kernel=selected_kernel, multi_class=selected_mc_method
        )

        # Fit GPC model to data
        model = reg.fit(dfx_encoded, dfy_encoded)

        # Get the number of unique class labels
        num_classes = len(model.classes_)

        # Create a schema that contains the class names by getting the last
        # "num_classes" number of columns from the input table.
        # class_probability_schema is then used if user wants to predict
        # probability estimates for the test data.
        if num_classes:
            class_probability_schema = input_table_schema[-num_classes:].get()
        else:
            class_probability_schema = None

        return utils.ClassificationModelObject(
            self._create_spec(input_table_schema, class_probability_schema),
            model,
            label_encoder,
            one_hot_encoder,
            missing_value_handling_setting,
        )
