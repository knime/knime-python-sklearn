# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


import knime.extension as knext
import sklearn_ext
from util import utils
from sklearn.cross_decomposition import PLSRegression

# General settings for partial least squares nodes
@knext.parameter_group(label="Input")
class PartialLeastSquaresGeneralSettings:

    feature_columns = knext.MultiColumnParameter(
        "Feature columns",
        "Selection of columns used as features. Columns with nominal and numerical data are allowed.",
        column_filter=utils.is_nominal_numerical,
    )

    target_columns = knext.MultiColumnParameter(
        "Target columns",
        "Selection of column(s) used as target(s). Only columns with numerical data are allowed.",
        column_filter=utils.is_numerical,
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the features",
        """Define whether missing values in the input data should be skipped or whether 
        the node execution should fail on missing values.""",
        utils.MissingValueHandling.SkipRow.name,
        utils.MissingValueHandling,
    )

    def validate(self, values):
        n_features = len(values["feature_columns"])
        n_targets = len(values["target_columns"])

        if n_targets > n_features:
            raise knext.InvalidParametersError(
                f"""The number of targets ({n_targets}) cannot be larger than the number
                of features ({n_features})."""
            )


@knext.parameter_group(label="Algorithm Settings")
class PartialLeastSquaresAlgorithmSettings:
    n_components = knext.IntParameter(
        "Number of components to keep",
        """The number of basic components to compute. 
        Should be in `[1, min(number_of_samples, number_of_features, number_of_targets)]`. 
        Number of samples is the number of rows in the input table.""",
        default_value=1,
        min_value=1,
    )


@knext.node(
    name="Sklearn Partial Least Squares Regression Learner (Labs)",
    node_type=knext.NodeType.LEARNER,
    category=sklearn_ext.regression_learners_category,
    icon_path="icons/sklearn-logo.png",
)
@knext.input_table(
    name="Input table",
    description="""Numerical and nominal columns can be selected 
                    as feature columns from this table, and the
                    target columns must be numerical.""",
)
@knext.output_port(
    "Trained model",
    "Trained partial least squares regression model.",
    port_type=utils.regression_model_port_type,
)
class PartialLeastSquaresRegressionLearner(knext.PythonNode):
    """Partial least squares regression learner

    Learns partial least squares regression implemented by
    [scikit-learn](https://scikit-learn.org/) library.

    The model is trained with the selected numerical target column(s), and feature columns
    (can be numerical or nominal) from the input table.
    At least one numerical column and another numerical or nominal column is expected.
    By default, the rightmost numerical column is selected as the target column and all the
    remaining columns are selected as features.

    If there are at least two numerical columns and two other numerical or nominal
    columns available, rightmost two numerical columns are selected
    as targets and all the remaining columns are selected as features by default.
    If there are only two numerical (or one numerical and one nominal) columns are
    available, the rightmost column is selected as the target
    and the other column is selected as the feature by default.
    """

    general_settings = PartialLeastSquaresGeneralSettings()
    algorithm_settings = PartialLeastSquaresAlgorithmSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_table: knext.Schema,
    ):
        return self._create_spec(input_table)

    def _create_spec(
        self,
        input_table: knext.Schema,
    ):

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

        # Default feature and target column selection
        if (
            self.general_settings.target_columns
            and self.general_settings.feature_columns
        ) is None:
            # If there are at least two numerical columns and two other numerical or nominal
            # columns, n_components set to value 2
            if len(numerical_columns) >= 2 and len(nominal_numerical_columns) >= 4:
                self.algorithm_settings.n_components = 2
                # Set the last two numerical columns as default target columns
                if self.general_settings.target_columns is None:
                    self.general_settings.target_columns = [
                        c[0] for c in numerical_columns[-2:]
                    ]
                    self.general_settings.feature_columns = [
                        i[0]
                        for i in nominal_numerical_columns
                        if i[0] not in self.general_settings.target_columns
                    ]
            else:
                # Set the rightmost numerical column as the default target column
                self.general_settings.target_columns = [numerical_columns[-1][0]]
                # Set all the remaining columns as feature columns
                self.general_settings.feature_columns = [
                    i[0]
                    for i in nominal_numerical_columns
                    if i[0] not in self.general_settings.target_columns
                ]

        # Create schema from feature columns
        target_schema = knext.Schema.from_columns(
            [c for c in input_table if c.name in self.general_settings.target_columns]
        )

        # Create schema from feature columns
        feature_schema = knext.Schema.from_columns(
            [c for c in input_table if c.name in self.general_settings.feature_columns]
        )

        # Check if feature column(s) have been specified
        if not self.general_settings.feature_columns:
            raise knext.InvalidParametersError(
                """Feature column(s) have not been specified."""
            )

        # Check if target column(s) have been specified
        if not self.general_settings.target_columns:
            raise knext.InvalidParametersError(
                """Target column(s) have not been specified."""
            )

        # Check whether a column is selected as both a feature and a target
        for target_column in self.general_settings.target_columns:
            if target_column in self.general_settings.feature_columns:
                raise knext.InvalidParametersError(
                    f"""Numerical column "{target_column}"
                    cannot be selected as both target and feature column."""
                )

        n_features = len(self.general_settings.feature_columns)
        n_components = self.algorithm_settings.n_components

        if n_components > n_features:
            raise knext.InvalidParametersError(
                f"""The number of components ({n_components}) cannot be larger than the number
                of features ({n_features})."""
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

        # Number of components must be smaller than the number of samples
        if self.algorithm_settings.n_components > input_table.num_rows:
            raise knext.InvalidParametersError(
                f"""The number of components to keep {self.algorithm_settings.n_components}
                must be smaller than or equal to the number of samples (number of rows in 
                the input table) {input_table.num_rows}."""
            )

        # Convert input table to pandas dataframes
        dfX = input_table.to_pandas()

        # Skip rows with missing values if "SkipRow" option is selected
        # or fail execution if "Fail" is selected and there are missing values
        missing_value_handling_setting = utils.MissingValueHandling[
            self.general_settings.missing_value_handling
        ]

        feature_column_names = self.general_settings.feature_columns
        target_column_names = self.general_settings.target_columns

        df = utils.handle_missing_values(
            dfX,
            feature_column_names,
            target_column_names,
            missing_value_handling_setting,
        )

        # Filter feature and target columns
        dfx = df[feature_column_names]
        dfy = df[target_column_names]

        # Encode feature columns with one-hot encoder
        dfx_encoded, one_hot_encoder = utils.encode_train_feature_columns(dfx)

        # Perform PLS regression
        pls = PLSRegression(n_components=self.algorithm_settings.n_components)

        # Fit PLS model to data
        fitted_model = pls.fit(dfx_encoded, dfy)

        return utils.RegressionModelObject(
            self._create_spec(
                input_table.schema,
            ),
            fitted_model,
            one_hot_encoder,
            missing_value_handling_setting,
        )
