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
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

# General settings for regression predictor node
@knext.parameter_group(label="Output")
class RegressionPredictorGeneralSettings:
    prediction_column = knext.StringParameter(
        "Custom prediction column name",
        "If no name is specified for the prediction column, it will default to `<target_column_name>_pred`.",
        default_value="",
    )


@knext.node(
    name="Regression Predictor (sklearn)",
    node_type=knext.NodeType.PREDICTOR,
    category=sklearn_ext.predictors_category,
    icon_path="icons/sklearn-logo.png",
)
@knext.input_port(
    "Trained model",
    "Trained regression predictor model.",
    port_type=utils.regression_model_port_type,
)
@knext.input_table(
    name="Input Data",
    description="""All feature columns used for training must also be present 
                    in the table for prediction.""",
)
@knext.output_table(
    name="Output Data",
    description="""The input table concatenated with a predictions column(s) with a 
                _pred suffix in their name.""",
)
class RegressionPredictor:
    """Regression Predictor

    The Regression Predictor Node accepts a trained model from any of the regression learner
    nodes and computes the predicted output for each row of the given input table.

    It is only executable if the test data contains the feature columns that were used by
    the learner model.
    """

    predictor_settings = RegressionPredictorGeneralSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        model_spec: utils.RegressionModelObjectSpec,
        table_schema: knext.Schema,
    ):
        """
        Checks if all selected feature columns are present in the test data.
        Extends the input schema with "_pred" suffix added target column names.
        """

        name_type = [(c.name, c.ktype) for c in table_schema]

        if not all((c.name, c.ktype) in name_type for c in model_spec.feature_schema):
            raise knext.InvalidParametersError(
                f"""The input table does not contain all selected feature columns. 
                Missing columns: {[c.name for c in model_spec.feature_schema if (c.name, c.ktype)
                not in name_type]}
                 """
            )

        LOGGER.info(
            f"""Default columns {' '.join([i for i in (model_spec.feature_schema.column_names + 
            model_spec.target_schema.column_names)])} selected during training were 
            selected for prediction."""
        )

        y_pred = utils.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_spec.target_schema
        )

        # Add prediction column names in the schema
        for column_name in y_pred:
            table_schema = table_schema.append(
                knext.Column(ktype=knext.double(), name=column_name)
            )

        return table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        port_object: utils.RegressionModelObject,
        table: knext.Table,
    ):
        """
        Feature columns and the target column(s) are selected and transformed.
        """

        df = table.to_pandas()

        # Get feature columns
        x_value_columns = port_object.spec.feature_schema.column_names

        # Get target columns
        y_value_columns = port_object.spec.target_schema.column_names

        # Get target column for the prediction column
        y_pred = utils.get_prediction_column_name(
            self.predictor_settings.prediction_column, port_object.spec.target_schema
        )

        # Skip rows with missing values if "SkipRow" option of the learner node is selected
        # or fail execution if "Fail" is selected and there are missing values
        df = utils.handle_missing_values(
            df,
            x_value_columns,
            y_value_columns,
            port_object.handle_missing_values,
        )

        features = df[x_value_columns]

        # Encode test feature columns with one-hot encoder
        dfx_encoded = utils.encode_test_feature_columns(
            features,
            port_object.one_hot_encoder,
        )

        features_encoded = dfx_encoded

        # Prediction
        dfx_predictions = pd.DataFrame(
            port_object.predict(features_encoded), columns=y_pred
        )

        # Concatenate predictions dataframe with features dataframe
        df = utils.concatenate_predictions_with_input_table(df, dfx_predictions)

        return knext.Table.from_pandas(df)


# General settings for classification predictor node
@knext.parameter_group(label="Output")
class ClassificationPredictorGeneralSettings:

    prediction_column = knext.StringParameter(
        "Custom prediction column name",
        "If no name is specified for the prediction column, it will default to `<target_column_name>_pred`.",
        default_value="",
    )

    predict_probs = knext.BoolParameter(
        "Predict probability estimates",
        "Predict probability estimates for each target class.",
        True,
    )

    prob_columns_suffix = knext.StringParameter(
        "Suffix for probability columns",
        "Allows to add a suffix for the class probability columns.",
        default_value="",
    )


@knext.node(
    name="Classification Predictor (sklearn)",
    node_type=knext.NodeType.PREDICTOR,
    category=sklearn_ext.predictors_category,
    icon_path="icons/sklearn-logo.png",
)
@knext.input_port(
    "Trained Model",
    "Trained classification predictor model.",
    port_type=utils.classification_model_port_type,
)
@knext.input_table(
    name="Input Data",
    description="""All feature columns used for training must also be present 
                    in the table for prediction.""",
)
@knext.output_table(
    name="Output Data",
    description="""The input table concatenated with a predictions column and 
                    optionally class probabilities.""",
)
class ClassificationPredictor:
    """Classification Predictor

    The Classification Predictor Node accepts a trained model from any of the classification
    learner nodes and computes the predicted class for each row of the given input table.
    Optionally, class probability estimates for each row can also be predicted.

    It is only executable if the test data contains the columns that are used by the learner model.
    """

    predictor_settings = ClassificationPredictorGeneralSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        model_spec: utils.ClassificationModelObjectSpec,
        table_schema: knext.Schema,
    ):
        """
        Checks if all selected feature columns are present in the test data.
        Extends the input schema with "_pred" suffix added target column names.
        """

        name_type = [(c.name, c.ktype) for c in table_schema]

        if not all((c.name, c.ktype) in name_type for c in model_spec.feature_schema):
            raise knext.InvalidParametersError(
                f"""The input table does not contain all selected feature columns. 
                Missing columns: {[c.name for c in model_spec.feature_schema if (c.name, c.ktype) 
                not in name_type]}
                 """
            )

        LOGGER.info(
            f"""Default columns {' '.join([i[0] for i in name_type])} selected during training 
            were selected for prediction."""
        )

        y_pred = utils.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_spec.target_schema
        )

        # Add prediction column names in the schema
        for column_name in y_pred:
            table_schema = table_schema.append(
                knext.Column(ktype=knext.string(), name=column_name)
            )

        # Add probability estimate column names in the schema
        if self.predictor_settings.predict_probs:
            for column in model_spec.class_probability_schema:
                if self.predictor_settings.prob_columns_suffix:
                    table_schema = table_schema.append(
                        knext.Column(
                            ktype=column.ktype,
                            name=f"{column.name}{self.predictor_settings.prob_columns_suffix}",
                        )
                    )
                else:
                    table_schema = table_schema.append(
                        knext.Column(ktype=column.ktype, name=column.name)
                    )

        return table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        port_object: utils.ClassificationModelObject,
        input_test_table: knext.Table,
    ):
        """
        During execution, GP classification is performed on the test data.
        Optionally, class probability estimates are calculated.

        Input: Test data.
        Output: Table with predicted target column.
        """

        # Convert test table to pandas dataframe
        df = input_test_table.to_pandas()

        # Get feature columns
        x_value_columns = port_object.spec.feature_schema.column_names

        # Get target column
        y_value_columns = port_object.spec.target_schema.column_names

        # Skip rows with missing values if "SkipRow" option of the learner node is selected
        # or fail execution if "Fail" is selected and there are missing values
        df = utils.handle_missing_values(
            df,
            x_value_columns,
            y_value_columns,
            port_object.handle_missing_values,
        )

        # Get feature column names
        feature_columns = port_object.spec.feature_schema.column_names

        # Get target column for the prediction column
        y_pred = utils.get_prediction_column_name(
            self.predictor_settings.prediction_column, port_object.spec.target_schema
        )

        # Get selected feature columns
        features = df[feature_columns]

        # Encode feature columns with one-hot encoder
        dfx_encoded = utils.encode_test_feature_columns(
            features,
            port_object.one_hot_encoder,
        )

        # Perform classification on encoded test data
        dfx_predictions = pd.DataFrame(port_object.predict(dfx_encoded), columns=y_pred)

        # Decode target class names
        dfx_predictions = port_object.decode_target_values(dfx_predictions)

        # Concatenate predictions dataframe with features dataframe
        df = utils.concatenate_predictions_with_input_table(df, dfx_predictions)

        if self.predictor_settings.predict_probs:
            # Probability estimates
            estimates = port_object.predict_probabilities(dfx_encoded)

            if self.predictor_settings.prediction_column:
                # Get original target column name (and not the custom name)
                # for class probability column names
                y_pred = utils.get_prediction_column_name(
                    "", port_object.spec.target_schema
                )

            # Class probability estimates' column names are adjusted to
            # “P ({predicted_column_name}={class_name})”
            class_probability_column_names = (
                port_object.get_class_probability_column_names(
                    y_pred,
                    self.predictor_settings.prob_columns_suffix,
                )
            )

            # Convert estimates to dataframe
            estimates_df = pd.DataFrame(
                estimates, columns=class_probability_column_names
            )

            # Adjust dataframe indexes
            estimates_df.index = df.index

            # Concatenate probability estimates dataframe with the main(features + predictions)
            # dataframe
            df = pd.concat([df, estimates_df], axis=1)

        return knext.Table.from_pandas(df)
