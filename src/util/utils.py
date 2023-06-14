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
from typing import Union
import logging
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.gaussian_process.kernels import (
    RationalQuadratic,
    ConstantKernel,
    DotProduct,
    RBF,
    WhiteKernel,
)

LOGGER = logging.getLogger(__name__)

kernels = {
    "RationalQuadratic": RationalQuadratic,
    "ConsantKernel": ConstantKernel,
    "DotProduct": DotProduct,
    "RBF": RBF,
    "WhiteKernel": WhiteKernel,
}


class MissingValueHandling(knext.EnumParameterOptions):
    SkipRow = (
        "Skip rows with missing values.",
        "Rows with missing values will not be used for the training.",
    )
    Fail = (
        "Fail on observing missing values.",
        "Learner node will fail during the execution.",
    )


def is_numerical(column):
    # Filter numerical columns
    return (
        column.ktype == knext.double()
        or column.ktype == knext.int32()
        or column.ktype == knext.int64()
    )


def is_nominal(column):
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def is_nominal_numerical(column):
    # Filter nominal and numerical columns
    return is_numerical(column) or is_nominal(column)


def is_binary(df):
    # Check if a column is binary
    series = df.squeeze()
    return sorted(series.unique()) == [0, 1]


def split_nominal_numerical(dfx):
    # Split nominal and numerical columns for encoding
    dfx_nominal = dfx.select_dtypes(include=["string", "bool"])
    dfx_numerical = dfx.select_dtypes(exclude=["string", "bool"])

    return dfx_nominal, dfx_numerical


def skip_missing_values(df: pd.DataFrame, col_names: list[str]):
    # Drops rows with missing values
    df_cleaned = df.dropna(subset=col_names, how="any")

    n_rows = len(df)
    n_cleaned_rows = len(df_cleaned)

    if n_cleaned_rows < n_rows:
        LOGGER.info(
            f"{len(df) - len(df_cleaned)} / {len(df)} number of rows are skipped."
        )

    return df_cleaned


def handle_missing_values(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: Union[str, list[str]],
    missing_value_handling_setting: MissingValueHandling,
):
    # Drops rows if SkipRow option is selected, otherwise fails
    # if there are any missing values in the data (=Fail option is selected)
    if not isinstance(target_columns, list):
        col_names = feature_columns + [target_columns]
    else:
        col_names = feature_columns + target_columns

    df_filtered = df[col_names]

    if (
        missing_value_handling_setting == MissingValueHandling.SkipRow
        and df_filtered.isna().any().any()
    ):
        df = skip_missing_values(df, col_names)
    else:
        if df_filtered.isna().any().any():
            raise knext.InvalidParametersError(
                "There are missing values in the input data."
            )

    if df.empty:
        raise knext.InvalidParametersError(
            f"""All rows are skipped due to missing values."""
        )

    return df


def encode_train_feature_columns(dfx):
    """
    Encode categorical training dataframe columns into a one-hot numeric array.
    Encoded numpy array is then converted back into a pandas dataframe.

    Returns: concatenated dataframe with columns ordered as ->
            [numerical columns, encoded nominal columns]
    """

    # Separate nominal and numerical columns
    dfx_nominal, dfx_numerical = split_nominal_numerical(dfx)

    # Encode nominal columns
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    dfx_nominal_encoded = pd.DataFrame(one_hot_encoder.fit_transform(dfx_nominal))

    # Concatenate nominal and numerical dataframes back
    dfx_numerical.index = dfx_nominal_encoded.index
    dfx_encoded = pd.concat((dfx_numerical, dfx_nominal_encoded), 1)

    # Convert feature column names to string
    # (due to: feature names are only supported if all input features
    # have string names in scikit-learn 1.2)
    dfx_encoded.columns = dfx_encoded.columns.astype(str)

    return dfx_encoded, one_hot_encoder


def encode_test_feature_columns(dfx, encoder):
    """
    Encode categorical test dataframe columns into a one-hot numeric array.
    Encoded numpy array is then converted back into a pandas dataframe.

    Each column is encoded using the training data encoder.

    Returns: concatenated dataframe with columns ordered as ->
            [numerical columns, encoded nominal columns]
    """

    # Separate nominal and numerical columns
    dfx_nominal, dfx_numerical = split_nominal_numerical(dfx)

    # Encode nominal columns using trainind data encoder
    dfx_nominal_encoded = pd.DataFrame(encoder.transform(dfx_nominal))

    # Concatenate nominal and numerical dataframes back
    dfx_numerical.index = dfx_nominal_encoded.index
    dfx_encoded = pd.concat((dfx_numerical, dfx_nominal_encoded), 1)

    # Convert feature column names to string
    # (due to: feature names are only supported if all input features
    # have string names in scikit-learn 1.2)
    dfx_encoded.columns = dfx_encoded.columns.astype(str)

    return dfx_encoded


def encode_target_column(y, target_column):
    """
    Encode target labels between 0..(n_classes-1).
    Encoded target numpy array is then converted back into pandas dataframe.

    Output: Encoded target column and its label encoder object to be used for decoding.
    """

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Convert np array to dataframe
    dfy_series = pd.Series(y_encoded.squeeze())
    dfy_encoded = pd.DataFrame({target_column: dfy_series})

    return dfy_encoded, le


def get_prediction_column_name(prediction_column, target_schema):
    """
    Adds "_pred" suffix to prediction column names.
    """
    if prediction_column.strip() != "":
        y_pred = [prediction_column for y in target_schema.column_names]
    else:
        y_pred = [f"{y}_pred" for y in target_schema.column_names]

    return y_pred


def get_prob_suffix_column_name(prob_column_suffix, class_probability_schema):
    """
    Adds "prob_column_suffix" suffix to class probability column names.
    """
    if prob_column_suffix.strip() != "":
        y_pred = []
        y_pred.append(prob_column_suffix)
    else:
        y_pred = [f"{y}{prob_column_suffix}" for y in class_probability_schema]

    return y_pred


class RegressionModelObjectSpec(knext.PortObjectSpec):
    """
    Spec for regression model port object.
    """

    def __init__(
        self,
        feature_schema: knext.Schema,
        target_schema: knext.Schema,
    ) -> None:
        self._feature_schema = feature_schema
        self._target_schema = target_schema

    def serialize(self) -> dict:
        return {
            "feature_schema": self._feature_schema.serialize(),
            "target_schema": self._target_schema.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RegressionModelObjectSpec":
        return cls(
            knext.Schema.deserialize(data["feature_schema"]),
            knext.Schema.deserialize(data["target_schema"]),
        )

    @property
    def feature_schema(self) -> knext.Schema:
        return self._feature_schema

    @property
    def target_schema(self) -> knext.Schema:
        return self._target_schema


class RegressionModelObject(knext.PortObject):
    def __init__(
        self,
        spec: RegressionModelObjectSpec,
        model,
        one_hot_encoder,
        missing_value_handling_setting,
    ) -> None:
        super().__init__(spec)
        self._model = model
        self._one_hot_encoder = one_hot_encoder
        self._missing_value_handling_setting = missing_value_handling_setting

    def serialize(self) -> bytes:
        return pickle.dumps(
            (self._model, self._one_hot_encoder, self._missing_value_handling_setting)
        )

    @property
    def spec(self) -> RegressionModelObjectSpec:
        return super().spec

    @property
    def one_hot_encoder(self) -> OneHotEncoder:
        return self._one_hot_encoder

    @property
    def handle_missing_values(self) -> knext.EnumParameter:
        return self._missing_value_handling_setting

    @classmethod
    def deserialize(
        cls, spec: RegressionModelObjectSpec, data: bytes
    ) -> "RegressionModelObject":
        model, one_hot_encoder, missing_value_handling_setting = pickle.loads(data)
        return cls(spec, model, one_hot_encoder, missing_value_handling_setting)

    def predict(self, data):
        return self._model.predict(data)


class ClassificationModelObjectSpec(knext.PortObjectSpec):
    """
    Spec for classification model port object.
    """

    def __init__(
        self,
        feature_schema: knext.Schema,
        target_schema: knext.Schema,
        class_probability_schema: knext.Schema,
    ) -> None:
        self._feature_schema = feature_schema
        self._target_schema = target_schema
        self._class_probability_schema = class_probability_schema

    def serialize(self) -> dict:
        return {
            "feature_schema": self._feature_schema.serialize(),
            "target_schema": self._target_schema.serialize(),
            "class_probability_schema": self._class_probability_schema.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RegressionModelObjectSpec":
        return cls(
            knext.Schema.deserialize(data["feature_schema"]),
            knext.Schema.deserialize(data["target_schema"]),
            knext.Schema.deserialize(data["class_probability_schema"]),
        )

    @property
    def feature_schema(self) -> knext.Schema:
        return self._feature_schema

    @property
    def target_schema(self) -> knext.Schema:
        return self._target_schema

    @property
    def class_probability_schema(self) -> knext.Schema:
        return self._class_probability_schema


class ClassificationModelObject(knext.PortObject):
    def __init__(
        self,
        spec: ClassificationModelObjectSpec,
        model,
        label_enc,
        one_hot_encoder,
        missing_value_handling_setting,
    ) -> None:
        super().__init__(spec)
        self._model = model
        self._label_enc = label_enc
        self._one_hot_encoder = one_hot_encoder
        self._missing_value_handling_setting = missing_value_handling_setting

    def serialize(self) -> bytes:
        return pickle.dumps(
            (
                self._model,
                self._label_enc,
                self._one_hot_encoder,
                self._missing_value_handling_setting,
            )
        )

    @property
    def spec(self) -> ClassificationModelObjectSpec:
        return super().spec

    @property
    def one_hot_encoder(self) -> OneHotEncoder:
        return self._one_hot_encoder

    @property
    def handle_missing_values(self) -> knext.EnumParameter:
        return self._missing_value_handling_setting

    @classmethod
    def deserialize(
        cls, spec: ClassificationModelObjectSpec, data: bytes
    ) -> "ClassificationModelObject":
        (
            model,
            label_encoder,
            one_hot_encoder,
            missing_value_handling_setting,
        ) = pickle.loads(data)
        return cls(
            spec, model, label_encoder, one_hot_encoder, missing_value_handling_setting
        )

    def predict(self, data):
        return self._model.predict(data)

    def predict_probabilities(self, data):
        return self._model.predict_proba(data)

    def decode_target_values(self, predicted_column):
        # Encoder name mapping in "encoded_class : original_class" format
        le_name_mapping = dict(
            zip(range(len(self._label_enc.classes_)), self._label_enc.classes_)
        )
        decoded_column = predicted_column.replace(le_name_mapping)

        return decoded_column

    def get_class_probability_column_names(self, predicted_column_name, suffix):
        # Class probability column names are adjusted to “P ({predicted_col_name}={class_name})”
        class_probability_column_names = self._label_enc.classes_
        for i in range(len(class_probability_column_names)):
            class_probability_column_names[
                i
            ] = f"P ({predicted_column_name[0]}={class_probability_column_names[i]}){suffix}"

        return class_probability_column_names


regression_model_port_type = knext.port_type(
    name="Regression Predictor model port type",
    object_class=RegressionModelObject,
    spec_class=RegressionModelObjectSpec,
)

classification_model_port_type = knext.port_type(
    name="Classification Predictor model port type",
    object_class=ClassificationModelObject,
    spec_class=ClassificationModelObjectSpec,
)

# General settings for Lasso and GPR learner nodes
@knext.parameter_group(label="Input")
class RegressionGeneralSettings:

    feature_columns = knext.MultiColumnParameter(
        "Feature columns",
        """Selection of columns used as feature columns. Columns with nominal and numerical 
        data are allowed.""",
        column_filter=is_nominal_numerical,
    )

    target_column = knext.ColumnParameter(
        "Target column",
        """Selection of column used as the target column. Only columns with numerical data
        are allowed.""",
        column_filter=is_numerical,
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the features",
        """Define whether missing values in the input data should be skipped or whether the 
        node execution should fail on missing values.""",
        MissingValueHandling.SkipRow.name,
        MissingValueHandling,
    )
