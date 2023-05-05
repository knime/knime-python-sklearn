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

import logging #test
import knime.extension as knext
import sklearn_ext
import utils
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as lof
LOGGER = logging.getLogger(__name__)

@knext.parameter_group(label="Input")
class ClassificationInputSettings:

    feature_columns = knext.MultiColumnParameter(
        "Feature columns",
        "Selection of columns used as feature columns. Columns can be integers or doubles.",
        column_filter=utils.is_numerical,
    )
    
    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the features",
        "Define whether missing values in the input data should be skipped or whether the node execution should fail on missing values.",
        utils.MissingValueHandling.SkipRow.name,
        utils.MissingValueHandling,
    )
 

@knext.parameter_group(label="Local Outlier Factor Settings")
class LOFAlgorithmSettings:
    n_neighbors= knext.IntParameter(
        "n_neighbors",
        "Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, all samples will be used.",
        default_value= 20,
        min_value=0
        )
    
    double_param = knext.DoubleParameter(
        "Contamination",
        "Express a belief about what percentage of the data are likely to be anomalous. It then ranks all the data points by their outlier score, and flags the highest scoring ones as anomalous",
        default_value = 0.1,
        min_value=0.0,
        max_value=0.5)
    
    boolean_param = knext.BoolParameter(
        "Novelty",
        "By default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). Set novelty to True if you want to use LocalOutlierFactor for novelty detection. In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; and note that the results obtained this way may differ from the standard LOF results.",
         default_value=True
         )

@knext.node(
    name="Local Outlier Factor",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons\wrench.png",
    category= sklearn_ext.local_outlier_factor_category
)


@knext.input_table(
    name="Input Data",
    description="Training data set (numerical)"
)

@knext.output_port(
    name="Trained model",
    description="Trained outlier detection",
    port_type=utils.classification_model_port_type
)

class LOFLearner(knext.PythonNode):
    """
    This node trains a Local Outlier Factor analysis
    based on the scikit python library.
    For reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
   
    """   
    general_settings = ClassificationInputSettings()
    algorithm_settings = LOFAlgorithmSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_table: knext.Schema,
    ):
        return self._create_spec(input_table, class_probability_schema= None)
    
    def _create_spec(
        self,
        input_table: knext.Schema,
        class_probability_schema: knext.Schema,
    ):

                # Check if feature column(s) have been specified
        if not self.general_settings.feature_columns:
            raise knext.InvalidParametersError(
                """Feature column(s) have not been specified."""
            )

        # Create schema from feature columns
        feature_schema = knext.Schema.from_columns(
            [c for c in input_table if c.name in self.general_settings.feature_columns]
        )

        #Create schema for target column
        target_schema = knext.Schema.from_columns("")

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")



        return utils.ClassificationModelObjectSpec(
            feature_schema,target_schema, class_probability_schema
        )


 
    def execute(self, execution_context: knext.ExecutionContext, input_table: knext.Table
                ):
        #Convert input table to Panda DataFrame
        dfX= input_table.to_pandas()
        
        # Ignore rows with  if "Ignore" option is selected
        # or fail execution if "Fail" is selected and there are
        missing_value_handling_setting = utils.MissingValueHandling[
            self.general_settings.missing_value_handling
        ]
        dfX = utils.handle_missing_values(dfX, missing_value_handling_setting)


         # Filter feature and target columns
        dfx = dfX.filter(items=self.general_settings.feature_columns)
        dfy = None

         # Encode feature columns with one-hot encoder
        dfx_encoded, one_hot_encoder = utils.encode_train_feature_columns(dfx)

         # Encode target column labels between 0..(n_classes-1)
        dfy_encoded, label_encoder = utils.encode_target_column(
            dfy,
            knext.Schema.from_columns(""),
            #self.general_settings.target_column,
        )


        clf = lof(n_neighbors=self.algorithm_settings.n_neighbors, novelty=self.algorithm_settings.boolean_param, contamination=self.algorithm_settings.double_param)

         # Fit LoF model to data
        model=clf.fit(
            dfx_encoded,
            dfy_encoded,
            )

        class_probability_schema = None
        

        return utils.ClassificationModelObject(
            self._create_spec(input_table.schema,class_probability_schema),
            model,
            label_encoder,
            one_hot_encoder,
            missing_value_handling_setting,
        )