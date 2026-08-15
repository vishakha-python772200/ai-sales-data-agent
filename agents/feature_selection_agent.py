"""
agents/feature_selection_agent.py
-------------------------------------
Selects the most relevant features before training:
 - Drops near-zero-variance columns
 - Drops columns highly correlated with each other (redundant, keeps one)
 - Ranks remaining features by correlation with the target (regression)
   or mutual information (classification)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression # this features removed for  zero / near variance remove,
# to completed classfication task and calulate the feature importance ,Regression task sathi feature importance calculate karto

from agents.base_agent import BaseAgent
from utils.file_handler import save_csv, save_json
from utils.validator import validate_dataframe_not_empty, validate_target_column 
import config

CORRELATION_DROP_THRESHOLD = 0.95 # they define threshold 
#Jar don features madhla correlation 0.95 peksha jast asel,
# tar ek redundant feature remove kela jail.
VARIANCE_THRESHOLD = 0.0  # drop columns where all values are identical


class FeatureSelectionAgent(BaseAgent): # createing a new class from baseagents 
    """Selects relevant features and drops redundant/low-value ones."""

    def __init__(self):
        super().__init__(name="FeatureSelectionAgent")

    def run(self, df: pd.DataFrame, target_column: str = None,
            save_path: str = None) -> pd.DataFrame:
        validate_dataframe_not_empty(df, context="FeatureSelectionAgent input")
        target_column = target_column or config.TARGET_COLUMN
        validate_target_column(df, target_column)

        df = df.copy()
        y = df[target_column]
        X = df.drop(columns=[target_column])

        numeric_X = X.select_dtypes(include=[np.number])

        # 1. Drop zero-variance columns
#jya numeric columns madhe sagle values same asta this columns 
# ya column madhe kahi variance nahi  
# this is a standard ml practice

        if not numeric_X.empty:
            selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
            selector.fit(numeric_X)
            kept_cols = numeric_X.columns[selector.get_support()]
            dropped_cols = [c for c in numeric_X.columns if c not in kept_cols]
            if dropped_cols:
                self.logger.info(f"Dropped zero-variance columns: {dropped_cols}")
            numeric_X = numeric_X[kept_cols]

        # 2. Drop highly correlated (redundant) columns 
        #after we removed zero variance columns then the remaning columns hase similarity ya not 
        # we used matric for this 
        # this is used var duplicated compariasone
        # if there theroshold are same so removed one columns from them
        # we used triu bcz ha fakt upper triangel theto 
        #Duplicate comparison hot nahi.
       #Processing fast hote.
       # we used triu bcz jar mateirc madhal comperion 2 nhi same then tyamule triu=1 akch dil to 
        #Logic clean rahato.

        if numeric_X.shape[1] > 1:
            corr_matrix = numeric_X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_DROP_THRESHOLD)]
            if to_drop:
                numeric_X = numeric_X.drop(columns=to_drop)
                self.logger.info(f"Dropped highly correlated columns (>{CORRELATION_DROP_THRESHOLD}): {to_drop}")

        # 3. Rank remaining features by relevance to target
        #urlelya features madhun konta feature target sathi jast useful aahe
        # te calculate karto.
        #
        # Ya process la Feature Ranking mhanatat.

        ranking = {}
        # first we create empty dictonary 
        # features calculated karta error yeu shakto try excepted error lavycha 
        try:
            if config.TASK_TYPE == "classification":
                scores = mutual_info_classif(numeric_X.fillna(0), y, random_state=config.RANDOM_STATE)
            else:
                scores = mutual_info_regression(numeric_X.fillna(0), y, random_state=config.RANDOM_STATE)
            ranking = dict(sorted(zip(numeric_X.columns, scores.tolist()), key=lambda x: x[1], reverse=True))
            self.logger.info(f"Feature importance ranking computed for {len(ranking)} features.")
        except Exception as e:
            self.logger.warning(f"Could not compute feature ranking: {e}")

        # Reassemble final dataframe: selected numeric + non-numeric (already encoded upstream) + target
        non_numeric_X = X.drop(columns=numeric_X.columns.tolist(), errors="ignore")
        final_df = pd.concat([numeric_X, non_numeric_X, y], axis=1)

        save_json(ranking, f"{config.REPORTS_DIR}/feature_importance_ranking.json")

        output_path = save_path or config.PROCESSED_DATA_FILE
        save_csv(final_df, output_path)

        self.logger.info(f"Feature selection complete. Final shape: {final_df.shape}")
        return final_df
