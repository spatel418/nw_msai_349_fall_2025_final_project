import pandas as pd
from logger_config import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import (
    LinearRegression,
    BayesianRidge,
    ElasticNet,
)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from services.regression.regression_file_validator import (
    RegressionFileValidator,
)


class RegressionCorrelation:
    def __init__(self, input_training_df, target):
        self.input_training = input_training_df
        self.target = target

    def correlation_coef(self):
        return self.input_training.corr(method="spearman")

    def calculate_correlated_predictors(self, spearman):
        correlation_dict = {}
        spearman = spearman.drop([self.target], axis=1).fillna(0)
        for column in spearman.columns:
            column_correlations = spearman[column].drop([column, self.target])
            correlation_categories = column_correlations.apply(
                self.categorize_correlations
            )
            correlation_dict[column] = {}
            correlation_summary = pd.DataFrame(
                {
                    "Correlation Value": column_correlations,
                    "Category": correlation_categories,
                }
            )
            for index, row in correlation_summary.iterrows():
                # Use the index (column name) as the key, and the tuple (correlation value, category) as the value
                correlation_dict[column][index] = (
                    abs(row["Correlation Value"]),
                    (
                        "+"
                        if row["Correlation Value"] >= 0.01
                        else "-" if row["Correlation Value"] <= 0.01 else ""
                    ),
                    row["Category"],
                )
        return correlation_dict

    def categorize_correlations(self, corr_value):
        if abs(corr_value) >= 0.7:
            return "Strong Correlation"
        elif abs(corr_value) >= 0.3:
            return "Medium Correlation"
        else:
            return "Low to No Correlation"

    def calculate_target_correlations(self, spearman):
        spearman = spearman[self.target].drop([self.target]).fillna(0)
        correlation_dict = {}
        for index, row in spearman.items():
            correlation_dict[index] = (
                abs(row),
                "+" if row >= 0.01 else "-" if row <= 0.01 else "",
            )
        correlation_dict = dict(
            sorted(correlation_dict.items(), key=lambda item: item[1][0], reverse=True)
        )
        return correlation_dict

    def run_correlation(self):
        spearman = self.correlation_coef()
        predictor_correlations = self.calculate_correlated_predictors(spearman)
        target_correlations = self.calculate_target_correlations(spearman)
        return predictor_correlations, target_correlations


class RegressionPredictorImportance:
    def __init__(self, input_training_df, target, model_choice):
        self.input_training = input_training_df
        self.target = target
        self.model_choice = model_choice

    def populate_model_dict(self):
        model_dict = {}
        model_dict["linear"] = LinearRegression
        model_dict["bayesian"] = BayesianRidge
        model_dict["elasticnet"] = ElasticNet
        model_dict["xgboost"] = xgb.XGBRegressor
        model_dict["randomforest"] = RandomForestRegressor
        if isinstance(self.model_choice, list):
            model_dict = {
                k: model_dict[k] for k in self.model_choice if k in model_dict
            }
        return model_dict

    def run_predictor_importance(self):
        model_dict = self.populate_model_dict()
        predictor_importance_dict = {}
        X = self.input_training.drop(self.target, axis=1)
        y = self.input_training[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        for k, v in model_dict.items():
            model = v()
            model.fit(X_train, y_train)
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=3)
            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": perm_importance.importances_mean}
            )
            max_importance = importance_df[
                "Importance"
            ].max()  # Get the maximum importance
            importance_df["Normalized Importance"] = (
                importance_df["Importance"] / max_importance
                if max_importance != 0
                else 0
            )

            # Convert the dataframe into a dictionary and add to the result
            importance_dict = (
                importance_df.fillna(0)
                .set_index("Feature")["Normalized Importance"]
                .to_dict()
            )
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
            )
            predictor_importance_dict[k] = importance_dict
        return predictor_importance_dict


# Builder pattern for clean method chaining interface--very testable
class RegressionAnalysisBuilder:
    def __init__(self, input_training_df, target, model_choice, correlation):
        self.input_training = input_training_df
        self.regression_file_validator = RegressionFileValidator(
            input_training_df.columns.tolist()
        )
        self.target = target
        self.model_choice = model_choice
        self.correlation = correlation

    def normalize(self, df):
        scaler = MinMaxScaler()
        df = df.select_dtypes(include=["number"]).fillna(0).drop(columns=[self.target])
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = pd.concat([df, self.input_training[[self.target]]], axis=1)
        return df

    def analyze(self):
        validation_status = self.regression_file_validator.run_validation(
            self.input_training
        )
        if not isinstance(validation_status, pd.DataFrame):
            return validation_status
        input_df = self.normalize(self.input_training)
        if self.correlation:
            regression_correlation = RegressionCorrelation(input_df, self.target)
            predictor_correlations, target_correlations = (
                regression_correlation.run_correlation()
            )
            return predictor_correlations, target_correlations, None

        else:
            regression_predictor_importance = RegressionPredictorImportance(
                input_df, self.target, self.model_choice
            )
            predictor_importance = (
                regression_predictor_importance.run_predictor_importance()
            )
            return None, None, predictor_importance
