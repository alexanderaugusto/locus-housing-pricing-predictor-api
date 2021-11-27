from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

class CombineNewAttributes(BaseEstimator, TransformerMixin): 
  def __init__(self, dataset):
    self.dataset = dataset

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    newRow = {
      'type': X[0][0],
      'area': X[0][1],
      'bedroom': X[0][2],
      'bathroom': X[0][3],
      'garage': X[0][4],
      'latitude': X[0][5],
      'longitude': X[0][6],
      'bathroom_per_bedroom': X[0][3] / X[0][2]
    } 
    df_result = self.dataset.append(newRow, ignore_index=True)

    numeric_values = df_result.select_dtypes(include=[np.number])

    numeric_attrs = list(numeric_values)
    categorical_attrs = ["type"]

    pipeline_numeric = Pipeline([
      ('imputer', SimpleImputer(strategy="median")),
      ('std_scaler', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
      ("num", pipeline_numeric, numeric_attrs),
      ("cat", OneHotEncoder(), categorical_attrs),
    ])

    result = full_pipeline.fit_transform(df_result)

    return [result[-1]]