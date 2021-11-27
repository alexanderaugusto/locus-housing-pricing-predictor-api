from sklearn.pipeline import Pipeline
import pandas as pd
import urllib.request
import pickle
from modules.CustomTransforms import CombineNewAttributes

class ProcessDataAndPredict():
    def __init__(self, data):
        self.data = data

        # Baixando o modelo compilado do github e salvando no ambiente
        urllib.request.urlretrieve("https://raw.githubusercontent.com/VanessaSwerts/locus-data/master/src/model/assets/final_model.sav", "assets/final_model.sav")
        self.model = pickle.load(open('assets/final_model.sav', 'rb'))

    def process(self):
        # Importando o dataset modificado
        df_deploy = pd.read_csv("https://raw.githubusercontent.com/VanessaSwerts/locus-data/master/src/datasets/deploy.csv")

        features = ["type", "area", 'bedroom', 'bathroom', "garage", "latitude","longitude", "bathroom_per_bedroom"]
        X = df_deploy[features]

        self.model = Pipeline(
            steps=[
                ('combine_attrs', CombineNewAttributes(X)),
                ('dtc', self.model),
            ]
        )
    
    def predict(self):
        prediction = self.model.predict(self.data)
        return prediction.tolist()