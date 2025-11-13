import torch
from model import IrisNetworkModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd

le = LabelEncoder()

mapping = {0: ('setosa', "images/setosa.jpg"), 1: ('versicolor', "images/versicolor.jpg"), 2: ('virginica',"images/virginica.png")}
# mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}



model_2 = IrisNetworkModel()
model_2.load_state_dict(torch.load('iris_model.pth'))

def predict_iris(sepal_length, sepal_width, petal_length,petal_width):
    model_2.eval()
    input_tensor = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model_2(input_tensor)
        _ , predicted = torch.max(outputs, 1)

    flower_name,flower_filepath= mapping[predicted.item()]
    return flower_name,flower_filepath


if __name__ == "__main__":
    flower_name = predict_iris(sepal_length=5.1,sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    print(flower_name)