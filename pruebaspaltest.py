from utils_data import read_dataset
from KNN import KNN
from PIL import Image
import numpy as np

train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = \
    read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

knn = KNN(train_imgs, train_class_labels)

# Preguntas 15 y 16: primera imagen del test set
first_test = test_imgs[0:1]

pred_k5 = knn.predict(first_test, k=5)
print(f"P15 - Primera imagen test, K=5: {pred_k5[0]}")

pred_k2 = knn.predict(first_test, k=2)
print(f"P16 - Primera imagen test, K=2: {pred_k2[0]}")

# Preguntas 17, 18, 19: necesito saber qué imagen te muestran en el enunciado
# ¿Puedes subir o describir la imagen de las preguntas 17, 18 y 19?