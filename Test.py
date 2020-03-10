from face_train import Model
from skimage import io,transform
import json
if __name__ == '__main__':
    model = Model()
    model.load_model(file_path='./model/face.model')
    img = io.imread("D:/python/workspace/Face1/data/lp/3.jpg")
    print(img.shape)
    probability, name_number = model.face_predict(img)
    with open('contrast_table', 'r') as f:
        contrast_table = json.loads(f.read())
    print(probability)
    print(contrast_table[str(name_number)])

