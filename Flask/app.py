from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import joblib, os, cv2, time, warnings
from PIL import Image
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_cl = load_model('./static/model/MJ_분류_Last_best_SGD.h5')

app = Flask(__name__)
# run_with_ngrok(app)
@app.route('/')
def index():
    menu = {'ho': 1, 'cl': 1, 'gr':0}
    return render_template('index.html', menu = menu)

@app.route('/Classification', methods=['GET', 'POST'])
def Classification():
    menu = {'ho': 0, 'cl': 1, 'gr':0}
    if request.method == 'GET':
        return render_template('module/Classification.html', menu=menu)
    else:
        # 이미지 받아서 전처리
        categories =['evee', 'isang', 'jammanbo', 'jiwoo','leeseul',  'pie', 'pikachu', 'squirtle']

        file = request.files['file']
        fname = file.filename
        file.save(os.path.join('static/img/upload/',fname))

        time.sleep(1)
        img = plt.imread(os.path.join('static/img/upload/',fname))
        img = cv2.resize(img, (128, 128))

        edged = cv2.Canny(img, 10, 200)
        edged_img = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        arr_img = np.array(edged_img)
        arr_img = arr_img.reshape(-1, 128, 128, 3)

        pred_img = model_cl.predict(arr_img/255)
        label = pred_img[0].argmax()
        prob = pred_img[0][label]
        poketmon = categories[label]

        result = {'poketmon':poketmon, 'prob':round(prob,2)}
        return render_template('module/Classification_res.html', menu=menu, res=result, img_file='/img/upload/'+fname)


@app.route('/Gray', methods=['GET', 'POST'])
def Gray():
    menu = {'ho': 0, 'cl': 0, 'gr':1}
    if request.method == 'GET':
        pname = request.form.get('pname')
        fname = request.form.get('fname')
        return render_template('spinner.html', menu=menu, pname=pname, fname=fname)
    else:
        # classification_res에서 hidden으로 넘긴 pname과 fanme 받아오기
        pname = request.form.get('pname')
        dict = {'evee':'./static/model/total_line2gray_eevee_30.h5', 'isang':'',
                'jammanbo':'', 'jiwoo':'',
                'leeseul':'',  'pie':'',
                'pikachu':'', 'squirtle':'./static/model/total_line2gray_squirtle50.h5'}
        model_gr = load_model(dict[pname])
        model_gr.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        fname = request.form.get('fname')
        fname = fname.split('/')[3]
        print(pname, fname)
        img = plt.imread("./static/img/upload/"+fname)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        img_edges = []
        edges = cv2.Canny(img, 150, 200)
        img_edges.append(np.expand_dims(edges, axis=-1))
        img_edges = np.array(img_edges) / 127.5 - 1

        img_gray = model_gr.predict(img_edges)
        plt.imsave('static/img/gray/'+fname, img_gray.reshape(128, 128),cmap='gray')

        return render_template('module/Gray_res.html', menu=menu, img_file='/img/gray/'+fname)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
if __name__ == '__main__':
    app.run(debug=True)   #
