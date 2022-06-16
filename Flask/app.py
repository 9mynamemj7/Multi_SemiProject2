from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os, cv2, time, warnings, math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_cl = load_model('./static/model/MJ_분류_Last_best_SGD.h5')

app = Flask(__name__)
run_with_ngrok(app)
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
        pname = request.args.get('pname')
        fname = request.args.get('fname')
        # args.
        print('Gray_GET:',pname,'|', fname)
        return render_template('module/Gray.html', menu=menu, pname=pname, fname=fname)
    else:
        # classification_res에서 hidden으로 넘긴 pname과 fanme 받아오기
        pname = request.form.get('pname')[:-1]
        fname = request.form.get('fname')
        fname = fname.split('/')[3]
        print('Gray_POST:',pname, '|',fname)

        dict = {'evee':'./static/model/total_line2gray_eevee_30.h5', 'isang':'./static/model/total_line2gray_isang50.h5',
                'jammanbo':'./static/model/total_line2gray_jammanbo50.h5', 'jiwoo':'./static/model/best_line2gray_jiwoo4.h5',
                'leeseul':'./static/model/total_line2gray_leeseul49.h5',  'pie':'./static/model/total_line2gray_pie_50.h5',
                'pikachu':'./static/model/total_line2gray_pikachu30.h5', 'squirtle':'./static/model/total_line2gray_squirtle50.h5'}
        model_gr = load_model(dict[pname])
        model_gr.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        img = plt.imread("./static/img/upload/"+fname)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        img_edges = []
        edges = cv2.Canny(img, 150, 200)
        img_edges.append(np.expand_dims(edges, axis=-1))
        img_edges = np.array(img_edges) / 127.5 - 1

        img_gray = model_gr.predict(img_edges)
        plt.imsave('static/img/gray/'+fname, img_gray.reshape(128, 128),cmap='gray')

        return render_template('module/Gray_res.html', menu=menu, img_file='/img/gray/'+fname, pname=pname)

@app.route('/Coloring', methods=['GET', 'POST'])
def Coloring():
    menu = {'ho': 0, 'cl': 0, 'gr':1}
    if request.method == 'GET':
        pname = request.args.get('pname')
        fname = request.args.get('fname')
        # args.
        print('Color_GET:',pname,'|', fname)
        return render_template('module/Coloring.html', menu=menu, pname=pname, fname=fname)
    else:
        # classification_res에서 hidden으로 넘긴 pname과 fanme 받아오기
        pname = request.form.get('pname')[:-1]
        fname = request.form.get('fname')
        fname = fname.split('/')[-2]
        print('Color_POST:',pname,'|', fname)

        dict = {'evee':'./static/model/Color/evee_coloring_best_epochs3000_val_accuracy.h5',
                'isang':'./static/model/Color/isang_coloring_best_epochs3000_val_loss.h5',
                'jammanbo':'./static/model/Color/jammanbo_coloring_best_epochs3000_val_loss.h5',
                'jiwoo':'./static/model/Color/jiwoo_coloring_best_model.h5',
                'leeseul':'./static/model/Color/leeseul_coloring_best_epochs3000_val_accuracy.h5',
                'pie':'./static/model/Color/pie_coloring_best_epochs3000_val_accuracy.h5',
                'pikachu':'./static/model/Color/pikachu_coloring_best_epochs3000_val_accuracy.h5',
                'squirtle':'./static/model/Color/squirtle_coloring_best_epochs3000_val_accuracy.h5'}

        ## 함수
        # RGB -> LAB 이미지로 변환
        def rgb2lab(rgb):
            assert rgb.dtype == 'uint8'
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)

        # LAB -> RGB 이미지로 변환
        def lab2rgb(lab):
            assert lab.dtype == 'uint8'
            return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        # data_list 내 RGB 이미지를 LAB array로 반환
        def get_lab_from_data_list(img_path):
            x_lab = []
            rgb = img_to_array(load_img(img_path, target_size=(img_size, img_size))).astype(
                np.uint8)  # RGB 이미지를 128*128 사이즈로 불러와 array로 변환
            lab = rgb2lab(rgb)  # RGB array -> LAB array로 변환
            x_lab.append(lab)  # LAB array를 리스트에 저장
            return np.stack(x_lab)  # 리스트를 array로 변환

        # 위 셀의 3가지 함수를 활용한 최종 함수
        # data_list 내 RGB 이미지를 generator(L array, AB array)로 반환
        # generator 1개 당 이미지 batch_size개
        # 위 셀의 3가지 함수를 활용한 최종 함수
        # data_list 내 RGB 이미지를 generator(L array, AB array)로 반환
        # generator 1개 당 이미지 batch_size개
        def generator_with_preprocessing(img_path, batch_size, shuffle=False):
            while True:
                # batch_list = img_path[i:i + batch_size]          # batch_size만큼 이미지 가져오기
                batch_lab = get_lab_from_data_list(img_path)  # 이미지 LAB array로 변환
                batch_l = batch_lab[:, :, :, 0:1]  # L 값만 추출 -> (batch_size, 128, 128, 1)
                batch_ab = batch_lab[:, :, :, 1:]  # AB 값만 추출 -> (batch_size, 128, 128, 2)
                yield (batch_l, batch_ab)

        batch_size = 30; img_size=128
        GAN_gen = generator_with_preprocessing('./static/img/gray/'+fname, batch_size)
        GAN_steps = math.ceil(1 / batch_size)
        model_color = load_model(dict[pname])
        GAN_preds = model_color.predict_generator(GAN_gen, steps=GAN_steps)  # GAN AB

        # GAN L 추출
        x_GAN = []
        for i, (l, _) in enumerate(GAN_gen):
            x_GAN.append(l)
            if i == (GAN_steps - 1):
                break
        x_GAN = np.vstack(x_GAN)

        # GAN L과 GAN AB를 결합하여 LAB array 생성 -> RGB로 변환
        GAN_preds_lab = np.concatenate((x_GAN, GAN_preds), 3).astype(np.uint8)
        GAN_preds_rgb = []
        preds_rgb = lab2rgb(GAN_preds_lab[0, :, :, :])
        GAN_preds_rgb.append(preds_rgb)
        GAN_preds_rgb = np.stack(GAN_preds_rgb)

        # GAN 이미지 & 채색 결과 함께 출력
        fname = fname.split('/')[-1]
        plt.imsave('./static/img/Coloring/'+fname, GAN_preds_rgb[0])

        return render_template('module/Coloring_res.html', menu=menu, img_file='/img/Coloring/'+fname, pname=pname)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
if __name__ == '__main__':
    app.run()   #debug=True
