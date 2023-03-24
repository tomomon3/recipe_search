# 必要なモジュールのインポート
import torch
from animal import transform, Net # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
#---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.preprocessing import image
#---

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    # Renderの時
    #net.load_state_dict(torch.load('./dog_cat.pt', map_location=torch.device('cpu')))
    # ローカルの時
    net.load_state_dict(torch.load('./src/dog_cat.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

    #---
    #モデルの読み込み
    model = tf.keras.applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling="avg",
    classes=1000
    )

    # 処理本文
    #img_path='/content/PekingDuck.jpeg'
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = resnet.preprocess_input(x)
    batch_tensor = tf.expand_dims(x, axis=0)
    result = model.predict(batch_tensor)
    result=np.reshape(result ,-1)
    results, sims = search(result, features, 9)
    return results



#　推論したラベルから犬か猫かを返す関数
def getName(label):
    if label==0:
        return '猫'
    elif label==1:
        return '犬'

# --計算用関数　ここから
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_top_n_indexes(array, num):
    idx = np.argpartition(array, -num)[-num:]
    return idx[np.argsort(array[idx])][::-1]


def search(query_vector, features, num):
    sims = []
    for vector in features:
        sim = cos_sim(query_vector, vector) # ①
        sims.append(sim)
    sims = np.array(sims)
    indexes = get_top_n_indexes(sims, num) # ②
    return indexes, sims[indexes] # ③
# --計算用関数　ここまで



# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            animalName_ = getName(pred)
            return render_template('result.html', animalName=animalName_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)