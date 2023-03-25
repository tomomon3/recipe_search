#　参考
#Python × Flask × Tensorflow.Keras 猫の品種を予測するWebアプリ
# https://qiita.com/3BMKATYWKA/items/52d1c838eb34133042a3


from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
from image_process import examine_recipe
from datetime import datetime
import os
import cv2
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)

# モデルの読み込み
print('model loading...')
#model = tf.keras.applications.ResNet152(
model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling="avg",
    classes=1000
)
print('-----------------')
# fearturesの読み込み
print('features loading....')
features = np.load('np_save50.npy')
print('features len : ',len(features))
# fearturesの画像indexの読み込み
print('df_idx loading....')
df_idx = pd.read_csv('df_idx_all.csv', index_col=0)
print('df_idx len : ',len(df_idx))

@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        print('------POST-------')
        # アプロードされたファイルをいったん保存する
        file = request.files['file']
        # 画像書き込み用バッファを確保
        buf = BytesIO()
        print('save buffer......')
        # 画像データをバッファに書き込む
        file.save(buf)
        print('Resizing.....')
        input_img = load_img(buf, target_size=(224, 224))
        print('-----------------')

        # 猫の種別を調べる関数の実行
        results ,sims= examine_recipe(input_img, model, features)
        print("results")
        print(results)
        print('-----------------')

        no1_foodImageUrl = df_idx['foodImageUrl'][results[0]]
        no2_foodImageUrl = df_idx['foodImageUrl'][results[1]]
        no3_foodImageUrl = df_idx['foodImageUrl'][results[2]]
        no4_foodImageUrl = df_idx['foodImageUrl'][results[3]]
        no5_foodImageUrl = df_idx['foodImageUrl'][results[4]]
        no6_foodImageUrl = df_idx['foodImageUrl'][results[5]]
        no7_foodImageUrl = df_idx['foodImageUrl'][results[6]]
        no8_foodImageUrl = df_idx['foodImageUrl'][results[7]]
        no9_foodImageUrl = df_idx['foodImageUrl'][results[8]]

        no1_recipeUrl = df_idx['recipeUrl'][results[0]]
        no2_recipeUrl = df_idx['recipeUrl'][results[1]]
        no3_recipeUrl = df_idx['recipeUrl'][results[2]]
        no4_recipeUrl = df_idx['recipeUrl'][results[3]]
        no5_recipeUrl = df_idx['recipeUrl'][results[4]]
        no6_recipeUrl = df_idx['recipeUrl'][results[5]]
        no7_recipeUrl = df_idx['recipeUrl'][results[6]]
        no8_recipeUrl = df_idx['recipeUrl'][results[7]]
        no9_recipeUrl = df_idx['recipeUrl'][results[8]]

        no1_recipeTitle = df_idx['recipeTitle'][results[0]]
        no2_recipeTitle = df_idx['recipeTitle'][results[1]]
        no3_recipeTitle = df_idx['recipeTitle'][results[2]]
        no4_recipeTitle = df_idx['recipeTitle'][results[3]]
        no5_recipeTitle = df_idx['recipeTitle'][results[4]]
        no6_recipeTitle = df_idx['recipeTitle'][results[5]]
        no7_recipeTitle = df_idx['recipeTitle'][results[6]]
        no8_recipeTitle = df_idx['recipeTitle'][results[7]]
        no9_recipeTitle = df_idx['recipeTitle'][results[8]]

        # バイナリデータをbase64でエンコード
        # utf-8でデコード
        input_img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8") 
        # 付帯情報を付与する
        input_img_b64data = "data:image/png;base64,{}".format(input_img_b64str) 

        return render_template("index.html", input_img_b64data=input_img_b64data, 
        no1_foodImageUrl=no1_foodImageUrl, no2_foodImageUrl=no2_foodImageUrl, no3_foodImageUrl=no3_foodImageUrl,
        no4_foodImageUrl=no4_foodImageUrl, no5_foodImageUrl=no5_foodImageUrl, no6_foodImageUrl=no6_foodImageUrl,
        no7_foodImageUrl=no7_foodImageUrl, no8_foodImageUrl=no8_foodImageUrl, no9_foodImageUrl=no9_foodImageUrl,
        no1_recipeUrl=no1_recipeUrl, no2_recipeUrl=no2_recipeUrl, no3_recipeUrl=no3_recipeUrl,
        no4_recipeUrl=no4_recipeUrl, no5_recipeUrl=no5_recipeUrl, no6_recipeUrl=no6_recipeUrl,
        no7_recipeUrl=no7_recipeUrl, no8_recipeUrl=no8_recipeUrl, no9_recipeUrl=no9_recipeUrl,
        no1_recipeTitle=no1_recipeTitle, no2_recipeTitle=no2_recipeTitle, no3_recipeTitle=no3_recipeTitle,
        no4_recipeTitle=no4_recipeTitle, no5_recipeTitle=no5_recipeTitle, no6_recipeTitle=no6_recipeTitle,
        no7_recipeTitle=no7_recipeTitle, no8_recipeTitle=no8_recipeTitle, no9_recipeTitle=no9_recipeTitle)
    
    elif request.method == "GET":
        print('----GET-----')
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host="0.0.0.0")
