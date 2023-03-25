from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import resnet
import tensorflow as tf


def examine_cat_breeds(image, model, cat_list):
    # 行列に変換
    img_array = img_to_array(image)
    # 3dim->4dim
    img_dims = np.expand_dims(img_array, axis=0)
    # Predict class（preds：クラスごとの確率が格納された12×1行列）
    preds = model.predict(preprocess_input(img_dims))
    preds_reshape = preds.reshape(-1,preds.shape[0])
    # cat_list(リスト)を12×1行列に変換
    cat_array = np.array(cat_list).reshape(len(cat_list),-1)
    # 確率高い順にソートする
    preds_sort = preds_reshape[np.argsort(preds_reshape[:, 0])[::-1]]
    # 確率の降順に合わせて猫の順番も変える
    cat_sort = cat_array[np.argsort(preds_reshape[:, 0])[::-1]]
    # preds_reshape と cat_arrayを結合
    set_result = np.concatenate([cat_sort, preds_sort], 1)
    return set_result[0:3, :]


def examine_recipe(image, model, features):
    x = img_to_array(image)
    x = resnet.preprocess_input(x)
    batch_tensor = tf.expand_dims(x, axis=0)
    result = model.predict(batch_tensor)
    result=np.reshape(result ,-1)
    results, sims = search(result, features, 9)
    return results, sims

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