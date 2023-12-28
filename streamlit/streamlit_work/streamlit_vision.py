#!python3.11
from IPython.display import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

import json
import requests
import faiss
import numpy as np
import streamlit as st
from moviepy.editor import *
import tempfile
import time
from dotenv import load_dotenv

load_dotenv()

# 処理時間計測
start = time.time()


# set env
VISION_ENDPOINT = os.getenv('VISION_ENDPOINT')
VISION_API_KEY = os.getenv('VISION_API_KEY')

# 音楽設定の選択肢
music_options = {
    "アゲ系": "age.mp3",  # 音楽ファイルを適切なものに置き換えてください
    "cafe": "cafe.mp3",   # 音楽ファイルを適切なものに置き換えてください
    "chill": "chill.mp3", # 音楽ファイルを適切なものに置き換えてください
    "cool": "cool.mp3"    # 音楽ファイルを適切なものに置き換えてください
}

# 解像度の選択肢
resolution_options = {
    "500": (500, 280),  # 解像度を適切な比に設定してください
    "700": (700, 394),  # 解像度を適切な比に設定してください
    "900": (900, 506)   # 解像度を適切な比に設定してください
}

# embed image
images = []
labels = []
vectors = []
image_file_pass = []
num = 5  # the number of images
endpoint = os.getenv("VISION_ENDPOINT") + "/computervision/retrieval:vectorizeImage?api-version=2023-02-01-preview&modelVersion=latest"
headers = {
    "Content-Type": "application/octet-stream",  # リクエストボディは画像のバイナリデータ
    "Ocp-Apim-Subscription-Key": os.getenv("VISION_API_KEY")
}


def embed_images(images):
    i = 0
    for idx, image in enumerate(images):
        i += 1
        embed_cell_start = time.time()
        with open(image, mode="rb") as f:
            image_bin = f.read()
        # Vectorize Image API を使って画像をベクトル化
        response = requests.post(endpoint, headers=headers, data=image_bin)
        # print(response.json())
        image_vec = np.array(response.json()["vector"], dtype="float32").reshape(1, -1)
        vectors.append(image_vec)
        embed_cell_end = time.time()
        embed_cell_time = embed_cell_end - embed_cell_start
        print('embed image No.%s' %i)
        print('embed cell time %s' %embed_cell_time)
    return vectors


# create index
def create_index(vectors):
    dimension = 1024
    index_flat_l2 = faiss.IndexFlatL2(dimension)

    for vector in vectors:
        index_flat_l2.add(vector)
    return index_flat_l2


# search images
def search_faiss_by_text(query_text, n=2):
    endpoint = os.getenv("VISION_ENDPOINT") + "/computervision/retrieval:vectorizeText?api-version=2023-02-01-preview&modelVersion=latest"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": os.getenv("VISION_API_KEY")
    }
    data = {
        "text": query_text
    }
    # Vectorize Text API を使ってクエリをベクトル化
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    query_vector = np.array(response.json()["vector"], dtype="float32").reshape(1, -1)
    # Faiss 検索
    D, I = index_flat_l2.search(query_vector, n)
    return D, I


def down_load_file(uploaded_files):
    if uploaded_files is not None:
        if uploaded_files:
            # 選択されたファイルの一覧を表示
            st.write("選択されたファイル:")
            for file in uploaded_files:
                st.write(file.name)

            # ダウンロードボタンを各ファイルごとに作成
            for i, file in enumerate(uploaded_files):
                st.download_button(
                    label=f"ファイル No.{i+1}をダウンロード",
                    data=file.getvalue(),
                    key=f"download_button_{i+1}",
                    file_name=file.name
                )


# タイトルを設定
st.title('ショートムービー再生アプリ')
# サイドバーのユーザー入力
# 一旦画像ファイル
uploaded_files = st.sidebar.file_uploader("動画ファイルをアップロードしてください", accept_multiple_files=True,type=["jpg","png","mp4", "mov"])
# 変数
selected_files_range = min([len(uploaded_files), 3])
down_load_file(uploaded_files)

selected_music = st.sidebar.selectbox("音楽設定を選択してください", list(music_options.keys()))
selected_resolution = st.sidebar.selectbox("解像度を選択してください", list(resolution_options.keys()))
purpose_text = st.sidebar.text_input("動画のまとめて欲しい内容を教えてください")

# 全ての入力が完了しているか確認
all_inputs_provided = uploaded_files is not None and selected_music and selected_resolution and purpose_text

# 生成ボタン
generate_button = st.sidebar.button('動画を生成', disabled=not all_inputs_provided)

n=6

# 動画と音楽を結合して再生
if generate_button and all_inputs_provided:
    if len(uploaded_files) > 0:
        for tmp_file in uploaded_files:
            tmp_file_pass = "C:/Users/sobata/Downloads/" + tmp_file.name
            images.append(tmp_file_pass)
        print(images)
        
        # アップロードされたファイルをベクトル化する
        embed_start = time.time()
        vectors = embed_images(images)
        embed_end = time.time()
        index_start = time.time()
        index_flat_l2 = create_index(vectors)
        index_end = time.time()
        search_start = time.time()
        D, I = search_faiss_by_text(purpose_text, n)
        search_end = time.time()
        
        embed_time = embed_end - embed_start
        print("embed time %s" %embed_time)
        index_time = index_end - index_start
        print("index time %s" %index_time)
        search_time = search_end - search_start
        print("search time %s" %search_time)

        # テキストに合ったベクトル化された画像を選択
        selected_files = []
        for j in range(selected_files_range):
            selected_files.append(images[I[0][j]])
        print(selected_files)
        st.write(purpose_text)

        with tempfile.TemporaryDirectory() as tmpdirname:
            clips = []
            for image_file in selected_files:
                # 一時ファイルを作成し、アップロードされたファイルの内容を書き込む
                temp_image_path = os.path.join(tmpdirname, image_file)
                # ImageClip を作成
                clip = ImageClip(temp_image_path, duration=3)
                clips.append(clip)

            # 画像クリップを結合
            final_clip = concatenate_videoclips(clips, method="compose")

            # 音楽ファイルのパスを取得
            music_file_path = music_options[selected_music]

            # 音楽クリップを作成
            audio_clip = AudioFileClip(music_file_path)

            # 音楽の長さを動画の長さに合わせる
            audio_clip = audio_clip.subclip(0, final_clip.duration)

            # 音楽を動画に設定
            final_clip = final_clip.set_audio(audio_clip)

            with st.spinner('Wait for it...'):
                # 動画を一時ファイルに書き出す
                temp_video_file = os.path.join(tmpdirname, 'temp_video.mp4')
                final_clip.write_videofile(temp_video_file, fps=24)
                time.sleep(2)
            st.success('Done!')

            st.video(temp_video_file)


else:
    if generate_button:
        st.sidebar.warning('すべての入力を完了してください。')

# 処理時間計測
end = time.time()
time_diff = end - start
print(time_diff)

