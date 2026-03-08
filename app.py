# --- ブロック 1: ライブラリのインポートと初期設定 ---
import streamlit as st
import cv2
import numpy as np
import csv
import json
import io
from PIL import Image
import os
from google import genai
from google.genai import types
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Streamlitのページ設定
st.set_page_config(page_title="ボウリング解析システム", page_icon="🎳", layout="wide")

st.title("🎳 ボウリング解析システム")

st.markdown("""
**【使い方】**
1. 左側のサイドバーにGeminiのAPIキーを入力します。
2. 下のボタンからボウリングのスコアシート画像をアップロードします。
3. 自動的に解析が始まり、結果が表示されます！
""")

# --- サイドバー：APIキー入力 ---
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_api_key = st.text_input("Gemini APIキーを入力", type="password")
    st.markdown("※APIキーがないと累計スコアのAI読取ができません。")

# --- メイン画面：画像アップロード ---
uploaded_file = st.file_uploader("スコアシートの画像を選択してください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is None:
    st.stop()

if not gemini_api_key:
    st.error("⚠️ 左側のサイドバーにAPIキーを入力してください。")
    st.stop()

client = genai.Client(api_key=gemini_api_key)
fallback_models = [
    'gemini-3.0-pro', 'gemini-2.5-pro', 'gemini-2.0-pro-exp-02-05',
    'gemini-1.5-pro-latest', 'gemini-1.5-pro'
]

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)

if img is None:
    st.error("⚠️ 画像の読み込みに失敗しました。")
    st.stop()

st.info("⚙️ 解析を開始します...")
status_text = st.empty()
    
    
    

# =========================================================
# 📍 【ブロック 2】 AIプロンプトの定義
# =========================================================
prompt = """
あなたはプロのボウリングスコア記録員です。
画像はボウリングのスコアシートから、スコア部分だけを切り取って縦に並べたものです。
あらかじめ画像解析AIによって、各フレームの投球結果（1投目の倒本数、スペア「/」、ストライク「X」など）が赤色で書き込まれています。これをヒントにしてください。
以下の【ルール】に従って、フレームごとの「累計トータルスコア」のみを正確に読み取り、JSON形式で出力してください。

【ルール】
1. 各ゲームの行の下段には、「累計トータルスコア」が書かれています。1F〜10Fまでの10個の累計スコア数字を配列にしてください。
2. 小さな四角の中の投球結果（Xや/など）は不要です。累計スコアの数字のみを読んでください。
3. Markdownの記号(```json)などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット】
{
  "date": "2026/02/28",
  "time": "14:30",
  "lane": "12",
  "games": [
    {
      "game_num": "GAME 1",
      "frame_totals": [20, 47, 56, 86, 115, 135, 155, 185, 205, 225],
      "total": "225"
    }
  ]
}
"""

# =========================================================
# 📍 【ブロック 3】 関数定義1（ピン判定、スコア計算）
# =========================================================
def get_pins_from_crop(crop_img, thresh_val_empty, thresh_val_circle):
    total_pixels = crop_img.shape[0] * crop_img.shape[1]
    if total_pixels == 0: return None, 0.0
    ink_pixels = cv2.countNonZero(crop_img)
    ink_percent = (ink_pixels / total_pixels) * 100
    if ink_percent < thresh_val_empty: return "EMPTY", ink_percent
    elif ink_percent < thresh_val_circle: return "CIRCLE", ink_percent
    else: return "DOUBLE", ink_percent

def calculate_bowling_score(throws):
    def get_val(idx):
        if idx >= len(throws): return 0
        v = str(throws[idx]).upper().replace("R:", "")
        if v == 'X': return 10
        if v == '/': return 10 - get_val(idx - 1)
        if v in ['-', '', 'G']: return 0
        try: return int(v)
        except: return 0

    frame_totals = []
    current_score = 0
    t_idx = 0

    for frame in range(10):
        if frame == 9:
            f_score = get_val(18) + get_val(19) + get_val(20)
            current_score += f_score
            frame_totals.append(current_score)
            break

        v1 = get_val(t_idx)
        if str(throws[t_idx]).upper().replace("R:", "") == 'X':
            bonus = get_val(t_idx + 2)
            if str(throws[t_idx + 2]).upper().replace("R:", "") == 'X' and frame < 8:
                bonus += get_val(t_idx + 4)
            else:
                bonus += get_val(t_idx + 3)
            current_score += 10 + bonus
            t_idx += 2
        else:
            v2 = get_val(t_idx + 1)
            if str(throws[t_idx + 1]).replace("R:", "") == '/':
                bonus = get_val(t_idx + 2)
                current_score += 10 + bonus
            else:
                current_score += v1 + v2
            t_idx += 2

        frame_totals.append(current_score)
    return frame_totals

    

    # =========================================================
    # 📍 【ブロック 5】 メインループ開始と画像前処理
    # =========================================================
    status_text.text("画像の前処理を実行中...")
    target_width = 1200
    scale_img = target_width / img.shape[1]
    target_height = int(img.shape[0] * scale_img)
    img_resized = cv2.resize(img, (target_width, target_height))

    output_img = img_resized.copy()
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    b_channel = img_resized[:, :, 0]
    thresh_ink = cv2.adaptiveThreshold(b_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # =========================================================
    # 📍 【ブロック 6】 水色線（横線）の抽出とゲーム行のグループ化
    # =========================================================
    status_text.text("水色線（ゲーム枠）を検出中...")
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    h_dilate = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)), iterations=1)
    h_contours, _ = cv2.findContours(h_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extension_px = 50
    blue_lines = []

    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > target_width * 0.4 and h < 50:
            [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = vx[0]; vy = vy[0]; x0 = x0[0]; y0 = y0[0]
            x_start = float(x)
            x_end = float(x + w)
            x_start_ext = x_start - extension_px
            x_end_ext = x_end + extension_px
            slope = vy / vx if vx != 0 else 0
            y_start_ext = y0 + slope * (x_start_ext - x0)
            y_end_ext = y0 + slope * (x_end_ext - x0)
            y_center = (y_start_ext + y_end_ext) / 2

            blue_lines.append({
                'y_center': y_center,
                'start': (int(x_start_ext), int(y_start_ext)),
                'end': (int(x_end_ext), int(y_end_ext))
            })

            cv2.line(output_img, (int(x_start_ext), int(y_start_ext)), (int(x_end_ext), int(y_end_ext)), (255, 255, 0), 3)

    if not blue_lines:
        st.error("⚠️ ゲームの行（水色線）が見つかりませんでした。")
        st.stop()

    blue_lines.sort(key=lambda line: line['y_center'])
    raw_groups = []
    current_group = [blue_lines[0]]

    for line in blue_lines[1:]:
        if abs(line['y_center'] - current_group[-1]['y_center']) <= 40:
            current_group.append(line)
        else:
            raw_groups.append(current_group)
            current_group = [line]

# =========================================================
# 📍 【ブロック 4】 関数定義2（座標回転、テキスト描画）
# =========================================================
def get_angled_box_pts(local_x, local_y, w, h, ref_x, ref_y, theta):
    p1 = np.array([local_x, local_y])
    p2 = np.array([local_x + w, local_y])
    p3 = np.array([local_x + w, local_y + h])
    p4 = np.array([local_x, local_y + h])

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    pts = []
    for p in [p1, p2, p3, p4]:
        rot_p = R.dot(p) + np.array([ref_x, ref_y])
        pts.append([int(round(rot_p[0])), int(round(rot_p[1]))])
    return np.array(pts, np.int32).reshape((-1, 1, 2))

def put_rotated_text(img, text, local_x, local_y, ref_x, ref_y, theta, color, scale=0.7, thickness=2):
    if not text: return
    p = np.array([local_x, local_y])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rot_p = R.dot(p) + np.array([ref_x, ref_y])
    cv2.putText(img, text, (int(rot_p[0]), int(rot_p[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

all_games_export_data = []
throw_cols = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47]
target_indices = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46, 48]

COLOR_OPENCV = (255, 0, 0)
COLOR_AI = (0, 0, 220)
COLOR_PERCENT = (50, 50, 50)

# =========================================================
# 📍 【ブロック 5】 メインループ開始と画像前処理
# =========================================================
status_text.text("画像の前処理を実行中...")
target_width = 1200
scale_img = target_width / img.shape[1]
target_height = int(img.shape[0] * scale_img)
img_resized = cv2.resize(img, (target_width, target_height))

output_img = img_resized.copy()
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

b_channel = img_resized[:, :, 0]
thresh_ink = cv2.adaptiveThreshold(b_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

# =========================================================
# 📍 【ブロック 6】 水色線（横線）の抽出とゲーム行のグループ化
# =========================================================
status_text.text("水色線（ゲーム枠）を検出中...")
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
h_dilate = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)), iterations=1)
h_contours, _ = cv2.findContours(h_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

extension_px = 50
blue_lines = []

for cnt in h_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > target_width * 0.4 and h < 50:
        [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = vx[0]; vy = vy[0]; x0 = x0[0]; y0 = y0[0]
        x_start = float(x)
        x_end = float(x + w)
        x_start_ext = x_start - extension_px
        x_end_ext = x_end + extension_px
        slope = vy / vx if vx != 0 else 0
        y_start_ext = y0 + slope * (x_start_ext - x0)
        y_end_ext = y0 + slope * (x_end_ext - x0)
        y_center = (y_start_ext + y_end_ext) / 2

        blue_lines.append({
            'y_center': y_center,
            'start': (int(x_start_ext), int(y_start_ext)),
            'end': (int(x_end_ext), int(y_end_ext))
        })

        cv2.line(output_img, (int(x_start_ext), int(y_start_ext)), (int(x_end_ext), int(y_end_ext)), (255, 255, 0), 3)

if not blue_lines:
    st.error("⚠️ ゲームの行（水色線）が見つかりませんでした。")
    st.stop()

blue_lines.sort(key=lambda line: line['y_center'])
raw_groups = []
current_group = [blue_lines[0]]

for line in blue_lines[1:]:
    if abs(line['y_center'] - current_group[-1]['y_center']) <= 40:
        current_group.append(line)
    else:
        raw_groups.append(current_group)
        current_group = [line]
if current_group:
    raw_groups.append(current_group)

valid_groups = [g for g in raw_groups if len(g) >= 3]
st.info(f"📝 {len(valid_groups)} 個のゲーム行（グループ）を検出しました。")

# =========================================================
# 📍 【ブロック 7】 画像全体の回転補正
# =========================================================
angles = []
group_refs = []
group_lines_rotated_y = []

for group_idx, group in enumerate(valid_groups):
    top_blue_line = group[0]
    min_y = min(line['start'][1] for line in group)
    max_y = max(line['start'][1] for line in group)
    group_top_y = min_y - 10
    group_bottom_y = max_y + 10
    min_x = min(line['start'][0] for line in group)
    max_x = max(line['end'][0] for line in group)

    crop_y1 = max(0, group_top_y)
    crop_y2 = min(img_resized.shape[0], group_bottom_y)
    crop_x1 = max(0, int(min_x))
    crop_x2 = min(img_resized.shape[1], int(max_x))

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1: continue

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_mask = cv2.morphologyEx(thresh[crop_y1:crop_y2, crop_x1:crop_x2], cv2.MORPH_OPEN, v_kernel)
    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_vertical_xs = []
    for v_cnt in v_contours:
        vx, vy, vw, vh = cv2.boundingRect(v_cnt)
        if vh > 20:
            real_x = crop_x1 + vx
            if real_x > img_resized.shape[1] * 0.03:
                valid_vertical_xs.append(real_x)

    if valid_vertical_xs:
        leftmost_x = min(valid_vertical_xs)
        rightmost_x = max(valid_vertical_xs)

        x_start, y_start = top_blue_line['start']
        x_end, y_end = top_blue_line['end']
        line_slope = (y_end - y_start) / (x_end - x_start) if x_end != x_start else 0

        ref1_x = leftmost_x
        ref1_y = int(y_start + line_slope * (ref1_x - x_start))
        ref2_x = rightmost_x
        ref2_y = int(y_start + line_slope * (ref2_x - x_start))

        group_refs.append( ((ref1_x, ref1_y), (ref2_x, ref2_y)) )

        dy = ref2_y - ref1_y
        dx = ref2_x - ref1_x
        theta_group = np.arctan2(dy, dx)
        angles.append(np.degrees(theta_group))

        M_temp = cv2.getRotationMatrix2D((0, 0), np.degrees(theta_group), 1.0)
        rotated_ys = []
        for line in group:
            cx = (line['start'][0] + line['end'][0]) / 2.0
            cy = line['y_center']
            pt = np.array([cx, cy, 1.0])
            rot_pt = np.dot(M_temp, pt)
            rotated_ys.append(rot_pt[1])
        rotated_ys.sort()
        group_lines_rotated_y.append(rotated_ys)

if angles:
    avg_angle = np.mean(angles)
    h, w = output_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)

    output_img = cv2.warpAffine(output_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    thresh_ink_rotated = cv2.warpAffine(thresh_ink, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img_color_rotated = cv2.warpAffine(img_resized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
else:
    thresh_ink_rotated = thresh_ink.copy()
    img_color_rotated = img_resized.copy()
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

img_for_ai = img_color_rotated.copy()


# =========================================================
# 📍 【ブロック 8-1】 ピンク線検出と赤点（基準点）のマーキング
# =========================================================
status_text.text("基準点（赤点）と投球枠を計算中...")
hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([170, 255, 255])
mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

# ★ Y座標調整（-45ピクセル）を適用
y_offset = -45

for group_idx, group in enumerate(valid_groups):
    if len(group_lines_rotated_y[group_idx]) < 2: continue
    
    # ユーザー指定のロジック：1ゲーム目は一番上の青線を基準点から除外（インデックス1と2を使用）
    if group_idx == 0 and len(group_lines_rotated_y[group_idx]) >= 3:
        y1 = group_lines_rotated_y[group_idx][1]
        y2 = group_lines_rotated_y[group_idx][2]
    else:
        y1 = group_lines_rotated_y[group_idx][0]
        y2 = group_lines_rotated_y[group_idx][1]

    ref1, ref2 = group_refs[group_idx]
    x_start = ref1[0]
    x_end = ref2[0]
    line_slope = (ref2[1] - ref1[1]) / (ref2[0] - ref1[0]) if ref2[0] != ref1[0] else 0

    group_pink_pts = []
    group_red_pts = []

    for idx, t_col in enumerate(throw_cols):
        pt_x = x_start + t_col * 22
        pt_y_base = int(y1 + line_slope * (pt_x - x_start))
        
        # ピンク線検出エリアの絞り込み
        search_y1 = max(0, pt_y_base - 30)
        search_y2 = min(img_resized.shape[0], pt_y_base + 30)
        search_x1 = max(0, int(pt_x) - 10)
        search_x2 = min(img_resized.shape[1], int(pt_x) + 10)
        
        pink_area = mask_pink[search_y1:search_y2, search_x1:search_x2]
        pink_y_coords, pink_x_coords = np.where(pink_area > 0)
        
        if len(pink_y_coords) > 0:
            avg_pink_y = int(np.mean(pink_y_coords)) + search_y1
            pt_y = avg_pink_y + y_offset
            group_pink_pts.append((int(pt_x), avg_pink_y))
        else:
            pt_y = pt_y_base + y_offset
            
        group_red_pts.append((int(pt_x), pt_y))
        cv2.circle(output_img, (int(pt_x), pt_y), 5, (0, 0, 255), -1)

# =========================================================
# 📍 【ブロック 9】 各投球のピン判定（スプリット・ガター等）
# =========================================================
    status_text.text(f"ゲーム {group_idx + 1} のピン判定を実行中...")
    throws_data = []
    for idx, (rx, ry) in enumerate(group_red_pts):
        crop_y1 = max(0, ry - 15)
        crop_y2 = min(thresh_ink_rotated.shape[0], ry + 15)
        crop_x1 = max(0, rx - 10)
        crop_x2 = min(thresh_ink_rotated.shape[1], rx + 10)
        
        crop = thresh_ink_rotated[crop_y1:crop_y2, crop_x1:crop_x2]
        pin_result, percent = get_pins_from_crop(crop, 5.0, 30.0)
        
        throws_data.append(pin_result)

# =========================================================
# 📍 【ブロック 10】 スコア計算とデータ整形
# =========================================================
    frame_scores = calculate_bowling_score(throws_data)
    all_games_export_data.append({
        "game_num": f"GAME {group_idx + 1}",
        "frame_totals": frame_scores,
        "total": str(frame_scores[-1]) if frame_scores else "0"
    })

# =========================================================
# 📍 【ブロック 11】 AI（Gemini）への送信と結果表示
# =========================================================
status_text.text("AIによる累計スコア読み取りを実行中...")

_, buffer = cv2.imencode('.png', img_for_ai)
img_for_genai = types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/png')

ai_response = None
for model_name in fallback_models:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, img_for_genai]
        )
        ai_response = response.text
        break
    except Exception as e:
        continue

if not ai_response:
    st.error("⚠️ AIモデルでのスコア読み取りに失敗しました。")
    st.stop()

# JSONの抽出とパース
json_str = ai_response.replace("```json", "").replace("```", "").strip()
try:
    ai_data = json.loads(json_str)
except json.JSONDecodeError:
    st.error("⚠️ AIの応答形式が不正でした。")
    st.write(ai_response)
    st.stop()

status_text.empty()
st.success("✅ 解析が完了しました！")

# 画面への結果表示
st.subheader("📊 読み取り結果")
for game in ai_data.get("games", []):
    st.markdown(f"**{game.get('game_num', 'GAME')}** - トータル: {game.get('total', 'N/A')}")
    st.write(f"各フレーム累計: {game.get('frame_totals', [])}")

# CSVダウンロード機能
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(["Game", "Total Score", "Frame Totals"])
for game in ai_data.get("games", []):
    writer.writerow([game.get("game_num"), game.get("total"), str(game.get("frame_totals"))])

st.download_button(
    label="📥 CSVをダウンロード",
    data=csv_buffer.getvalue(),
    file_name="bowling_scores.csv",
    mime="text/csv"
)

# 解析済み画像の表示
st.subheader("📸 解析画像")
st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
