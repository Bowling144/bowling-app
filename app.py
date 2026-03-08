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

# =========================================================
# 📍 【ブロック 2】 Googleドライブからの画像取得
# =========================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

st.markdown("### 📥 ボウリング画像の読み込み")

# 画像データを保持する仕組み
if "drive_image_bytes" not in st.session_state:
    st.session_state.drive_image_bytes = None

# ドライブから取得するボタン
if st.button("🔄 ドライブから最新の画像を読み込む"):
    with st.spinner("Googleドライブを探索中..."):
        try:
            # ⚠️修正: 改行文字(コントロール文字)が原因でエラーになるのを防ぐ (strict=False)
            creds_json_str = st.secrets["google_credentials"]
            creds_info = json.loads(creds_json_str, strict=False)
            
            # 念のため、秘密鍵の中の改行がGoogle認証用に正しく認識されるよう補正
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            drive_service = build('drive', 'v3', credentials=creds)

            # 共有されたフォルダ内の最新の画像を1枚探す
            results = drive_service.files().list(
                q="mimeType contains 'image/' and trashed=false",
                orderBy="createdTime desc",
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            items = results.get('files', [])

            if not items:
                st.error("⚠️ 画像が見つかりません。共有した Bowling_App フォルダに画像が入っているか確認してください。")
            else:
                latest_file = items[0]
                file_id = latest_file['id']
                
                # 画像のダウンロード実行
                request = drive_service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                
                st.session_state.drive_image_bytes = fh.getvalue()
                st.success(f"✅ 最新の画像「{latest_file['name']}」をセットしました！")
        except Exception as e:
            st.error(f"⚠️ 読み込みエラー: {e}")

# 画像がセットされるまではここで待機
if st.session_state.drive_image_bytes is None:
    st.info("👆 上のボタンを押して、最新のスコアシートを読み込んでください。")
    st.stop()

if not gemini_api_key:
    st.error("⚠️ 左側のサイドバーにAPIキーを入力してください。")
    st.stop()

client = genai.Client(api_key=gemini_api_key)

# 取得した画像をOpenCV形式に変換
image_bytes = np.frombuffer(st.session_state.drive_image_bytes, np.uint8)
img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

if img is None:
    st.error("⚠️ 画像データの変換に失敗しました。")
    st.stop()

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
# 📍 【ブロック 8-1】 先行解析ループ（全枠のピクセル率収集）
# =========================================================
status_text.text("ピンとインクの分布を解析中...")
L0 = 0
scale_0 = 0
pin_positions = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0), (1, 0.5), (1, 1.5), (1, 2.5), (2, 1.0), (2, 2.0), (3, 1.5)]
parsed_games = []

games_data = []
all_global_pin_pcts = []
all_global_light_purple_pcts = []

for group_idx, ((r1x, r1y), (r2x, r2y)) in enumerate(group_refs):
    pt1 = np.array([r1x, r1y, 1.0])
    pt2 = np.array([r2x, r2y, 1.0])
    new_ref1 = np.dot(M, pt1)
    new_ref2 = np.dot(M, pt2)

    cv2.circle(output_img, (int(new_ref1[0]), int(new_ref1[1])), 6, (0, 0, 255), -1)

    dx = new_ref2[0] - new_ref1[0]
    dy = new_ref2[1] - new_ref1[1]

    if group_idx == 0:
        L0 = dx
        scale_0 = L0 / 184.5
        ratio = 1.0
        theta = 0.0
    else:
        ratio = dx / L0
        theta = np.arctan2(dy, dx)

    current_scale = scale_0 * ratio

    offset_x = 13.7 * current_scale
    y_offset = -41
    start_x_base = offset_x
    start_y_base = y_offset + dx * (16.7 / 184.5)

    rotated_ys = group_lines_rotated_y[group_idx]
    y0 = rotated_ys[0]
    y1 = rotated_ys[1] if len(rotated_ys) > 1 else y0 + 10 * current_scale
    y2 = rotated_ys[2] if len(rotated_ys) > 2 else y1 + 10 * current_scale

    local_y0 = y0 - new_ref1[1]
    local_y1 = y1 - new_ref1[1]
    m_y = 2

    py1_local = local_y0 + m_y
    ph_full = (local_y1 - local_y0) - (m_y * 2)

    ph_box = ph_full * 0.7
    py1_local_box = py1_local + (ph_full - ph_box)

    gy_local = start_y_base + 0.3 * current_scale

    box_w = 15.17 * current_scale
    box_h = 17.2 * current_scale
    y_box_w = box_w / 4.0
    y_box_h = box_h / 4.0

    game_info = {
        'new_ref1': new_ref1, 'theta': theta, 'current_scale': current_scale,
        'start_x_base': start_x_base, 'py1_local': py1_local, 'ph': ph_full,
        'gy_local': gy_local, 'box_w': box_w, 'box_h': box_h,
        'y_box_w': y_box_w, 'y_box_h': y_box_h,
        'purple_data': {}, 'light_purple_data': {}, 'pin_data': {}
    }

    for f in range(10):
        px1_light = start_x_base + f * box_w
        pw_light = 6.69 * current_scale
        pts_light = get_angled_box_pts(px1_light, py1_local, pw_light, ph_full, new_ref1[0], new_ref1[1], theta)
        rx, ry, rw, rh = cv2.boundingRect(pts_light)
        m = 2
        if ry+m < ry+rh-m and rx+m < rx+rw-m:
            crop = thresh_ink_rotated[max(0, ry+m):ry+rh-m, max(0, rx+m):rx+rw-m]
            pixels = crop.shape[0] * crop.shape[1]
            pct = (cv2.countNonZero(crop) / pixels * 100) if pixels > 0 else 0
        else: pct = 0
        game_info['light_purple_data'][f] = {'pct': pct, 'pts': pts_light, 'rx': rx, 'ry': ry}
        all_global_light_purple_pcts.append(pct)

        px1_purple = start_x_base + f * box_w + 6.69 * current_scale
        pw_purple = (14.28 - 6.69) * current_scale
        pts_purple = get_angled_box_pts(px1_purple, py1_local_box, pw_purple, ph_box, new_ref1[0], new_ref1[1], theta)
        rx, ry, rw, rh = cv2.boundingRect(pts_purple)
        if ry+m < ry+rh-m and rx+m < rx+rw-m:
            crop = thresh_ink_rotated[max(0, ry+m):ry+rh-m, max(0, rx+m):rx+rw-m]
            pixels = crop.shape[0] * crop.shape[1]
            pct = (cv2.countNonZero(crop) / pixels * 100) if pixels > 0 else 0
        else: pct = 0
        game_info['purple_data'][f] = {'pct': pct, 'pts': pts_purple, 'rx': rx, 'ry': ry}

        if f == 9:
            px1_3 = start_x_base + 9 * box_w + 14.28 * current_scale
            pw_3 = (21.87 - 14.28) * current_scale
            pts_3 = get_angled_box_pts(px1_3, py1_local_box, pw_3, ph_box, new_ref1[0], new_ref1[1], theta)
            rx3, ry3, rw3, rh3 = cv2.boundingRect(pts_3)
            if ry3+m < ry3+rh3-m and rx3+m < rx3+rw3-m:
                crop = thresh_ink_rotated[max(0, ry3+m):ry3+rh3-m, max(0, rx3+m):rx3+rw3-m]
                pixels = crop.shape[0] * crop.shape[1]
                pct = (cv2.countNonZero(crop) / pixels * 100) if pixels > 0 else 0
            else: pct = 0
            game_info['purple_data']['10_3'] = {'pct': pct, 'pts': pts_3, 'rx': rx3, 'ry': ry3}

    for f in range(12):
        gx_local = start_x_base + f * box_w
        for row_idx, col_offset in pin_positions:
            yx1_local = gx_local + col_offset * y_box_w + 1
            yy1_local = gy_local + row_idx * y_box_h + 1
            yw = y_box_w - 2
            yh = y_box_h - 3
            pts_y = get_angled_box_pts(yx1_local, yy1_local, yw, yh, new_ref1[0], new_ref1[1], theta)
            rx, ry, rw, rh = cv2.boundingRect(pts_y)
            crop_y = thresh_ink_rotated[max(0, ry):ry+rh, max(0, rx):rx+rw]
            pixels_y = crop_y.shape[0] * crop_y.shape[1]
            pin_pct = (cv2.countNonZero(crop_y) / pixels_y * 100) if pixels_y > 0 else 0
            game_info['pin_data'][(f, row_idx, col_offset)] = {'pct': pin_pct, 'pts': pts_y}
            all_global_pin_pcts.append(pin_pct)

    games_data.append(game_info)

# =========================================================
# 📍 【ブロック 8-2】 動的閾値の算出と分布図(ヒストグラム)の合成
# =========================================================
dyn_thresh_empty = 20.0
dyn_thresh_pink = 24.0

if all_global_pin_pcts:
    hist, bin_edges = np.histogram(all_global_pin_pcts, bins=100, range=(0, 100))
    peak1_idx = np.argmax(hist[:25])
    peak2_idx = 25 + np.argmax(hist[25:])

    if hist[peak2_idx] > 0 and peak2_idx > peak1_idx + 5:
        between_hist = hist[peak1_idx:peak2_idx+1]
        zero_indices = np.where(between_hist == 0)[0]
        if len(zero_indices) > 0:
            longest_zeros = []
            current_zeros = []
            for i in zero_indices:
                if not current_zeros or i == current_zeros[-1] + 1:
                    current_zeros.append(i)
                else:
                    if len(current_zeros) > len(longest_zeros): longest_zeros = current_zeros
                    current_zeros = [i]
            if len(current_zeros) > len(longest_zeros): longest_zeros = current_zeros
            valley_idx = longest_zeros[int(len(longest_zeros) * 0.6)]
            dyn_thresh_empty = peak1_idx + valley_idx
        else:
            valley_idx = np.argmin(between_hist)
            dyn_thresh_empty = peak1_idx + valley_idx
    else:
        dyn_thresh_empty = np.max(all_global_pin_pcts) + 5.0

dyn_thresh_circle = dyn_thresh_empty + 12.0

valid_1st_throw_pcts = []
for game_info in games_data:
    for f in range(10):
        frame_pins = []
        for row_idx, col_offset in pin_positions:
            pin_pct = game_info['pin_data'][(f, row_idx, col_offset)]['pct']
            if pin_pct >= dyn_thresh_empty:
                frame_pins.append(1)
        if len(frame_pins) > 0:
            valid_1st_throw_pcts.append(game_info['light_purple_data'][f]['pct'])

if valid_1st_throw_pcts:
    dyn_thresh_pink = np.percentile(valid_1st_throw_pcts, 90) + 3.0

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(4.5, 2.25))
ax2 = ax1.twinx()
ax1.hist(all_global_pin_pcts, bins=50, range=(0,100), color='yellow', alpha=0.6, label=f'Pins (Thresh: {dyn_thresh_empty:.1f}%)')
ax2.hist(all_global_light_purple_pcts, bins=50, range=(0,100), color='mediumpurple', alpha=0.6, label=f'1st Throw (Thresh: {dyn_thresh_pink:.1f}%)')
ax1.axvline(dyn_thresh_empty, color='yellow', linestyle='dashed', linewidth=2)
ax1.axvline(dyn_thresh_pink, color='magenta', linestyle='dashed', linewidth=2)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize='small')
ax1.set_title("Pixel Distribution & Auto Thresholds", fontsize='small')
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=100)
buf.seek(0)
graph_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
plt.close(fig)

gh, gw, _ = graph_img.shape
oh, ow, _ = output_img.shape
if oh >= gh and ow >= gw:
    output_img[0:gh, ow-gw:ow] = graph_img

# =========================================================
# 📍 【ブロック 8-3】 全ゲームの判定と描画ループ
# =========================================================
for group_idx, game_info in enumerate(games_data):
    new_ref1 = game_info['new_ref1']
    theta = game_info['theta']
    start_x_base = game_info['start_x_base']
    py1_local = game_info['py1_local']
    current_scale = game_info['current_scale']
    box_w = game_info['box_w']

    pink_inks = {}
    for f in range(10):
        l_data = game_info['light_purple_data'][f]
        cv2.polylines(output_img, [l_data['pts']], isClosed=True, color=(216, 191, 216), thickness=1)
        p_data = game_info['purple_data'][f]
        cv2.polylines(output_img, [p_data['pts']], isClosed=True, color=(255, 105, 180), thickness=2)
        pink_inks[f] = p_data['pct']
        put_rotated_text(output_img, f"{p_data['pct']:.0f}%", p_data['rx'], p_data['ry'] - 5, 0, 0, 0, COLOR_PERCENT, scale=0.4, thickness=1)

    if f == 9:
        p_data_3 = game_info['purple_data']['10_3']
        cv2.polylines(output_img, [p_data_3['pts']], isClosed=True, color=(255, 105, 180), thickness=2)
        pink_inks['10_3'] = p_data_3['pct']
        put_rotated_text(output_img, f"{p_data_3['pct']:.0f}%", p_data_3['rx'], p_data_3['ry'] - 5, 0, 0, 0, COLOR_PERCENT, scale=0.4, thickness=1)

    all_frame_pins = []
    for f in range(12):
        frame_pins = []
        for row_idx, col_offset in pin_positions:
            pin_data = game_info['pin_data'][(f, row_idx, col_offset)]
            pin_pct = pin_data['pct']
            pts_y = pin_data['pts']
            cv2.polylines(output_img, [pts_y], isClosed=True, color=(0, 255, 255), thickness=1)

            if pin_pct < dyn_thresh_empty: result = "EMPTY"
            elif pin_pct < dyn_thresh_circle: result = "CIRCLE"
            else: result = "DOUBLE"

            if row_idx == 0: pin_num = 7 + int(col_offset)
            elif row_idx == 1: pin_num = 4 + int(col_offset - 0.5)
            elif row_idx == 2: pin_num = 2 + int(col_offset - 1.0)
            elif row_idx == 3: pin_num = 1

            if result in ["CIRCLE", "DOUBLE"]:
                frame_pins.append(pin_num)
                p_top_left = tuple(pts_y[0][0])
                p_bottom_right = tuple(pts_y[2][0])
                cv2.line(output_img, p_top_left, p_bottom_right, (0, 255, 255), 2)
        frame_pins.sort()
        all_frame_pins.append(frame_pins)

    temp_throws = [""] * 21
    for f in range(9):
        v1 = 10 - len(all_frame_pins[f])
        str1 = 'X' if v1 == 10 else ('-' if v1 == 0 else str(v1))
        temp_throws[f*2] = str1
        if str1 == 'X':
            temp_throws[f*2+1] = ""
        elif pink_inks[f] >= dyn_thresh_pink:
            temp_throws[f*2+1] = "/"

    p9, p10, p11 = all_frame_pins[9], all_frame_pins[10], all_frame_pins[11]
    v1_10 = 10 - len(p9)
    str1_10 = 'X' if v1_10 == 10 else ('-' if v1_10 == 0 else str(v1_10))
    temp_throws[18] = str1_10

    if str1_10 == 'X':
        v2_10 = 10 - len(p10)
        str2_10 = 'X' if v2_10 == 10 else ('-' if v2_10 == 0 else str(v2_10))
        temp_throws[19] = str2_10
        if str2_10 == 'X':
            v3_10 = 10 - len(p11)
            str3_10 = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
            temp_throws[20] = str3_10
        elif pink_inks['10_3'] >= dyn_thresh_pink:
            temp_throws[20] = "/"
    else:
        if pink_inks[9] >= dyn_thresh_pink: temp_throws[19] = "/"
        v3_10 = 10 - len(p10)
        str3_10 = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
        temp_throws[20] = str3_10

    for f in range(9):
        t1 = temp_throws[f*2]
        if t1: put_rotated_text(img_for_ai, t1, start_x_base + f * box_w + 3 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, (0, 0, 255), scale=0.7, thickness=2)
        t2 = temp_throws[f*2+1]
        if t2: put_rotated_text(img_for_ai, t2, start_x_base + f * box_w + 10 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, (0, 0, 255), scale=0.7, thickness=2)

    f = 9
    t1 = temp_throws[18]
    if t1: put_rotated_text(img_for_ai, t1, start_x_base + f * box_w + 3 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, (0, 0, 255), scale=0.7, thickness=2)
    t2 = temp_throws[19]
    if t2: put_rotated_text(img_for_ai, t2, start_x_base + f * box_w + 10 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, (0, 0, 255), scale=0.7, thickness=2)
    t3 = temp_throws[20]
    if t3: put_rotated_text(img_for_ai, t3, start_x_base + f * box_w + 17 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, (0, 0, 255), scale=0.7, thickness=2)

    parsed_games.append({
        'pink_inks': pink_inks, 'all_frame_pins': all_frame_pins,
        'p9': p9, 'p10': p10, 'p11': p11, 'theta': theta,
        'current_scale': current_scale, 'start_x_base': start_x_base,
        'box_w': box_w, 'py1_local': py1_local, 'new_ref1': new_ref1
    })

# =========================================================
# 📍 【ブロック 9】 AI用のカンペ画像切り抜きとGemini読み取り
# =========================================================
status_text.text("AIによる累計スコア読み取りを実行中...")
score_crops = []
for group_idx, ((r1x, r1y), (r2x, r2y)) in enumerate(group_refs):
    pt1 = np.array([r1x, r1y, 1.0])
    pt2 = np.array([r2x, r2y, 1.0])
    new_ref1 = np.dot(M, pt1)
    new_ref2 = np.dot(M, pt2)

    dx_crop = new_ref2[0] - new_ref1[0]
    scale_crop = dx_crop / 184.2
    base_y_crop = new_ref1[1] - 41 + dx_crop * (16.7 / 184.2)

    crop_y1 = max(0, int(base_y_crop - 15 * scale_crop))
    crop_y2 = min(img_for_ai.shape[0], int(base_y_crop + 20 * scale_crop))
    crop_x1 = max(0, int(new_ref1[0] - 10))
    crop_x2 = min(img_for_ai.shape[1], int(new_ref2[0] + 10))

    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
        score_crops.append(img_for_ai[crop_y1:crop_y2, crop_x1:crop_x2])

if score_crops:
    max_w = max(crop.shape[1] for crop in score_crops)
    padded_crops = []
    for crop in score_crops:
        pad_w = max_w - crop.shape[1]
        padded = cv2.copyMakeBorder(crop, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded_crops.append(padded)
    stacked_scores = cv2.vconcat(padded_crops)
    img_pil = Image.fromarray(cv2.cvtColor(stacked_scores, cv2.COLOR_BGR2RGB))
else:
    img_pil = Image.fromarray(cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB))

gemini_data = {"date": "", "time": "", "lane": "", "games": []}
success = False

for attempt_model in fallback_models:
    try:
        response = client.models.generate_content(
            model=attempt_model,
            contents=[prompt, img_pil],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.split('\n')
            raw_text = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else raw_text
        gemini_data = json.loads(raw_text)
        success = True
        break
    except Exception as e:
        continue

if not success:
    st.error("⚠️ AIモデルでのスコア読み取りに失敗しました。")
    st.stop()

# =========================================================
# 📍 【ブロック 10】 後行ループ（AI結果の統合と最終出力）
# =========================================================
status_text.text("解析結果を統合しています...")

if isinstance(gemini_data, list):
    gemini_data = {"date": "", "time": "", "lane": "", "games": gemini_data}
elif not isinstance(gemini_data, dict):
    gemini_data = {"date": "", "time": "", "lane": "", "games": []}

for group_idx, parsed_data in enumerate(parsed_games):
    pink_inks = parsed_data['pink_inks']
    all_frame_pins = parsed_data['all_frame_pins']
    p9 = parsed_data['p9']
    p10 = parsed_data['p10']
    p11 = parsed_data['p11']
    theta = parsed_data['theta']
    current_scale = parsed_data['current_scale']
    start_x_base = parsed_data['start_x_base']
    box_w = parsed_data['box_w']
    py1_local = parsed_data['py1_local']
    new_ref1 = parsed_data['new_ref1']

    row_data = [""] * 50
    row_data[0] = str(gemini_data.get("date") or "").replace("-", "/")
    row_data[1] = str(gemini_data.get("time") or "")
    row_data[2] = str(gemini_data.get("lane") or "")

    games_list = gemini_data.get("games") or []
    g_info = games_list[group_idx] if group_idx < len(games_list) else {}

    ai_frame_totals = g_info.get("frame_totals") or []
    if not isinstance(ai_frame_totals, list): ai_frame_totals = []
    while len(ai_frame_totals) < 10: ai_frame_totals.append(0)

    ai_total = g_info.get("total") or ""
    row_data[3] = g_info.get("game_num") or f"GAME {group_idx+1}"
    row_data[49] = str(ai_total)

    for f in range(9): row_data[target_indices[f]] = ",".join(map(str, all_frame_pins[f]))
    row_data[target_indices[9]] = ",".join(map(str, p9))

    if len(p9) == 0:
        row_data[target_indices[10]] = ",".join(map(str, p10))
        row_data[target_indices[11]] = ",".join(map(str, p11))
    else:
        row_data[target_indices[10]] = ""
        row_data[target_indices[11]] = ",".join(map(str, p10))

    final_throws = [""] * 21
    throw_colors = [COLOR_OPENCV] * 21

    for f in range(9):
        v1 = 10 - len(all_frame_pins[f])
        str1 = 'X' if v1 == 10 else ('-' if v1 == 0 else str(v1))
        final_throws[f*2] = str1
        throw_colors[f*2] = COLOR_OPENCV

        if str1 == 'X':
            final_throws[f*2+1] = ""
        else:
            if pink_inks[f] >= dyn_thresh_pink:
                final_throws[f*2+1] = "R:/"
                throw_colors[f*2+1] = COLOR_OPENCV
            else:
                curr_total = int(ai_frame_totals[f]) if str(ai_frame_totals[f]).isdigit() else 0
                prev_total = int(ai_frame_totals[f-1]) if f > 0 and str(ai_frame_totals[f-1]).isdigit() else 0
                diff = curr_total - prev_total

                if diff >= 10:
                    final_throws[f*2+1] = "R:/"
                    throw_colors[f*2+1] = COLOR_AI
                else:
                    v2 = diff - v1
                    if v2 < 0: v2 = 0
                    if v2 + v1 > 9: v2 = 9 - v1
                    final_throws[f*2+1] = "R:-" if v2 == 0 else f"R:{v2}"
                    throw_colors[f*2+1] = COLOR_AI

    curr_total_10 = int(ai_frame_totals[9]) if str(ai_frame_totals[9]).isdigit() else 0
    prev_total_10 = int(ai_frame_totals[8]) if str(ai_frame_totals[8]).isdigit() else 0
    diff_10 = curr_total_10 - prev_total_10

    v1_10 = 10 - len(p9)
    str1_10 = 'X' if v1_10 == 10 else ('-' if v1_10 == 0 else str(v1_10))
    final_throws[18] = str1_10
    throw_colors[18] = COLOR_OPENCV

    if str1_10 == 'X':
        v2_10 = 10 - len(p10)
        str2_10 = 'X' if v2_10 == 10 else ('-' if v2_10 == 0 else str(v2_10))
        final_throws[19] = str2_10
        throw_colors[19] = COLOR_OPENCV

        if str2_10 == 'X':
            v3_10 = 10 - len(p11)
            str3_10 = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
            final_throws[20] = str3_10
            throw_colors[20] = COLOR_OPENCV
        else:
            if pink_inks['10_3'] >= dyn_thresh_pink:
                final_throws[20] = "R:/"
                throw_colors[20] = COLOR_OPENCV
            else:
                if (diff_10 - 10) >= 10:
                    final_throws[20] = "R:/"
                    throw_colors[20] = COLOR_AI
                else:
                    v3_10 = diff_10 - 10 - v2_10
                    if v3_10 < 0: v3_10 = 0
                    if v3_10 + v2_10 > 9: v3_10 = 9 - v2_10
                    final_throws[20] = "R:-" if v3_10 == 0 else f"R:{v3_10}"
                    throw_colors[20] = COLOR_AI
    else:
        if pink_inks[9] >= dyn_thresh_pink:
            final_throws[19] = "R:/"
            throw_colors[19] = COLOR_OPENCV
            v3_10 = 10 - len(p10)
            str3_10 = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
            final_throws[20] = str3_10
            throw_colors[20] = COLOR_OPENCV
        else:
            if diff_10 >= 10:
                final_throws[19] = "R:/"
                throw_colors[19] = COLOR_AI
                v3_10 = diff_10 - 10
                if v3_10 < 0: v3_10 = 0
                if v3_10 > 10: v3_10 = 10
                str3_10 = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
                final_throws[20] = f"R:{str3_10}" if str3_10 != 'X' else "R:X"
                throw_colors[20] = COLOR_AI
            else:
                v2_10 = diff_10 - v1_10
                if v2_10 < 0: v2_10 = 0
                if v2_10 + v1_10 > 9: v2_10 = 9 - v1_10
                final_throws[19] = "R:-" if v2_10 == 0 else f"R:{v2_10}"
                throw_colors[19] = COLOR_AI
                final_throws[20] = ""

    for t_idx, col_idx in enumerate(throw_cols):
        row_data[col_idx] = final_throws[t_idx]
    all_games_export_data.append(row_data)

    for f in range(9):
        t1 = final_throws[f*2].replace("R:", "")
        put_rotated_text(output_img, t1, start_x_base + f * box_w + 3 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[f*2])
        t2 = final_throws[f*2+1].replace("R:", "")
        put_rotated_text(output_img, t2, start_x_base + f * box_w + 10 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[f*2+1])

    f = 9
    t1 = final_throws[18].replace("R:", "")
    put_rotated_text(output_img, t1, start_x_base + f * box_w + 3 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[18])
    t2 = final_throws[19].replace("R:", "")
    put_rotated_text(output_img, t2, start_x_base + f * box_w + 10 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[19])
    t3 = final_throws[20].replace("R:", "")
    put_rotated_text(output_img, t3, start_x_base + f * box_w + 17 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[20])

    calc_totals = calculate_bowling_score(final_throws)
    ai_tot_int = int(ai_total) if str(ai_total).isdigit() else int(ai_frame_totals[-1]) if ai_frame_totals else 0
    result_text_x = start_x_base + 9 * box_w + 5 * current_scale
    result_text_y = py1_local - 10 * current_scale

    if calc_totals and len(ai_frame_totals) > 0 and calc_totals[-1] == ai_tot_int:
        check_str = f"MATCH ({calc_totals[-1]})"
        check_color = (0, 150, 0)
    else:
        calc_val = calc_totals[-1] if calc_totals else 0
        check_str = f"DIFF! ({calc_val} vs {ai_tot_int})"
        check_color = COLOR_AI

    put_rotated_text(output_img, check_str, result_text_x, result_text_y, new_ref1[0], new_ref1[1], theta, check_color, scale=0.6, thickness=2)

# =========================================================
# 📍 【ブロック 11】 結果画面・登録フォームとCSV保存
# =========================================================
status_text.empty()
st.success("✅ 解析が完了しました！")

st.subheader("📸 解析画像")
st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)

st.subheader("📝 データの確認とマスター登録")

# フォーム開始（チェックボックスを操作しても毎回AIが再実行されるのを防ぎます）
with st.form("register_form"):
    st.markdown("### 👤 プレイヤー選択")
    player_list = ["001_田中一吉", "002_田中佳恵", "003_田中蒼之助", "004_田中柾吉", "005_米田稔", "999_ゲスト"]
    selected_player = st.selectbox("このスコアを誰のデータとして登録しますか？", player_list)
    
    st.markdown("---")
    st.markdown("### ⚙️ 登録対象の選択")
    # 一括チェックボックス
    register_all = st.checkbox("✅ 全てのゲームをマスターに登録する", value=True)
    st.caption("※個別に除外したい場合は、上のチェックを外して下のリストで選択してください。")
    
    st.markdown("---")
    
    game_checkboxes = []
    for group_idx, game in enumerate(gemini_data.get("games", [])):
        game_name = game.get('game_num', f'GAME {group_idx+1}')
        
        col1, col2 = st.columns([1, 3])
        with col1:
            is_checked = st.checkbox(f"登録 ({game_name})", value=True, key=f"check_{group_idx}")
            game_checkboxes.append(is_checked)
        with col2:
            st.markdown(f"**{game_name}** - トータルスコア: **{game.get('total', 'N/A')}**")
            st.write(f"各フレーム累計: {game.get('frame_totals', [])}")
            st.info("※次フェーズで、ここに「スコア・ピンの修正ボタン」が追加されます。")
        st.markdown("---")

    # 確定ボタン
    submit_btn = st.form_submit_button("💾 選択したデータを確定（CSV生成）")

# フォームが送信（確定ボタンが押）された後の処理
if submit_btn:
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    
    export_count = 0
    for group_idx, row in enumerate(all_games_export_data):
        # 一括チェックがONなら全て対象、OFFなら個別チェックのON/OFFに従う
        is_target = True if register_all else game_checkboxes[group_idx]
        
        if is_target:
            export_row = [selected_player] + row
            writer.writerow(export_row)
            export_count += 1
            
    if export_count > 0:
        st.success(f"✅ {selected_player} のデータを {export_count} 件確定しました！下のボタンから保存できます。")
        st.download_button(
            label="📥 確定済みデータをダウンロード",
            data=csv_buffer.getvalue(),
            file_name=f"{selected_player}_bowling.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ 登録対象のゲームが選択されていません。")

