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

fallback_models = [
    'gemini-3.0-pro',
    'gemini-2.5-pro',
    'gemini-2.0-pro-exp-02-05',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro'
]

# =========================================================
# 📍 【ブロック 2】 状態管理とGoogleドライブからの画像取得
# =========================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

# アプリの状態を記憶する「セッションステート」の初期化
if "app_state" not in st.session_state:
    st.session_state.app_state = "init"  # init, processing, done の3段階
if "drive_images" not in st.session_state:
    st.session_state.drive_images = []
if "analyzed_data" not in st.session_state:
    st.session_state.analyzed_data = []

st.markdown("### 📥 ボウリング画像の読み込み")

# ドライブから取得するボタン（ここで最大3枚指定します）
if st.button("🔄 ドライブから最新の画像を読み込む（最大3枚）"):
    with st.spinner("Googleドライブを探索中..."):
        try:
            creds_json_str = st.secrets["google_credentials"]
            creds_info = json.loads(creds_json_str, strict=False)
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            drive_service = build('drive', 'v3', credentials=creds)

            # ⚠️ ここで取得枚数を3枚に変更
            results = drive_service.files().list(
                q="mimeType contains 'image/' and trashed=false",
                orderBy="createdTime desc",
                pageSize=3,
                fields="files(id, name)"
            ).execute()
            items = results.get('files', [])

            if not items:
                st.error("⚠️ 画像が見つかりません。共有した Bowling_App フォルダに画像が入っているか確認してください。")
            else:
                downloaded_images = []
                for item in items:
                    request = drive_service.files().get_media(fileId=item['id'])
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    downloaded_images.append({'name': item['name'], 'bytes': fh.getvalue()})
                
                # 画像データを記憶させて、処理フェーズへ移行
                st.session_state.drive_images = downloaded_images
                st.session_state.app_state = "processing"
                st.rerun() # 画面をリフレッシュ
        except Exception as e:
            st.error(f"⚠️ 読み込みエラー: {e}")

if st.session_state.app_state == "init":
    st.info("👆 上のボタンを押して、最新のスコアシートを読み込んでください。")
    st.stop()

if not gemini_api_key:
    st.error("⚠️ 左側のサイドバーにAPIキーを入力してください。")
    st.stop()

client = genai.Client(api_key=gemini_api_key)

# =========================================================
# 📍 【ブロック 3】 AIプロンプトの定義（複数ゲーム対応版）
# =========================================================
prompt = """
あなたはプロのボウリングスコア記録員です。
画像はボウリングのスコアシートから、スコア部分だけを切り取って縦に並べたものです。
あらかじめ画像解析AIによって、各フレームの投球結果（1投目の倒本数、スペア「/」、ストライク「X」など）が赤色で書き込まれています。これをヒントにしてください。
以下の【ルール】に従って、フレームごとの「累計トータルスコア」のみを正確に読み取り、JSON形式で出力してください。

【ルール】
1. 各ゲームの行の下段には、「累計トータルスコア」が書かれています。1F〜10Fまでの10個の累計スコア数字を配列にしてください。
2. ⚠️重要⚠️ 画像に複数のゲーム（GAME 1, GAME 2...など）が写っている場合は、絶対に省略せず【写っているすべてのゲーム】のデータを配列 "games" に出力してください。
3. Markdownの記号(```json)などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット例】
{
  "date": "2026/02/28",
  "time": "14:30",
  "lane": "12",
  "games": [
    {
      "game_num": "GAME 1",
      "frame_totals": [20, 47, 56, 86, 115, 135, 155, 185, 205, 225],
      "total": "225"
    },
    {
      "game_num": "GAME 2",
      "frame_totals": [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],
      "total": "90"
    }
  ]
}
"""

# =========================================================
# 📍 【ブロック 4】 共通関数・定数定義
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

throw_cols = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47]
target_indices = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46, 48]
COLOR_OPENCV = (255, 0, 0)
COLOR_AI = (0, 0, 220)
COLOR_PERCENT = (50, 50, 50)

# =========================================================
# 📍 【ブロック 5〜10】 解析処理ループ（複数枚を裏側で記憶）
# =========================================================
if st.session_state.app_state == "processing":
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_analyzed_results = []

    total_images = len(st.session_state.drive_images)
    for idx, img_info in enumerate(st.session_state.drive_images):
        status_text.info(f"⚙️ 画像 {idx+1}/{total_images} を解析中... ({img_info['name']})")
        
        img = cv2.imdecode(np.frombuffer(img_info['bytes'], np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue

        target_width = 1200
        scale_img = target_width / img.shape[1]
        target_height = int(img.shape[0] * scale_img)
        img_resized = cv2.resize(img, (target_width, target_height))
        output_img = img_resized.copy()
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        b_channel = img_resized[:, :, 0]
        thresh_ink = cv2.adaptiveThreshold(b_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

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
                slope = vy / vx if vx != 0 else 0
                y_center = y0 + slope * (((x_start + x_end) / 2) - x0)
                blue_lines.append({
                    'y_center': y_center,
                    'start': (int(x_start - extension_px), int(y0 + slope * (x_start - extension_px - x0))),
                    'end': (int(x_end + extension_px), int(y0 + slope * (x_end + extension_px - x0)))
                })

        if not blue_lines: continue

        blue_lines.sort(key=lambda line: line['y_center'])
        raw_groups = []
        current_group = [blue_lines[0]]
        for line in blue_lines[1:]:
            if abs(line['y_center'] - current_group[-1]['y_center']) <= 40:
                current_group.append(line)
            else:
                raw_groups.append(current_group)
                current_group = [line]
        if current_group: raw_groups.append(current_group)
        valid_groups = [g for g in raw_groups if len(g) >= 3]

        angles = []
        group_refs = []
        group_lines_rotated_y = []

        for group in valid_groups:
            top_blue_line = group[0]
            min_y = min(line['start'][1] for line in group)
            max_y = max(line['start'][1] for line in group)
            min_x = min(line['start'][0] for line in group)
            max_x = max(line['end'][0] for line in group)

            crop_y1, crop_y2 = max(0, min_y - 10), min(img_resized.shape[0], max_y + 10)
            crop_x1, crop_x2 = max(0, int(min_x)), min(img_resized.shape[1], int(max_x))
            if crop_y2 <= crop_y1 or crop_x2 <= crop_x1: continue

            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            v_mask = cv2.morphologyEx(thresh[crop_y1:crop_y2, crop_x1:crop_x2], cv2.MORPH_OPEN, v_kernel)
            v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_vertical_xs = [crop_x1 + cv2.boundingRect(vc)[0] for vc in v_contours if cv2.boundingRect(vc)[3] > 20 and (crop_x1 + cv2.boundingRect(vc)[0]) > img_resized.shape[1] * 0.03]
            
            if valid_vertical_xs:
                ref1_x, ref2_x = min(valid_vertical_xs), max(valid_vertical_xs)
                x_start, y_start = top_blue_line['start']
                x_end, y_end = top_blue_line['end']
                line_slope = (y_end - y_start) / (x_end - x_start) if x_end != x_start else 0
                ref1_y = int(y_start + line_slope * (ref1_x - x_start))
                ref2_y = int(y_start + line_slope * (ref2_x - x_start))
                group_refs.append(((ref1_x, ref1_y), (ref2_x, ref2_y)))
                theta_group = np.arctan2(ref2_y - ref1_y, ref2_x - ref1_x)
                angles.append(np.degrees(theta_group))
                
                M_temp = cv2.getRotationMatrix2D((0, 0), np.degrees(theta_group), 1.0)
                rotated_ys = sorted([np.dot(M_temp, np.array([(l['start'][0] + l['end'][0])/2.0, l['y_center'], 1.0]))[1] for l in group])
                group_lines_rotated_y.append(rotated_ys)

        if angles:
            avg_angle = np.mean(angles)
            h, w = output_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), avg_angle, 1.0)
            output_img = cv2.warpAffine(output_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            thresh_ink_rotated = cv2.warpAffine(thresh_ink, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img_color_rotated = cv2.warpAffine(img_resized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        else:
            thresh_ink_rotated = thresh_ink.copy()
            img_color_rotated = img_resized.copy()
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        img_for_ai = img_color_rotated.copy()

        L0, scale_0 = 0, 0
        pin_positions = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0), (1, 0.5), (1, 1.5), (1, 2.5), (2, 1.0), (2, 2.0), (3, 1.5)]
        parsed_games, games_data = [], []
        all_global_pin_pcts, all_global_light_purple_pcts = [], []

        for group_idx, ((r1x, r1y), (r2x, r2y)) in enumerate(group_refs):
            new_ref1 = np.dot(M, np.array([r1x, r1y, 1.0]))
            new_ref2 = np.dot(M, np.array([r2x, r2y, 1.0]))
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
            start_x_base = 13.7 * current_scale
            start_y_base = -41 + dx * (16.7 / 184.5)

            rotated_ys = group_lines_rotated_y[group_idx]
            y0 = rotated_ys[0]
            y1 = rotated_ys[1] if len(rotated_ys) > 1 else y0 + 10 * current_scale
            py1_local = (y0 - new_ref1[1]) + 2
            ph_full = ((y1 - new_ref1[1]) - (y0 - new_ref1[1])) - 4
            ph_box = ph_full * 0.7
            py1_local_box = py1_local + (ph_full - ph_box)

            gy_local = start_y_base + 0.3 * current_scale
            box_w = 15.17 * current_scale
            box_h = 17.2 * current_scale
            y_box_w, y_box_h = box_w / 4.0, box_h / 4.0

            game_info = {
                'new_ref1': new_ref1, 'theta': theta, 'current_scale': current_scale,
                'start_x_base': start_x_base, 'py1_local': py1_local, 'ph': ph_full,
                'gy_local': gy_local, 'box_w': box_w, 'box_h': box_h,
                'y_box_w': y_box_w, 'y_box_h': y_box_h,
                'purple_data': {}, 'light_purple_data': {}, 'pin_data': {}
            }

            for f in range(10):
                px1_light = start_x_base + f * box_w
                pts_light = get_angled_box_pts(px1_light, py1_local, 6.69 * current_scale, ph_full, new_ref1[0], new_ref1[1], theta)
                rx, ry, rw, rh = cv2.boundingRect(pts_light)
                crop = thresh_ink_rotated[max(0, ry+2):ry+rh-2, max(0, rx+2):rx+rw-2]
                pct = (cv2.countNonZero(crop) / (crop.shape[0]*crop.shape[1]) * 100) if crop.shape[0]*crop.shape[1] > 0 else 0
                game_info['light_purple_data'][f] = {'pct': pct, 'pts': pts_light, 'rx': rx, 'ry': ry}
                all_global_light_purple_pcts.append(pct)

                px1_purple = start_x_base + f * box_w + 6.69 * current_scale
                pts_purple = get_angled_box_pts(px1_purple, py1_local_box, (14.28-6.69) * current_scale, ph_box, new_ref1[0], new_ref1[1], theta)
                rx, ry, rw, rh = cv2.boundingRect(pts_purple)
                crop = thresh_ink_rotated[max(0, ry+2):ry+rh-2, max(0, rx+2):rx+rw-2]
                pct = (cv2.countNonZero(crop) / (crop.shape[0]*crop.shape[1]) * 100) if crop.shape[0]*crop.shape[1] > 0 else 0
                game_info['purple_data'][f] = {'pct': pct, 'pts': pts_purple, 'rx': rx, 'ry': ry}

                if f == 9:
                    px1_3 = start_x_base + 9 * box_w + 14.28 * current_scale
                    pts_3 = get_angled_box_pts(px1_3, py1_local_box, (21.87-14.28) * current_scale, ph_box, new_ref1[0], new_ref1[1], theta)
                    rx3, ry3, rw3, rh3 = cv2.boundingRect(pts_3)
                    crop = thresh_ink_rotated[max(0, ry3+2):ry3+rh3-2, max(0, rx3+2):rx3+rw3-2]
                    pct = (cv2.countNonZero(crop) / (crop.shape[0]*crop.shape[1]) * 100) if crop.shape[0]*crop.shape[1] > 0 else 0
                    game_info['purple_data']['10_3'] = {'pct': pct, 'pts': pts_3, 'rx': rx3, 'ry': ry3}

            for f in range(12):
                gx_local = start_x_base + f * box_w
                for row_idx, col_offset in pin_positions:
                    yx1_local = gx_local + col_offset * y_box_w + 1
                    yy1_local = gy_local + row_idx * y_box_h + 1
                    pts_y = get_angled_box_pts(yx1_local, yy1_local, y_box_w-2, y_box_h-3, new_ref1[0], new_ref1[1], theta)
                    rx, ry, rw, rh = cv2.boundingRect(pts_y)
                    crop_y = thresh_ink_rotated[max(0, ry):ry+rh, max(0, rx):rx+rw]
                    pin_pct = (cv2.countNonZero(crop_y) / (crop_y.shape[0]*crop_y.shape[1]) * 100) if crop_y.shape[0]*crop_y.shape[1] > 0 else 0
                    game_info['pin_data'][(f, row_idx, col_offset)] = {'pct': pin_pct, 'pts': pts_y}
                    all_global_pin_pcts.append(pin_pct)
            games_data.append(game_info)

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
                    dyn_thresh_empty = peak1_idx + zero_indices[int(len(zero_indices) * 0.6)]
                else: dyn_thresh_empty = peak1_idx + np.argmin(between_hist)
            else: dyn_thresh_empty = np.max(all_global_pin_pcts) + 5.0

        dyn_thresh_circle = dyn_thresh_empty + 12.0
        valid_1st_throw_pcts = [game_info['light_purple_data'][f]['pct'] for game_info in games_data for f in range(10) if sum(1 for row_idx, col_offset in pin_positions if game_info['pin_data'][(f, row_idx, col_offset)]['pct'] >= dyn_thresh_empty) > 0]
        if valid_1st_throw_pcts: dyn_thresh_pink = np.percentile(valid_1st_throw_pcts, 90) + 3.0

        for group_idx, game_info in enumerate(games_data):
            pink_inks = {f: game_info['purple_data'][f]['pct'] for f in range(10)}
            pink_inks['10_3'] = game_info['purple_data']['10_3']['pct']
            
            all_frame_pins = []
            for f in range(12):
                frame_pins = []
                for row_idx, col_offset in pin_positions:
                    if game_info['pin_data'][(f, row_idx, col_offset)]['pct'] >= dyn_thresh_circle:
                        pin_num = {0: 7+int(col_offset), 1: 4+int(col_offset-0.5), 2: 2+int(col_offset-1.0), 3: 1}[row_idx]
                        frame_pins.append(pin_num)
                all_frame_pins.append(sorted(frame_pins))

            temp_throws = [""] * 21
            for f in range(9):
                v1 = 10 - len(all_frame_pins[f])
                str1 = 'X' if v1 == 10 else ('-' if v1 == 0 else str(v1))
                temp_throws[f*2] = str1
                if str1 == 'X': temp_throws[f*2+1] = ""
                elif pink_inks[f] >= dyn_thresh_pink: temp_throws[f*2+1] = "/"

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
                    temp_throws[20] = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
                elif pink_inks['10_3'] >= dyn_thresh_pink: temp_throws[20] = "/"
            else:
                if pink_inks[9] >= dyn_thresh_pink: temp_throws[19] = "/"
                v3_10 = 10 - len(p10)
                temp_throws[20] = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))

            for f in range(9):
                if temp_throws[f*2]: put_rotated_text(img_for_ai, temp_throws[f*2], game_info['start_x_base'] + f * game_info['box_w'] + 3 * game_info['current_scale'], game_info['py1_local'] - 2 * game_info['current_scale'], game_info['new_ref1'][0], game_info['new_ref1'][1], game_info['theta'], (0, 0, 255), scale=0.7, thickness=2)
                if temp_throws[f*2+1]: put_rotated_text(img_for_ai, temp_throws[f*2+1], game_info['start_x_base'] + f * game_info['box_w'] + 10 * game_info['current_scale'], game_info['py1_local'] - 2 * game_info['current_scale'], game_info['new_ref1'][0], game_info['new_ref1'][1], game_info['theta'], (0, 0, 255), scale=0.7, thickness=2)
            
            parsed_games.append({
                'pink_inks': pink_inks, 'all_frame_pins': all_frame_pins, 'p9': p9, 'p10': p10, 'p11': p11,
                'theta': game_info['theta'], 'current_scale': game_info['current_scale'],
                'start_x_base': game_info['start_x_base'], 'box_w': game_info['box_w'],
                'py1_local': game_info['py1_local'], 'new_ref1': game_info['new_ref1']
            })

        score_crops = []
        for group_idx, ((r1x, r1y), (r2x, r2y)) in enumerate(group_refs):
            dx_crop = r2x - r1x
            scale_crop = dx_crop / 184.2
            base_y_crop = r1y - 41 + dx_crop * (16.7 / 184.2)
            crop_y1, crop_y2 = max(0, int(base_y_crop - 15 * scale_crop)), min(img_for_ai.shape[0], int(base_y_crop + 20 * scale_crop))
            crop_x1, crop_x2 = max(0, int(r1x - 10)), min(img_for_ai.shape[1], int(r2x + 10))
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                score_crops.append(img_for_ai[crop_y1:crop_y2, crop_x1:crop_x2])

        if score_crops:
            max_w = max(crop.shape[1] for crop in score_crops)
            padded_crops = [cv2.copyMakeBorder(crop, 0, 0, 0, max_w - crop.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for crop in score_crops]
            img_pil = Image.fromarray(cv2.cvtColor(cv2.vconcat(padded_crops), cv2.COLOR_BGR2RGB))
        else: img_pil = Image.fromarray(cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB))

        gemini_data = {"date": "", "time": "", "lane": "", "games": []}
        for attempt_model in fallback_models:
            try:
                response = client.models.generate_content(
                    model=attempt_model, contents=[prompt, img_pil],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                raw_text = response.text.strip()
                if raw_text.startswith("
http://googleusercontent.com/immersive_entry_chip/0

上書き保存後、アプリをリロードしてボタンを押していただくと、**「フォルダに入れた画像を最大3枚まで一気に読み込み」「1〜3枚目まですべてのゲーム行を解析」「連続したGAME番号（GAME 1〜GAME 9など）として1つのフォームにまとめる」**という動作をするようになります。

これにてフェーズ3の下準備（大掛かりな改修）は終了です！無事に3枚分通して表示されるかご確認のうえ、成功したら次はいよいよ本命の「フレームごとのスコア修正画面」の作成に進みましょう！

