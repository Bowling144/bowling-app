import streamlit as st
import cv2
import numpy as np
import io
import csv
import json
import time
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- ページ設定 ---
st.set_page_config(page_title="ボウリング解析", layout="wide")
st.markdown("""
    <style>
    /* expanderの下の余白を消す */
    div[data-testid="stExpander"] {
        margin-bottom: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='color: turquoise; text-align: center;'> 🎳Eagle ROLLERS🎳</h1>", unsafe_allow_html=True)

# --- サイドバー：APIキー入力 ---
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_api_key = st.text_input("Gemini APIキーを入力", type="password")
    st.markdown("※APIキーがないと累計スコアのAI読取ができません。")


# ⚠️ AIモデル設定：Flash版を除外し、Proモデルに限定（存在しないモデル名によるエラーを防止）
fallback_models = [
    'gemini-3.0-pro',
    'gemini-2.5-pro',
    'gemini-2.0-pro-exp-02-05'
]


# =========================================================
# 📍 【ブロック 2】 状態管理とGoogleドライブからの画像取得
# =========================================================
if "app_state" not in st.session_state:
    st.session_state.app_state = "init"
if "raw_images_data" not in st.session_state:
    st.session_state.raw_images_data = []
if "analyzed_results" not in st.session_state:
    st.session_state.analyzed_results = None

st.markdown("<h3 style='text-align: center;'>  ☟　☟　☟　☟　☟　☟　☟　</h3>", unsafe_allow_html=True)

# --- 変更点：カラムでレイアウトを3分割し、中央のカラムにボタンを配置する ---
col1, col2, col3 = st.columns([1, 2, 1]) # [1, 2, 1] は左右の余白と中央の幅の比率です。
with col2:
    fetch_button = st.button("🔄 スコアシート取込（MAX３枚）🔄", use_container_width=True)

if fetch_button:
    with st.spinner("Googleドライブを探索中..."):
        try:
            creds_json_str = st.secrets["google_credentials"]
            creds_info = json.loads(creds_json_str, strict=False)
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=scopes
            )
            drive_service = build('drive', 'v3', credentials=creds)

            results = drive_service.files().list(
                q="mimeType contains 'image/' and trashed=false",
                orderBy="createdTime desc",
                pageSize=3,
                fields="files(id, name)"
            ).execute()
            items = results.get('files', [])

            if not items:
                st.error("⚠️ 画像が見つかりません。")
            else:
                fetched_images = []
                for item in items:
                    file_id = item['id']
                    request = drive_service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    fetched_images.append({"name": item['name'], "bytes": fh.getvalue()})

                st.session_state.raw_images_data = fetched_images
                st.session_state.analyzed_results = None
                st.success(f"✅ {len(fetched_images)}枚の画像をセットしました！")
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ 読み込みエラー: {e}")

if not st.session_state.raw_images_data:
    st.info("　")
    st.stop()

if not gemini_api_key:
    st.error("⚠️ 左側のサイドバーにAPIキーを入力してください。")
    st.stop()

client = genai.Client(api_key=gemini_api_key)
status_text = st.empty()

# =========================================================
# 📍 【AIプロンプトの定義】（既存のプロンプト部分と差し替え）
# =========================================================
prompt_metadata = """
画像はボウリングのスコアシートの全体写真です。
この画像から「日付」「最初のゲーム数」「開始時刻」「終了時刻」を探し出し、以下のJSON形式で出力してください。

【ルール】
1. 日付: 中央上部にある黒い文字。「YY/MM/DD」の形式で "date" に出力。
2. 最初のゲーム数: 一番上のゲームのスコア欄の左端に記載。フレームという文字の下GAMEの下に改行されて数字を記載。GAME1, GAME7, GAME13, GAME19, GAME25のいずれか。「1」などの数値のみを "start_game_num" に出力。
3. 開始時刻: 1枚のスコアシートの日付の右下に記載。1ゲーム目の開始時刻と終了時刻が左右に並んでいて、その左側の時刻が開始時刻。"HH:MM" 形式で "start_time" に出力。見つからなければ "時刻不明" にする。
4. 終了時刻: 1枚のスコアシートの一番最後のゲームの9フレーム目のスコア欄の上部に記載。開始時刻と終了時刻が左右に並んでいて、その右側の時刻が終了時刻。"HH:MM" 形式で "end_time" に出力。見つからなければ "時刻不明" にする。
5. Markdownの記号などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット例】
{
  "date": "26/02/07",
  "start_game_num": 1,
  "start_time": "14:12",
  "end_time": "15:30"
}
"""

prompt_score = """
あなたはプロのボウリングスコア記録員です。
画像はボウリングのスコアシートから、スコア部分だけを切り取って縦に並べたものです。
あらかじめ画像解析AIによって、各フレームの投球結果が赤色で書き込まれています。これをヒントにしてください。
以下の【ルール】に従って、フレームごとの「累計トータルスコア」を正確に読み取り、JSON形式で出力してください。

【ルール】
1. 各ゲームの行の下段には、「累計トータルスコア」が書かれています。1F〜10Fまでの10個の累計スコア数字を配列にしてください。
2. 複数のゲームが写っている場合は、写っているすべてのゲームのデータを配列 "games" に出力してください。
3. 日付や時間は読まなくて構いません。スコアの数字だけに集中してください。
4. Markdownの記号などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット例】
{
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
# 📍 【ブロック 4】 共通関数・定数定義（安定版の50列構成）
# =========================================================
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
pin_positions = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0), (1, 0.5), (1, 1.5), (1, 2.5), (2, 1.0), (2, 2.0), (3, 1.5)]

COLOR_OPENCV = (255, 0, 0)
COLOR_AI = (0, 0, 220)
COLOR_PERCENT = (50, 50, 50)

# =========================================================
# 📍 【ブロック 5〜10】 メイン解析ループ
# =========================================================
if st.session_state.analyzed_results is None:
    analyzed_results = []

    for img_idx, img_data in enumerate(st.session_state.raw_images_data):
        file_name = img_data["name"]
        status_text.info(f"⚙️ 画像 {img_idx+1}/{len(st.session_state.raw_images_data)} 枚目 ({file_name}) を解析中...")

        image_bytes = np.frombuffer(img_data["bytes"], np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.warning(f"⚠️ {file_name} の画像変換に失敗しました。スキップします。")
            continue

        all_games_export_data = []
        blue_lines = []
        raw_groups = []
        angles = []
        group_refs = []
        group_lines_rotated_y = []
        parsed_games = []
        games_data = []
        all_global_pin_pcts = []
        all_global_light_purple_pcts = []
        score_crops = []

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
            st.warning(f"⚠️ {file_name}: ゲームの行（水色線）が見つかりませんでした。スキップします。")
            continue

        blue_lines.sort(key=lambda line: line['y_center'])
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
                    if pin_pct >= dyn_thresh_empty: frame_pins.append(1)
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
            img_pil_scores = Image.fromarray(cv2.cvtColor(stacked_scores, cv2.COLOR_BGR2RGB))
        else:
            img_pil_scores = Image.fromarray(cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB))

        # ⚠️ ここが重要！ else の中には入れず、左にずらして「else」の文字と縦を揃える
        img_pil_full = Image.fromarray(cv2.cvtColor(img_color_rotated, cv2.COLOR_BGR2RGB))

        # ---------------------------------------------------------
        # 📍 【ブロック 9】 AIによるテキスト読み取り（スコア → 日時）
        # ---------------------------------------------------------
        status_text.info(f"⚙️ 画像 {img_idx+1}: AIがスコアを読み取り中...")
        time.sleep(3)


        ai_score_data = {"lane": "", "games": []}
        success_score = False
        last_error = ""
        used_model = "FAILED"
        for attempt_model in fallback_models:
            try:
                response = client.models.generate_content(
                    model=attempt_model,
                    contents=[prompt_score, img_pil_scores],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split('\n')
                    raw_text = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else raw_text
                ai_score_data = json.loads(raw_text)
                success_score = True
                used_model = attempt_model.upper()
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(2)
                continue

        if not success_score:
            st.warning(f"⚠️ {file_name}: AIのスコア読み取りに失敗しました。理由: {last_error}")

        status_text.info(f"⚙️ 画像 {img_idx+1}: AIが日付・時刻・ゲーム数を取得中...")
        time.sleep(3)

        ai_meta_data = {"date": "日付不明", "start_time": "時刻不明", "end_time": "時刻不明", "start_game_num": 1}
        for attempt_model in fallback_models:
            try:
                response = client.models.generate_content(
                    model=attempt_model,
                    contents=[prompt_metadata, img_pil_full],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split('\n')
                    raw_text = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else raw_text
                ai_meta_data = json.loads(raw_text)
                break
            except Exception:
                time.sleep(2)
                continue

        # ---------------------------------------------------------
        # 📍 【ブロック 10】 解析結果の統合とデータ整形
        # ---------------------------------------------------------
        cv2.putText(output_img, f"AI Ver: {used_model}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        if isinstance(ai_score_data, list):
            ai_score_data = {"games": ai_score_data}
        elif not isinstance(ai_score_data, dict):
            ai_score_data = {"games": []}

        global_date = str(ai_meta_data.get("date", "日付不明")).replace("-", "/")
        start_time = str(ai_meta_data.get("start_time", "時刻不明"))
        end_time = str(ai_meta_data.get("end_time", "時刻不明"))
        try:
            base_game_num = int(ai_meta_data.get("start_game_num", 1))
        except:
            base_game_num = 1

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

            row_data = [""] * 51
            row_data[0] = global_date
            row_data[1] = start_time
            row_data[2] = end_time 
            row_data[3] = str(ai_score_data.get("lane") or "")

            games_list = ai_score_data.get("games") or []
            g_info = games_list[group_idx] if group_idx < len(games_list) else {}

            ai_frame_totals = g_info.get("frame_totals") or []
            if not isinstance(ai_frame_totals, list): ai_frame_totals = []
            while len(ai_frame_totals) < 10: ai_frame_totals.append(0)

            ai_total = g_info.get("total") or ""
            current_game_num = base_game_num + group_idx
            row_data[4] = f"G{current_game_num}"
            row_data[50] = str(ai_total)

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

# --- 【ここから下を差し替え】 ---
            f = 9
            t1 = final_throws[18].replace("R:", "")
            put_rotated_text(output_img, t1, start_x_base + f * box_w + 3 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[18])
            t2 = final_throws[19].replace("R:", "")
            put_rotated_text(output_img, t2, start_x_base + f * box_w + 10 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[19])
            t3 = final_throws[20].replace("R:", "")
            put_rotated_text(output_img, t3, start_x_base + f * box_w + 17 * current_scale, py1_local - 2 * current_scale, new_ref1[0], new_ref1[1], theta, throw_colors[20])

            # 【修正点1】関数に渡す前に "R:" を削除し、エラー時はアプリを落とさずスキップする
            clean_throws = [str(t).replace("R:", "") for t in final_throws]
            try:
                calc_totals = calculate_bowling_score(clean_throws)
            except Exception:
                calc_totals = []

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

        analyzed_results.append({
            "file_name": file_name,
            "output_img": output_img,
            "all_games_export_data": all_games_export_data,
            "meta_data": ai_meta_data
        })

    st.session_state.analyzed_results = analyzed_results
    status_text.empty()
    st.rerun()

# =========================================================
# 📍 【ブロック 11】 結果画面・SPS自動登録フォーム
# =========================================================
import gspread

if st.session_state.analyzed_results:
    st.success("✅ 全ての画像の解析が完了しました！")

    # 注：修正ボタンを機能させるため、st.form による囲みを外して再構築しています
    player_list = ["001_田中一吉", "002_田中佳恵", "003_田中蒼之助", "004_田中柾吉", "005_米田稔", "999_ゲスト"]
    selected_player = st.selectbox("👤プレイヤー選択👤", player_list, label_visibility="collapsed")

    st.markdown("---")
    register_all = st.checkbox("全てのゲームをマスターに登録する", value=True)
    st.markdown("---")

    game_checkboxes = []

    for img_idx, res in enumerate(st.session_state.analyzed_results):
        st.markdown(f"#### 📄 画像 {img_idx+1}: {res['file_name']}")
        st.image(cv2.cvtColor(res['output_img'], cv2.COLOR_BGR2RGB), use_container_width=True)

        for local_idx, row in enumerate(res['all_games_export_data']):
            game_name = row[4]
            date_str = row[0]
            start_time = row[1]
            end_time = row[2]
            ai_total_str = row[50] if row[50] else "_"
            
            throws_for_calc = [str(row[i]).replace("R:", "") for i in throw_cols]
            try:
                calc_totals = calculate_bowling_score(throws_for_calc)
                calc_val = calc_totals[-1] if calc_totals else 0
            except Exception:
                calc_totals = []
                calc_val = 0

            ai_total_int = int(ai_total_str) if str(ai_total_str).isdigit() else 0
            if calc_totals and calc_val == ai_total_int:
                match_status = "✅計算一致"
            else:
                match_status = "⚠️不一致"

            display_text = f"{date_str}_{start_time}_{end_time}_{game_name}｜トータル:{ai_total_str}_{match_status}"
            is_checked = st.checkbox(display_text, value=True, key=f"check_{res['file_name']}_{local_idx}")

            game_checkboxes.append({
                "is_checked": is_checked,
                "export_row": row,
                "date": date_str,
                "start": start_time,
                "end": end_time
            })

            # --- 🛠️ 修正機能 UI（AI不使用） ---
            with st.expander(f"✏️ {game_name} を手動修正"):
                c_date, c_start, c_end = st.columns(3)
                with c_date:
                    new_date = st.text_input("日付", value=row[0], key=f"d_{img_idx}_{local_idx}")
                with c_start:
                    new_start = st.text_input("開始時刻", value=row[1], key=f"s_{img_idx}_{local_idx}")
                with c_end:
                    new_end = st.text_input("終了時刻", value=row[2], key=f"e_{img_idx}_{local_idx}")

                st.markdown("**🎳 投球結果と残ピン位置（1〜10番を選択）**")
                new_throws = []
                new_pins = []

                # 1〜9フレームの入力
                for f in range(9):
                    st.write(f"**{f+1}F**")
                    c1, c2, c3 = st.columns([1, 1, 3])
                    with c1:
                        t1 = st.text_input("1投目", value=str(row[throw_cols[f*2]]).replace("R:", ""), key=f"t1_{img_idx}_{local_idx}_{f}")
                    with c2:
                        t2 = st.text_input("2投目", value=str(row[throw_cols[f*2+1]]).replace("R:", ""), key=f"t2_{img_idx}_{local_idx}_{f}")
                    with c3:
                        curr_pins_str = str(row[target_indices[f]])
                        curr_pins = [int(p) for p in curr_pins_str.split(",")] if curr_pins_str else []
                        p_sel = st.multiselect("残ピン番号", options=list(range(1, 11)), default=curr_pins, key=f"p_{img_idx}_{local_idx}_{f}")
                    
                    new_throws.extend([t1, t2])
                    new_pins.append(",".join(map(str, p_sel)))

                # 10フレームの入力
                st.write("**10F**")
                c10_1, c10_2, c10_3 = st.columns(3)
                with c10_1:
                    t10_1 = st.text_input("1投目", value=str(row[throw_cols[18]]).replace("R:", ""), key=f"t1_{img_idx}_{local_idx}_9")
                    p1_str = str(row[target_indices[9]])
                    p1_def = [int(p) for p in p1_str.split(",")] if p1_str else []
                    p1_sel = st.multiselect("1投目後 残ピン", options=list(range(1, 11)), default=p1_def, key=f"p1_{img_idx}_{local_idx}_9")
                with c10_2:
                    t10_2 = st.text_input("2投目", value=str(row[throw_cols[19]]).replace("R:", ""), key=f"t2_{img_idx}_{local_idx}_9")
                    p2_str = str(row[target_indices[10]])
                    p2_def = [int(p) for p in p2_str.split(",")] if p2_str else []
                    p2_sel = st.multiselect("2投目後 残ピン", options=list(range(1, 11)), default=p2_def, key=f"p2_{img_idx}_{local_idx}_9")
                with c10_3:
                    t10_3 = st.text_input("3投目", value=str(row[throw_cols[20]]).replace("R:", ""), key=f"t3_{img_idx}_{local_idx}_9")
                    p3_str = str(row[target_indices[11]])
                    p3_def = [int(p) for p in p3_str.split(",")] if p3_str else []
                    p3_sel = st.multiselect("3投目後 残ピン", options=list(range(1, 11)), default=p3_def, key=f"p3_{img_idx}_{local_idx}_9")
                
                new_throws.extend([t10_1, t10_2, t10_3])
                new_pins.extend([",".join(map(str, p1_sel)), ",".join(map(str, p2_sel)), ",".join(map(str, p3_sel))])

                # 修正確定・自動計算ボタン
                if st.button("🔄 修正を反映して再計算", key=f"update_{img_idx}_{local_idx}"):
                    row[0] = new_date
                    row[1] = new_start
                    row[2] = new_end
                    
                    for i in range(21):
                        row[throw_cols[i]] = new_throws[i]
                    for i in range(12):
                        row[target_indices[i]] = new_pins[i]
                    
                    # 再計算ロジックを実行し、トータルスコア(row[50])を上書き
                    new_calc_totals = calculate_bowling_score(new_throws)
                    if new_calc_totals:
                        row[50] = str(new_calc_totals[-1])
                    
                    st.rerun()

    st.markdown("<h3 style='text-align: center;'>☟　☟　☟　☟　☟　☟　☟　</h3>", unsafe_allow_html=True)

    # --- ドライブ検索＆SPS自動登録処理 ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("☁️ 選択したプレイヤーのSPSへデータを登録", use_container_width=True, type="primary"):
            with st.spinner("Google Driveを検索し、データを登録中..."):
                try:
                    # 1. 権限とスコープの設定 (書き込み権限を含む)
                    creds_json_str = st.secrets["google_credentials"]
                    creds_info = json.loads(creds_json_str, strict=False)
                    if "private_key" in creds_info:
                        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
                    
                    scopes = [
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ]
                    creds_write = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
                    
                    gc = gspread.authorize(creds_write)
                    drive_service_write = build('drive', 'v3', credentials=creds_write)

                    # 2. 「Players_Data」フォルダを検索
                    query = "name = 'Players_Data' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                    results = drive_service_write.files().list(q=query, fields="files(id, name)").execute()
                    folders = results.get('files', [])
                    if not folders:
                        st.error("エラー: Googleドライブ内に「Players_Data」フォルダが見つかりません。共有設定（編集者権限）を確認してください。")
                        st.stop()
                    playersdata_id = folders[0]['id']

                    # 3. 選択したプレイヤーのフォルダを検索
                    query = f"name = '{selected_player}' and '{playersdata_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                    results = drive_service_write.files().list(q=query, fields="files(id, name)").execute()
                    p_folders = results.get('files', [])
                    if not p_folders:
                        st.error(f"エラー: 「Players_Data」内に「{selected_player}」のフォルダが見つかりません。")
                        st.stop()

                    # 4. フォルダ内のSPSを検索
                    query = f"'{player_folder_id}' in parents and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
                    results = drive_service_write.files().list(q=query, fields="files(id, name)").execute()
                    sheets = results.get('files', [])
                    if not sheets:
                        st.error(f"エラー: 「{selected_player}」フォルダ内にスプレッドシートが見つかりません。")
                        st.stop()
                    
                    sheet_id = sheets[0]['id']
                    
                    # 5. SPSの「マスター」シートを取得し、重複・上書きチェック
                    sh = gc.open_by_key(sheet_id)
                    try:
                        worksheet = sh.worksheet("マスター")
                    except gspread.exceptions.WorksheetNotFound:
                        st.error("エラー: スプレッドシート内に「マスター」という名前のシートが見つかりません。")
                        st.stop()

                    existing_data = worksheet.get_all_values()
                    
                    rows_to_append = []
                    update_count = 0

                    for item in game_checkboxes:
                        is_target = True if register_all else item["is_checked"]
                        if not is_target:
                            continue

                        row = item["export_row"]
                        new_date = row[0]
                        new_start = row[1]
                        new_end = row[2]

                        # SPSの54列フォーマットに変換 (日付, 開始, 終了, レーン, ゲーム, オイル長, オイル量, ボール, F1-1...)
                        formatted_row = [
                            row[0], row[1], row[2],  # 日付, 開始時刻, 終了時刻
                            row[3],                  # レーン
                            row[4],                  # ゲーム数（ゲーム名）
                            "", "", "",              # オイル長, オイル量, 使用ボール (将来用空欄)
                        ]
                        formatted_row.extend(row[5:]) 

                        # 重複確認ロジック（日付一致 ＋ 開始or終了の一致）
                        match_found = False
                        for i, ex_row in enumerate(existing_data):
                            if i == 0 or len(ex_row) < 3: 
                                continue # ヘッダー行や空行はスキップ
                            
                            ex_date = ex_row[0]
                            ex_start = ex_row[1]
                            ex_end = ex_row[2]

                            if ex_date == new_date and (ex_start == new_start or ex_end == new_end):
                                # 一致した行を上書き (gspreadは1始まりのため i+1)
                                row_num = i + 1
                                worksheet.update(range_name=f"A{row_num}", values=[formatted_row])
                                existing_data[i] = formatted_row # メモリ上のデータも更新（同一バッチ内の重複防止）
                                update_count += 1
                                match_found = True
                                break

                        # 一致するものがなければ新規追加リストへ
                        if not match_found:
                            rows_to_append.append(formatted_row)

                    # 6. 新規データの一括書き込み
                    if rows_to_append:
                        worksheet.append_rows(rows_to_append)
                    
                    add_count = len(rows_to_append)
                    st.success(f"🎉 登録完了！ 新規追加: {add_count}件 / 上書き更新: {update_count}件")

                except Exception as e:
                    st.error(f"SPSへの登録中にエラーが発生しました: {e}")
