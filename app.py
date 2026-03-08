import streamlit as st
import cv2
import numpy as np
import io
import csv
import json
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ⚠️ グラフ描画用の設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- ページ設定 ---
st.set_page_config(page_title="ボウリング解析", layout="wide")
st.title("🎳 ボウリングスコア自動解析システム")

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
if "app_state" not in st.session_state:
    st.session_state.app_state = "init"
if "drive_images" not in st.session_state:
    st.session_state.drive_images = []
if "analyzed_data" not in st.session_state:
    st.session_state.analyzed_data = []

st.markdown("### 📥 ボウリング画像の読み込み")

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
                
                st.session_state.drive_images = downloaded_images
                st.session_state.app_state = "processing"
                st.rerun()
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
# 📍 【ブロック 3】 AIプロンプトの定義
# =========================================================
# 📝 ロジック変更：各ゲームごとにtimeを出力するように指示を変更
prompt = """
あなたはプロのボウリングスコア記録員です。
画像はボウリングのスコアシートから、スコア部分だけを切り取って縦に並べたものです。
あらかじめ画像解析AIによって、各フレームの投球結果が赤色で書き込まれています。これをヒントにしてください。
以下の【ルール】に従って、フレームごとの「累計トータルスコア」と「各ゲームの開始時刻」を正確に読み取り、JSON形式で出力してください。

【ルール】
1. 各ゲームの行の下段には、「累計トータルスコア」が書かれています。1F〜10Fまでの10個の累計スコア数字を配列にしてください。
2. ⚠️重要⚠️ 画像に複数のゲームが写っている場合は、絶対に省略せず【写っているすべてのゲーム】のデータを配列 "games" に出力してください。
3. 各ゲームの行の左側などに「ゲーム開始時刻」が書かれている場合は、それぞれのゲームごとの時刻を読み取ってください。
4. Markdownの記号などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット例】
{
  "date": "2026/02/28",
  "lane": "12",
  "games": [
    {
      "game_num": "GAME 1",
      "time": "14:30",
      "frame_totals": [20, 47, 56, 86, 115, 135, 155, 185, 205, 225],
      "total": "225"
    },
    {
      "game_num": "GAME 2",
      "time": "14:45",
      "frame_totals": [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],
      "total": "90"
    }
  ]
}
"""

# =========================================================
# 📍 【ブロック 4】 共通関数・定数定義
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
COLOR_OPENCV = (255, 0, 0)
COLOR_AI = (0, 0, 220)
COLOR_PERCENT = (50, 50, 50)

# =========================================================
# 📍 【ブロック 5〜10】 解析処理ループ
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

        if all_global_pin_pcts:
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
            pink_inks = {f: game_info['purple_data'][f]['pct'] for f in range(10)}
            pink_inks['10_3'] = game_info['purple_data']['10_3']['pct']
            
            for f in range(10):
                l_data = game_info['light_purple_data'][f]
                cv2.polylines(output_img, [l_data['pts']], isClosed=True, color=(216, 191, 216), thickness=1)
                p_data = game_info['purple_data'][f]
                cv2.polylines(output_img, [p_data['pts']], isClosed=True, color=(255, 105, 180), thickness=2)
                put_rotated_text(output_img, f"{p_data['pct']:.0f}%", p_data['rx'], p_data['ry'] - 5, 0, 0, 0, COLOR_PERCENT, scale=0.4, thickness=1)
            p_data_3 = game_info['purple_data']['10_3']
            cv2.polylines(output_img, [p_data_3['pts']], isClosed=True, color=(255, 105, 180), thickness=2)
            put_rotated_text(output_img, f"{p_data_3['pct']:.0f}%", p_data_3['rx'], p_data_3['ry'] - 5, 0, 0, 0, COLOR_PERCENT, scale=0.4, thickness=1)

            all_frame_pins = []
            for f in range(12):
                frame_pins = []
                for row_idx, col_offset in pin_positions:
                    pin_data = game_info['pin_data'][(f, row_idx, col_offset)]
                    pts_y = pin_data['pts']
                    
                    cv2.polylines(output_img, [pts_y], isClosed=True, color=(0, 255, 255), thickness=1)

                    if pin_data['pct'] >= dyn_thresh_circle:
                        pin_num = {0: 7+int(col_offset), 1: 4+int(col_offset-0.5), 2: 2+int(col_offset-1.0), 3: 1}[row_idx]
                        frame_pins.append(pin_num)
                        
                        p_top_left = tuple(pts_y[0][0])
                        p_bottom_right = tuple(pts_y[2][0])
                        cv2.line(output_img, p_top_left, p_bottom_right, (0, 255, 255), 2)
                        
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
                backticks = "`" * 3
                if raw_text.startswith(backticks):
                    raw_text = "\n".join(raw_text.split('\n')[1:-1]).strip()
                gemini_data = json.loads(raw_text)
                break
            except Exception: continue

        games_list = gemini_data.get("games", [])
        all_games_export_data = []

        for group_idx, parsed_data in enumerate(parsed_games):
            g_info = games_list[group_idx] if group_idx < len(games_list) else {}
            ai_frame_totals = g_info.get("frame_totals", [])
            while len(ai_frame_totals) < 10: ai_frame_totals.append(0)
            
            row_data = [""] * 50
            row_data[0] = str(gemini_data.get("date") or "").replace("-", "/")
            # 📝 ロジック変更：ゲームごとのtimeを取得（無い場合は全体のtime、それでも無ければ空文字）
            row_data[1] = str(g_info.get("time") or gemini_data.get("time") or "")
            row_data[2] = str(gemini_data.get("lane") or "")
            row_data[49] = str(g_info.get("total") or "")

            for f in range(9): row_data[target_indices[f]] = ",".join(map(str, parsed_data['all_frame_pins'][f]))
            row_data[target_indices[9]] = ",".join(map(str, parsed_data['p9']))
            if len(parsed_data['p9']) == 0:
                row_data[target_indices[10]] = ",".join(map(str, parsed_data['p10']))
                row_data[target_indices[11]] = ",".join(map(str, parsed_data['p11']))
            else:
                row_data[target_indices[10]] = ""
                row_data[target_indices[11]] = ",".join(map(str, parsed_data['p10']))

            final_throws, throw_colors = [""] * 21, [COLOR_OPENCV] * 21
            for f in range(9):
                v1 = 10 - len(parsed_data['all_frame_pins'][f])
                str1 = 'X' if v1 == 10 else ('-' if v1 == 0 else str(v1))
                final_throws[f*2] = str1
                if str1 != 'X':
                    if parsed_data['pink_inks'][f] >= dyn_thresh_pink: final_throws[f*2+1] = "R:/"
                    else:
                        diff = int(ai_frame_totals[f] if str(ai_frame_totals[f]).isdigit() else 0) - int(ai_frame_totals[f-1] if f > 0 and str(ai_frame_totals[f-1]).isdigit() else 0)
                        if diff >= 10: final_throws[f*2+1], throw_colors[f*2+1] = "R:/", COLOR_AI
                        else: final_throws[f*2+1], throw_colors[f*2+1] = ("R:-" if max(0, min(9-v1, diff-v1)) == 0 else f"R:{max(0, min(9-v1, diff-v1))}"), COLOR_AI

            diff_10 = int(ai_frame_totals[9] if str(ai_frame_totals[9]).isdigit() else 0) - int(ai_frame_totals[8] if str(ai_frame_totals[8]).isdigit() else 0)
            v1_10 = 10 - len(parsed_data['p9'])
            final_throws[18] = 'X' if v1_10 == 10 else ('-' if v1_10 == 0 else str(v1_10))
            if final_throws[18] == 'X':
                v2_10 = 10 - len(parsed_data['p10'])
                final_throws[19] = 'X' if v2_10 == 10 else ('-' if v2_10 == 0 else str(v2_10))
                if final_throws[19] == 'X':
                    v3_10 = 10 - len(parsed_data['p11'])
                    final_throws[20] = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
                else:
                    if parsed_data['pink_inks']['10_3'] >= dyn_thresh_pink: final_throws[20] = "R:/"
                    elif diff_10 - 10 >= 10: final_throws[20], throw_colors[20] = "R:/", COLOR_AI
                    else: final_throws[20], throw_colors[20] = ("R:-" if max(0, min(9-v2_10, diff_10-10-v2_10)) == 0 else f"R:{max(0, min(9-v2_10, diff_10-10-v2_10))}"), COLOR_AI
            else:
                if parsed_data['pink_inks'][9] >= dyn_thresh_pink:
                    final_throws[19] = "R:/"
                    v3_10 = 10 - len(parsed_data['p10'])
                    final_throws[20] = 'X' if v3_10 == 10 else ('-' if v3_10 == 0 else str(v3_10))
                else:
                    if diff_10 >= 10:
                        final_throws[19], throw_colors[19] = "R:/", COLOR_AI
                        v3_10 = max(0, min(10, diff_10 - 10))
                        final_throws[20], throw_colors[20] = ("R:X" if v3_10 == 10 else ("R:-" if v3_10 == 0 else f"R:{v3_10}")), COLOR_AI
                    else:
                        final_throws[19], throw_colors[19] = ("R:-" if max(0, min(9-v1_10, diff_10-v1_10)) == 0 else f"R:{max(0, min(9-v1_10, diff_10-v1_10))}"), COLOR_AI

            for t_idx, col_idx in enumerate(throw_cols): row_data[col_idx] = final_throws[t_idx]
            all_games_export_data.append(row_data)

            for f in range(9):
                put_rotated_text(output_img, final_throws[f*2].replace("R:", ""), parsed_data['start_x_base'] + f * parsed_data['box_w'] + 3 * parsed_data['current_scale'], parsed_data['py1_local'] - 2 * parsed_data['current_scale'], parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], throw_colors[f*2])
                put_rotated_text(output_img, final_throws[f*2+1].replace("R:", ""), parsed_data['start_x_base'] + f * parsed_data['box_w'] + 10 * parsed_data['current_scale'], parsed_data['py1_local'] - 2 * parsed_data['current_scale'], parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], throw_colors[f*2+1])
            put_rotated_text(output_img, final_throws[18].replace("R:", ""), parsed_data['start_x_base'] + 9 * parsed_data['box_w'] + 3 * parsed_data['current_scale'], parsed_data['py1_local'] - 2 * parsed_data['current_scale'], parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], throw_colors[18])
            put_rotated_text(output_img, final_throws[19].replace("R:", ""), parsed_data['start_x_base'] + 9 * parsed_data['box_w'] + 10 * parsed_data['current_scale'], parsed_data['py1_local'] - 2 * parsed_data['current_scale'], parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], throw_colors[19])
            put_rotated_text(output_img, final_throws[20].replace("R:", ""), parsed_data['start_x_base'] + 9 * parsed_data['box_w'] + 17 * parsed_data['current_scale'], parsed_data['py1_local'] - 2 * parsed_data['current_scale'], parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], throw_colors[20])

            calc_totals = calculate_bowling_score(final_throws)
            ai_tot_int = int(g_info.get("total", 0)) if str(g_info.get("total", "")).isdigit() else int(ai_frame_totals[-1]) if ai_frame_totals else 0
            result_text_x = parsed_data['start_x_base'] + 9 * parsed_data['box_w'] + 5 * parsed_data['current_scale']
            result_text_y = parsed_data['py1_local'] - 10 * parsed_data['current_scale']

            if calc_totals and len(ai_frame_totals) > 0 and calc_totals[-1] == ai_tot_int:
                check_str = f"MATCH ({calc_totals[-1]})"
                check_color = (0, 150, 0)
            else:
                calc_val = calc_totals[-1] if calc_totals else 0
                check_str = f"DIFF! ({calc_val} vs {ai_tot_int})"
                check_color = COLOR_AI

            put_rotated_text(output_img, check_str, result_text_x, result_text_y, parsed_data['new_ref1'][0], parsed_data['new_ref1'][1], parsed_data['theta'], check_color, scale=0.6, thickness=2)

        all_analyzed_results.append({
            "filename": img_info['name'],
            "output_img": output_img,
            "export_data": all_games_export_data
        })
        progress_bar.progress((idx + 1) / total_images)

    st.session_state.analyzed_data = all_analyzed_results
    st.session_state.app_state = "done"
    st.rerun()

# =========================================================
# 📍 【ブロック 11】 統合結果画面・登録フォーム
# =========================================================
if st.session_state.app_state == "done":
    st.success("✅ 全ての画像の解析が完了しました！")
    
    with st.form("register_form"):
        st.markdown("### 👤 プレイヤー選択")
        player_list = ["001_田中一吉", "002_田中佳恵", "003_田中蒼之助", "004_田中柾吉", "005_米田稔", "999_ゲスト"]
        selected_player = st.selectbox("このスコアを誰のデータとして登録しますか？", player_list)
        
        st.markdown("---")
        register_all = st.checkbox("全てのゲームをマスターに登録する", value=True)
        st.markdown("---")
        
        game_checkboxes = []
        global_game_num = 1
        
        for data_idx, data in enumerate(st.session_state.analyzed_data):
            st.markdown(f"#### 📄 画像 {data_idx+1}: {data['filename']}")
            st.image(cv2.cvtColor(data['output_img'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            for local_idx, row in enumerate(data['export_data']):
                game_name = f"GAME {global_game_num}"
                st.session_state.analyzed_data[data_idx]['export_data'][local_idx][3] = game_name
                
                date_str = row[0] if row[0] else "日付不明"
                time_str = row[1] if row[1] else "時刻不明"
                ai_total_str = row[49]
                ai_total_int = int(ai_total_str) if str(ai_total_str).isdigit() else 0
                
                throws_for_calc = [row[i] for i in throw_cols]
                calc_totals = calculate_bowling_score(throws_for_calc)
                calc_val = calc_totals[-1] if calc_totals else 0

                if calc_totals and calc_val == ai_total_int:
                    match_status = "✅AI一致"
                else:
                    match_status = "⚠️AI不一致"
                
                display_text = f"{date_str} {game_name}_{time_str} ｜ スコア: {ai_total_str}_{match_status}"
                
                is_checked = st.checkbox(display_text, value=True, key=f"check_{data_idx}_{local_idx}")
                game_checkboxes.append((data_idx, local_idx, is_checked))
                
                global_game_num += 1
            
            st.markdown("---")

        submit_btn = st.form_submit_button("💾 選択したデータを確定（CSV生成）")

    if submit_btn:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        export_count = 0
        
        check_idx = 0
        for data_idx, data in enumerate(st.session_state.analyzed_data):
            for local_idx, row in enumerate(data['export_data']):
                is_target = True if register_all else game_checkboxes[check_idx][2]
                if is_target:
                    export_row = [selected_player] + row
                    writer.writerow(export_row)
                    export_count += 1
                check_idx += 1
                
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

