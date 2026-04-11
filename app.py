import streamlit as st
import cv2
import numpy as np
import io
import csv
import json
import time
import random
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
st.set_page_config(page_title="ボウリング解析", page_icon="🎳", layout="wide")

# ▼▼▼ プレイヤー分析画面のAWARD画面を参考にした共通ダークテーマ・統一CSS ▼▼▼
st.markdown("""
    <style>
    /* アプリ全体をAWARD風のダークテーマに */
    .stApp {
        background-color: #1a1a1c !important;
        color: silver !important;
    }
    
    /* ヘッダーや通常テキストの色をAWARD風（silver）に統一 */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: silver !important;
    }
    
    /* 入力ウィジェットのコンテナ（背景色と枠線） */
    div[data-baseweb="input"], 
    div[data-baseweb="select"] > div, 
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="checkbox"] > div {
        background-color: #2a2a2e !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    
    /* 入力テキストカラー */
    input, textarea, div[data-baseweb="select"] {
        color: white !important;
    }
    
    /* プルダウンのメニューリスト */
    ul[role="listbox"] {
        background-color: #2a2a2e !important;
        border: 1px solid #444 !important;
    }
    li[role="option"] {
        color: white !important;
    }
    li[role="option"]:hover {
        background-color: #1c1c1e !important;
    }

    /* ボタンの共通スタイル */
    button[kind="primary"], button[kind="secondary"] {
        background: linear-gradient(145deg, #2a2a2e, #1c1c1e) !important;
        border: 1px solid #333 !important;
        color: #bf953f !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    }
    button[kind="primary"]:hover, button[kind="secondary"]:hover {
        border-color: #bf953f !important;
        color: white !important;
    }

    /* Expander（折りたたみ） */
    div[data-testid="stExpander"] {
        background-color: #1c1c1e !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        margin-bottom: 0rem !important;
    }
    div[data-testid="stExpander"] summary p {
        color: #E0E0E0 !important;
        font-weight: bold !important;
    }
    
    /* サイドバー背景 */
    section[data-testid="stSidebar"] {
        background-color: #1c1c1e !important;
        border-right: 1px solid #333 !important;
    }
    
    /* データフレームの背景等 */
    div[data-testid="stDataFrame"] {
        background-color: #1a1a1c !important;
    }

    /* スマホ画面で強制的に横並び（st.columns）を維持するための安全な設定 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
        div[data-testid="stColumn"] {
            width: auto !important;
            flex: 1 1 0% !important;
            min-width: 0 !important;
        }
    }
    /* 10フレームなどの横幅を確保するため入力欄の余白を少し圧縮 */
    .stTextInput > div > div > input {
        padding: 0.3rem !important;
    }
    .stMultiSelect > div > div > div {
        padding: 0px !important;
    }
    
    /* ▼▼▼ アイコン要素を完全に消去し、中央揃え ▼▼▼ */
    div[data-testid="stPopover"] button span:has(span[data-testid="stIconMaterial"]),
    div[data-testid="stPopover"] button span[data-testid="stIconMaterial"] {
        display: none !important;
        width: 0px !important;
        margin: 0px !important;
    }
    
    /* ボタンの余白を消して中央に寄せる */
    div[data-testid="stPopover"] button {
        padding-left: 0px !important;
        padding-right: 0px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    /* 文字のコンテナを100%にして中央揃え */
    div[data-testid="stPopover"] button div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }
    div[data-testid="stPopover"] button p {
        margin: 0 !important;
        text-align: center !important;
    }

    /* ========================================================
       ▼ Streamlit用 確実なボタン色変更（マーカー方式） ▼
       ======================================================== */
    /* ゴールド強調ボタン（取込・修正完了など） */
    div[data-testid="stElementContainer"]:has(.gold-btn-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.gold-btn-marker) + div.element-container button {
        background: linear-gradient(145deg, #bf953f, #aa771c) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        border: 2px solid #fcf6ba !important;
        box-shadow: 0 0 15px rgba(191, 149, 63, 0.6) !important;
        border-radius: 12px !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        min-height: 55px !important;
    }
    /* Streamlitの<p>タグによる文字色上書きを防ぐ */
    div[data-testid="stElementContainer"]:has(.gold-btn-marker) + div[data-testid="stElementContainer"] button p,
    div.element-container:has(.gold-btn-marker) + div.element-container button p {
        color: #ffffff !important;
        font-weight: 900 !important;
    }
    div[data-testid="stElementContainer"]:has(.gold-btn-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.gold-btn-marker) + div.element-container button:hover {
        background: linear-gradient(145deg, #fcf6ba, #bf953f) !important;
        color: #ffffff !important;
        box-shadow: 0 0 25px rgba(191, 149, 63, 0.9) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stElementContainer"]:has(.gold-btn-marker) + div[data-testid="stElementContainer"] button:hover p,
    div.element-container:has(.gold-btn-marker) + div.element-container button:hover p {
        color: #ffffff !important;
    }

    /* 赤強調ボタン（登録実行：薄めの赤） */
    div[data-testid="stElementContainer"]:has(.red-btn-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.red-btn-marker) + div.element-container button {
        background: linear-gradient(145deg, #e66465, #c0392b) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        border: 1px solid #ff9f9f !important;
        box-shadow: 0 0 15px rgba(230, 100, 101, 0.4) !important;
        border-radius: 12px !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        min-height: 55px !important;
    }
    /* Streamlitの<p>タグによる文字色上書きを防ぐ */
    div[data-testid="stElementContainer"]:has(.red-btn-marker) + div[data-testid="stElementContainer"] button p,
    div.element-container:has(.red-btn-marker) + div.element-container button p {
        color: #ffffff !important;
        font-weight: 900 !important;
    }
    div[data-testid="stElementContainer"]:has(.red-btn-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.red-btn-marker) + div.element-container button:hover {
        background: linear-gradient(145deg, #ff7979, #e66465) !important;
        box-shadow: 0 0 25px rgba(230, 100, 101, 0.8) !important;
        transform: translateY(-2px) !important;
    }

    /* 赤強調ボタン（登録実行：薄めの赤） */
    div[data-testid="stElementContainer"]:has(.red-btn-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.red-btn-marker) + div.element-container button {
        background: linear-gradient(145deg, #e66465, #c0392b) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        border: 1px solid #ff9f9f !important;
        box-shadow: 0 0 15px rgba(230, 100, 101, 0.4) !important;
        border-radius: 12px !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        min-height: 55px !important;
    }
    div[data-testid="stElementContainer"]:has(.red-btn-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.red-btn-marker) + div.element-container button:hover {
        background: linear-gradient(145deg, #ff7979, #e66465) !important;
        box-shadow: 0 0 25px rgba(230, 100, 101, 0.8) !important;
        transform: translateY(-2px) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 36px; white-space: nowrap; margin-bottom: 16px; font-weight: bold; font-family: "Arial Black", Impact, sans-serif;'>
    <span style='background: linear-gradient(135deg, #bf953f 0%, #fcf6ba 20%, #555555 35%, #b38728 55%, #ffffff 75%, #aa771c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(2px 4px 4px rgba(0,0,0,0.8)); padding-right: 5px;'>EAGLE ROLLERS</span>
</div>
""", unsafe_allow_html=True)

# --- 共通でおしゃれなタイトルを描画する関数 ---
def render_section_title(title_text):
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 10px 20px; border-radius: 8px; border-left: 5px solid #bf953f; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <span style="color: silver; font-size: 18px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif; letter-spacing: 1px;">{title_text}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Google Sheets 接続ヘルパー ---
@st.cache_resource(ttl=600)
def get_gspread_client():
    import json
    import gspread
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds_json_str = st.secrets["google_credentials"]
    creds_info = json.loads(creds_json_str, strict=False)
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    gc = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    query = "name = 'EagleBowl_ROLLERS' and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    if results.get('files', []):
        return gc.open_by_key(results['files'][0]['id'])
    return None

# =========================================================
# ▼ 追加：お知らせ・イベント機能用共通関数 ▼
# =========================================================
def get_announcement_data(sh):
    try:
        return sh.worksheet("お知らせ").acell("A1").value or "現在、お知らせはありません。"
    except: return "現在、お知らせはありません。"

def update_announcement_data(sh, text):
    try:
        sh.worksheet("お知らせ").update(range_name="A1", values=[[text]])
        return True
    except: return False

def sync_calendar_to_sps(sh):
    """Google Driveの今月のPDFを読み込み、1ヶ月分のイベントをSPSに保存する"""
    import datetime
    import json
    import time
    import random
    import re
    from google.genai import types
    from google import genai
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    
    now = datetime.datetime.now()
    try:
        creds_json_str = st.secrets["google_credentials"]
        creds_info = json.loads(creds_json_str, strict=False)
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        
        drive_creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive'])
        drive_service = build('drive', 'v3', credentials=drive_creds)
        
        gemini_api_key = st.secrets.get("gemini_api_key", "")
        ai_client = genai.Client(api_key=gemini_api_key)

        f_query = "name = 'イベントスケジュール' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        folders = drive_service.files().list(q=f_query).execute().get('files', [])
        if not folders: return "フォルダが見つかりません。"
        
        p_query = f"'{folders[0]['id']}' in parents and name contains '{now.month}月' and mimeType = 'application/pdf'"
        files = drive_service.files().list(q=p_query).execute().get('files', [])
        if not files: return "今月のPDFが見つかりません。"
        
        content = drive_service.files().get_media(fileId=files[0]['id']).execute()
        
        # ▼ 時間や金額を確実に排除し、「行事名」のみをピンポイントで狙うプロンプトに修正
        prompt = "このカレンダー（1枚目）とイベント一覧（2枚目）から、1ヶ月分の【日付(M/D形式)】、【大会名(イベント名)】、および【行事名】を抽出し、純粋なJSON配列 [{'date':'4/1', 'event':'大会名', 'desc':'行事名'}, ...] 形式で出力してください。\n\n※厳守事項※\n1. 'desc' には、大会名のすぐ下に記載されている「行事名」の文字列のみを入れてください。\n2. 「ゲーム数（〇G）」「参加費・金額（¥〇〇など）」「時間（PM〇:〇〇など）」の数字や情報は、トラブル防止のため **絶対に** 含めないでください。\n3. イベントがない日は含めないでください。"
        
        max_retries = 5
        response = None
        last_error = ""
        success = False
        
        for attempt_model in ["gemini-2.5-pro", "gemini-1.5-pro"]:
            for attempt in range(max_retries):
                try:
                    response = ai_client.models.generate_content(
                        model=attempt_model,
                        contents=[types.Part.from_bytes(data=content, mime_type="application/pdf"), prompt]
                    )
                    if not response or not response.text:
                        raise ValueError("AIからの応答が空でした。")
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    error_msg = last_error.lower()
                    if attempt < max_retries - 1:
                        wait_sec = (2 ** (attempt + 1)) + random.uniform(0, 1)
                        time.sleep(wait_sec)
                        continue
                    break 
            if success:
                break
        
        if not success or not response:
            return f"AI解析エラー: サーバーが混雑しています。時間を置いて再度お試しください。（詳細: {last_error}）"

        raw_text = response.text
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return "AIが期待するJSON形式でデータを出力しませんでした。"
            
        data = json.loads(json_str)
        
        wks = sh.worksheet("イベントカレンダー")
        wks.clear()
        wks.update(range_name="A1", values=[[d.get('date', ''), d.get('event', ''), d.get('desc', '')] for d in data])
        return "更新完了！"
    except Exception as e: return f"エラー: {str(e)}"

def get_today_event_from_sps(sh):
    """SPSのイベントカレンダーから今日の日付のイベントと説明を取得"""
    import datetime
    now = datetime.datetime.now()
    t1, t2 = f"{now.month}/{now.day}", f"{now.month:02d}/{now.day:02d}"
    try:
        records = sh.worksheet("イベントカレンダー").get_all_values()
        for row in records:
            if len(row) >= 2 and (row[0] == t1 or row[0] == t2):
                event_name = row[1]
                event_desc = row[2] if len(row) > 2 else ""
                return event_name, event_desc
        return "イベント予定なし", ""
    except: return "イベント予定なし", ""
# ▲ 追加ここまで ▲
    try:
        return sh.worksheet("お知らせ").acell("A1").value or "現在、お知らせはありません。"
    except: return "現在、お知らせはありません。"

def update_announcement_data(sh, text):
    try:
        sh.worksheet("お知らせ").update(range_name="A1", values=[[text]])
        return True
    except: return False

def sync_calendar_to_sps(sh):
    """Google Driveの今月のPDFを読み込み、1ヶ月分のイベントをSPSに保存する"""
    import datetime
    import json
    import time
    import random
    import re
    from google.genai import types
    from google import genai
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    
    now = datetime.datetime.now()
    try:
        # 関数内で独立してAPI認証を行う（エラー回避のため）
        creds_json_str = st.secrets["google_credentials"]
        creds_info = json.loads(creds_json_str, strict=False)
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        
        drive_creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive'])
        drive_service = build('drive', 'v3', credentials=drive_creds)
        
        gemini_api_key = st.secrets.get("gemini_api_key", "")
        ai_client = genai.Client(api_key=gemini_api_key)

        f_query = "name = 'イベントスケジュール' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        folders = drive_service.files().list(q=f_query).execute().get('files', [])
        if not folders: return "フォルダが見つかりません。"
        
        p_query = f"'{folders[0]['id']}' in parents and name contains '{now.month}月' and mimeType = 'application/pdf'"
        files = drive_service.files().list(q=p_query).execute().get('files', [])
        if not files: return "今月のPDFが見つかりません。"
        
        content = drive_service.files().get_media(fileId=files[0]['id']).execute()
        
        # PDF2枚目の説明文も含めて取得するプロンプト
        prompt = "このカレンダー（1枚目）とイベント一覧（2枚目）から、1ヶ月分の【日付(M/D形式)】、【イベント名】、および【イベントの説明文章】（2枚目のイベント名の下の行事名やゲーム数・参加費・詳細説明など）を抽出し、純粋なJSON配列 [{'date':'4/1', 'event':'イベント名', 'desc':'説明文章'}, ...] 形式で出力して。イベントがない日は含めないで。"
        
        # --- サーバー高負荷対策（自動リトライ＆モデル切り替え） ---
        max_retries = 5
        response = None
        last_error = ""
        success = False
        
        # 2.5-pro がダメなら 1.5-pro で予備実行する
        for attempt_model in ["gemini-2.5-pro", "gemini-1.5-pro"]:
            for attempt in range(max_retries):
                try:
                    response = ai_client.models.generate_content(
                        model=attempt_model,
                        contents=[types.Part.from_bytes(data=content, mime_type="application/pdf"), prompt]
                    )
                    # 応答がない場合はエラー扱いにする
                    if not response or not response.text:
                        raise ValueError("AIからの応答が空でした。")
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    error_msg = last_error.lower()
                    # いかなるエラーでも、リトライ上限までは待機してやり直す
                    if attempt < max_retries - 1:
                        wait_sec = (2 ** (attempt + 1)) + random.uniform(0, 1)
                        time.sleep(wait_sec)
                        continue
                    break # このモデルでのリトライを諦める
            if success:
                break
        
        if not success or not response:
            return f"AI解析エラー: サーバーが混雑しています。時間を置いて再度お試しください。（詳細: {last_error}）"

        # 正規表現を使って確実にJSON配列部分だけを抽出する
        raw_text = response.text
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return "AIが期待するJSON形式でデータを出力しませんでした。"
            
        data = json.loads(json_str)
        
        wks = sh.worksheet("イベントカレンダー")
        wks.clear()
        # 日付、イベント名、説明文（desc）の3つを書き込む
        wks.update(range_name="A1", values=[[d.get('date', ''), d.get('event', ''), d.get('desc', '')] for d in data])
        return "更新完了！"
    except Exception as e: return f"エラー: {str(e)}"

def get_today_event_from_sps(sh):
    """SPSのイベントカレンダーから今日の日付のイベントと説明を取得"""
    import datetime
    now = datetime.datetime.now()
    t1, t2 = f"{now.month}/{now.day}", f"{now.month:02d}/{now.day:02d}"
    try:
        records = sh.worksheet("イベントカレンダー").get_all_values()
        for row in records:
            if len(row) >= 2 and (row[0] == t1 or row[0] == t2):
                event_name = row[1]
                event_desc = row[2] if len(row) > 2 else ""
                return event_name, event_desc
        return "イベント予定なし", ""
    except: return "イベント予定なし", ""
# ▲▲▲ 追加ここまで ▲▲▲

# --- セッション初期化 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.user_name = ""
    st.session_state.user_role = ""
    st.session_state.user_public = ""

# --- ログイン画面 ---
if not st.session_state.logged_in:
    render_section_title("ログイン")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_id = st.text_input("ユーザーID  \n(メールアドレス)")
        login_pw = st.text_input("パスワード", type="password")
        if st.button("ログイン", use_container_width=True):
            sh = get_gspread_client()
            if sh:
                ws = sh.worksheet("プレイヤー設定")
                data = ws.get_all_values()
                login_success = False
                for idx, row in enumerate(data):
                    # A列[0]:Email, B列[1]:名前, C列[2]:公開設定, D列[3]:権限, E列[4]:PW
                    if len(row) >= 5 and row[0] == login_id and row[4] == login_pw:
                        st.session_state.logged_in = True
                        st.session_state.user_email = row[0]
                        st.session_state.user_name = row[1]
                        st.session_state.user_public = row[2]
                        st.session_state.user_role = row[3]
                        st.session_state.user_row_index = idx + 1 # スプレッドシートの行番号(1始まり)
                        login_success = True
                        break
                if login_success:
                    st.rerun()
                else:
                    st.error("IDまたはパスワードが間違っています。")
            else:
                st.error("データベースに接続できません。")
    st.stop() # ログイン完了まで下の処理を行わない

# --- サイドバー：設定エリア ---
with st.sidebar:
    st.markdown(f"<div style='color: #bf953f; font-weight: bold; font-size: 18px; margin-bottom: 5px;'>{st.session_state.user_name} さん</div>", unsafe_allow_html=True)
    st.caption(f"権限: {st.session_state.user_role}")
    
    with st.expander("アカウント・友達設定"):
        new_pw = st.text_input("新しいパスワード", type="password")
        
        # 公開設定（3パターン）
        pub_options = ["公開", "友達限定公開", "非公開"]
        current_pub = st.session_state.user_public if st.session_state.user_public in pub_options else "公開"
        new_pub = st.radio("データ公開設定", pub_options, index=pub_options.index(current_pub))
        
        if st.button("設定を更新"):
            sh = get_gspread_client()
            if sh:
                ws = sh.worksheet("プレイヤー設定")
                if new_pw:
                    ws.update_cell(st.session_state.user_row_index, 5, new_pw) # E列(5)
                ws.update_cell(st.session_state.user_row_index, 3, new_pub) # C列(3)
                st.session_state.user_public = new_pub
                st.success("設定を更新しました！")
        
        st.markdown("---")
        st.markdown("**友達追加**")
        friend_email = st.text_input("友達のユーザーID (メールアドレス)")
        if st.button("友達を追加する"):
            if friend_email == st.session_state.user_email:
                st.warning("自分自身は追加できません。")
            elif friend_email:
                sh = get_gspread_client()
                if sh:
                    ws = sh.worksheet("プレイヤー設定")
                    data = ws.get_all_values()
                    
                    # 友達のメアドが存在するか確認
                    friend_name = None
                    for row in data:
                        if row[0] == friend_email:
                            friend_name = row[1]
                            break
                    
                    if friend_name:
                        # 自分の現在の友達リスト(F列)を取得して追加
                        my_row = data[st.session_state.user_row_index - 1]
                        current_friends = my_row[5] if len(my_row) > 5 else ""
                        friends_list = [f.strip() for f in current_friends.split(",")] if current_friends else []
                        
                        if friend_email not in friends_list:
                            friends_list.append(friend_email)
                            new_friends_str = ",".join(friends_list)
                            ws.update_cell(st.session_state.user_row_index, 6, new_friends_str) # F列(6)
                            st.success(f"{friend_name} さんを友達に追加しました！")
                        else:
                            st.info("すでに友達に追加されています。")
                    else:
                        st.error("入力されたIDのユーザーが見つかりません。")

        # 登録済み友達一覧の表示
        st.markdown("**登録済みの友達一覧**")
        if st.button("一覧を更新・表示"):
            sh = get_gspread_client()
            if sh:
                data = sh.worksheet("プレイヤー設定").get_all_values()
                my_row = data[st.session_state.user_row_index - 1]
                current_friends = my_row[5] if len(my_row) > 5 else ""
                if current_friends:
                    friends_list = [f.strip() for f in current_friends.split(",")]
                    friend_names = []
                    for r in data:
                        if r[0] in friends_list:
                            friend_names.append(r[1])
                    for fn in friend_names:
                        st.markdown(f"- {fn}")
                else:
                    st.markdown("友達はまだ登録されていません。")
    
    if st.session_state.user_role in ["開発者", "管理者"]:
        if st.button("🖥️ ユーザセルフ登録を起動", type="primary", use_container_width=True):
            st.session_state.kiosk_mode = True
            st.session_state.kiosk_step = "auth"
            st.rerun()

        # ▼ 追加：管理者・開発者専用メニューをサイドバーに配置 ▼
        st.markdown("---")
        st.subheader("🛠 管理ツール")
        with st.expander("📢 お知らせ・イベント編集"):
            # データベース接続を確保
            sh_admin = get_gspread_client()
            
            # お知らせ更新
            ann_current = get_announcement_data(sh_admin) if sh_admin else ""
            new_ann = st.text_area("お知らせ編集", value=ann_current, height=100)
            if st.button("お知らせを保存"):
                if sh_admin and update_announcement_data(sh_admin, new_ann): 
                    st.success("保存完了")
                    import time
                    time.sleep(1)
                    st.rerun()
            
            st.markdown("---")
            # カレンダー手動読込
            st.write("📅 カレンダーPDF解析")
            if st.button("AIで今月のPDFを読込・保存"):
                if sh_admin:
                    with st.spinner("AIが解析中..."):
                        res = sync_calendar_to_sps(sh_admin)
                        st.info(res)
                else:
                    st.error("データベースに接続できません。")
        # ▲ 追加ここまで ▲

        st.markdown("---")
        app_mode = st.radio("モード選択", ["スコア登録", "オイル情報入力", "プレイヤー分析", "データ比較"], index=0)
    else:
        app_mode = st.radio("モード選択", ["プレイヤー分析"], index=0)
        st.info("※非公開のプレイヤーのデータは表示されません")

# ＃★★★★テンキー入力用共通関数群★★★★
def tk_add(k, c): st.session_state[k] += c
def tk_del(k): st.session_state[k] = st.session_state[k][:-1] if len(st.session_state[k]) > 0 else ""
def tk_clr(k): st.session_state[k] = ""

def render_tenkey(label, state_key, default_val, format_type="none", is_pw=False):
    tracker_key = f"{state_key}_tracker"
    if state_key not in st.session_state or st.session_state.get(tracker_key) != default_val:
        dv = "" if default_val is None else str(default_val)
        st.session_state[state_key] = dv.replace("/", "").replace(":", "")
        st.session_state[tracker_key] = default_val
        
    raw_val = st.session_state[state_key]
    display_val = raw_val
    
    if format_type == "date":
        if len(raw_val) > 4: display_val = f"{raw_val[:2]}/{raw_val[2:4]}/{raw_val[4:6]}"
        elif len(raw_val) > 2: display_val = f"{raw_val[:2]}/{raw_val[2:4]}"
    elif format_type == "time":
        if len(raw_val) > 2: display_val = f"{raw_val[:2]}:{raw_val[2:4]}"
        
    display_text = "*" * len(display_val) if is_pw else display_val

    # 値を画面に反映させるため、テキストボックスの内部状態を直接上書きする
    st.session_state[f"disp_{state_key}"] = display_text

    col1, col2 = st.columns([4, 1])
    with col1:
        # value引数を外し、内部状態(st.session_state)に表示を任せる
        st.text_input(label, disabled=True, key=f"disp_{state_key}")
    with col2:
        with st.popover("⌨"):
            st.markdown("<div style='text-align:center; font-size:12px; color:gray; margin-bottom:5px;'>テンキー</div>", unsafe_allow_html=True)
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.button("7", on_click=tk_add, args=(state_key, "7"), key=f"tk_7_{state_key}", use_container_width=True)
            r1c2.button("8", on_click=tk_add, args=(state_key, "8"), key=f"tk_8_{state_key}", use_container_width=True)
            r1c3.button("9", on_click=tk_add, args=(state_key, "9"), key=f"tk_9_{state_key}", use_container_width=True)
            
            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.button("4", on_click=tk_add, args=(state_key, "4"), key=f"tk_4_{state_key}", use_container_width=True)
            r2c2.button("5", on_click=tk_add, args=(state_key, "5"), key=f"tk_5_{state_key}", use_container_width=True)
            r2c3.button("6", on_click=tk_add, args=(state_key, "6"), key=f"tk_6_{state_key}", use_container_width=True)
            
            r3c1, r3c2, r3c3 = st.columns(3)
            r3c1.button("1", on_click=tk_add, args=(state_key, "1"), key=f"tk_1_{state_key}", use_container_width=True)
            r3c2.button("2", on_click=tk_add, args=(state_key, "2"), key=f"tk_2_{state_key}", use_container_width=True)
            r3c3.button("3", on_click=tk_add, args=(state_key, "3"), key=f"tk_3_{state_key}", use_container_width=True)
            
            r4c1, r4c2, r4c3 = st.columns(3)
            r4c1.button("C", on_click=tk_clr, args=(state_key,), key=f"tk_C_{state_key}", use_container_width=True)
            r4c2.button("0", on_click=tk_add, args=(state_key, "0"), key=f"tk_0_{state_key}", use_container_width=True)
            r4c3.button("戻", on_click=tk_del, args=(state_key,), key=f"tk_D_{state_key}", use_container_width=True)
            
    return display_val

# =========================================================
# 【新機能】ユーザセルフ登録 (専用画面モード) の制御
# =========================================================
if st.session_state.get("kiosk_mode"):
    # 勝手にセッションが切れないようにバックグラウンドで1分ごとに通信を発生
    import streamlit.components.v1 as components
    components.html("<script>setInterval(function(){window.parent.postMessage('ping', '*');}, 60000);</script>", height=0, width=0)

    # サイドバーと標準ヘッダーを完全に隠すことで他の操作を封じる
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none !important;}
        header {display: none !important;}
        .kiosk-exit-btn {
            position: fixed; bottom: 10px; right: 10px; z-index: 9999; opacity: 0.1;
        }
        .kiosk-exit-btn:hover { opacity: 1.0; }
        .kiosk-header {
            font-family: 'Arial Black', Impact, sans-serif;
            color: #d4af37 !important;
            text-align: center; letter-spacing: 3px;
            font-size: 36px; margin-top: 50px; margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 開発者向けの隠し終了ボタン（右下）
    st.markdown("<div class='kiosk-exit-btn'>", unsafe_allow_html=True)
    with st.popover("✖"):
        st.markdown("<div style='color: silver; font-weight: bold; margin-bottom: 10px;'>キオスクモード終了</div>", unsafe_allow_html=True)
        exit_pw = st.text_input("管理者/開発者パスワードを入力", type="password", key="kiosk_exit_pw")
        if st.button("終了して戻る", use_container_width=True, key="kiosk_exit_btn"):
            sh = get_gspread_client()
            if sh:
                ws = sh.worksheet("プレイヤー設定")
                # ログイン時に保存している行番号から、現在ログイン中の大元ユーザーのパスワードを取得
                row_data = ws.row_values(st.session_state.user_row_index)
                if len(row_data) >= 5 and row_data[4] == exit_pw:
                    st.session_state.kiosk_mode = False
                    st.rerun()
                else:
                    st.error("パスワードが正しくありません。")
            else:
                st.error("データベースに接続できません。")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("kiosk_step") == "auth":
        st.markdown("<div class='kiosk-header'>CHECK-IN</div>", unsafe_allow_html=True)
        
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            sh = get_gspread_client()
            if sh:
                ws = sh.worksheet("プレイヤー設定")
                data = ws.get_all_values()
                players = [row[1] for row in data[1:] if len(row) >= 5 and row[1]]
                
                selected_user = st.selectbox("プレイヤーを選択してください", ["選択してください"] + players)
                kiosk_pw = render_tenkey("パスワードを入力してください", "tk_kiosk_pass", "", format_type="none", is_pw=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ 認証して登録画面へ進む", use_container_width=True):
                    if selected_user == "選択してください":
                        st.error("プレイヤーを選択してください。")
                    elif kiosk_pw == "":
                        st.error("パスワードを入力してください。")
                    else:
                        auth_success = False
                        for row in data[1:]:
                            if row[1] == selected_user and str(row[4]) == str(kiosk_pw):
                                auth_success = True
                                break
                        if auth_success:
                            st.session_state.kiosk_user = selected_user
                            st.session_state.kiosk_step = "register"
                            st.rerun()
                        else:
                            st.error("パスワードが正しくありません。")
        st.stop()
        
    # 認証後は既存のモード変数（app_mode）を上書きして合流
    elif st.session_state.kiosk_step == "register":
        app_mode = "スコア登録"
    elif st.session_state.kiosk_step == "stats":
        app_mode = "プレイヤー分析"

# =========================================================
# 【新機能】オイル情報入力モード
# =========================================================
if app_mode == "オイル情報入力":
    render_section_title("オイル情報入力")
    st.markdown("ボウリング場のオイル長とオイル量を記録・登録します。")
    
    # ① 修正機能（直接「オイル入力」シートを開くSPSリンクボタン）
    sh = get_gspread_client()
    if sh:
        try:
            oil_sheet = sh.worksheet("オイル入力")
            sheet_url = f"https://docs.google.com/spreadsheets/d/{sh.id}/edit#gid={oil_sheet.id}"
            st.link_button("オイル入力済みデータを修正する (SPSを展開)", sheet_url)
        except Exception as e:
            st.error("SPSに「オイル入力」シートが見つかりません。")
            
    # ③ 日時のデフォルト値生成（UTCに9時間を加えて日本時間にする）
    from datetime import datetime, timedelta, timezone
    # タイムゾーンをJST（UTC+9）に設定
    JST = timezone(timedelta(hours=+9), 'JST')
    now = datetime.now(JST)
    if "oil_input_date" not in st.session_state:
        st.session_state.oil_input_date = now.strftime("%y/%m/%d")
    if "oil_input_hour" not in st.session_state:
        st.session_state.oil_input_hour = f"{now.hour:02d}"
    if "oil_input_minute" not in st.session_state:
        st.session_state.oil_input_minute = f"{now.minute:02d}"
        
    # 入力欄のセッション初期化
    for i in range(1, 19):
        if f"oil_in_len_{i}" not in st.session_state:
            st.session_state[f"oil_in_len_{i}"] = ""
        if f"oil_in_vol_{i}" not in st.session_state:
            st.session_state[f"oil_in_vol_{i}"] = ""

    # ④ 各種ボタンのコールバック関数
    def copy_latest_oil():
        oil_data = st.session_state.get("oil_data", [])
        if not oil_data and sh: 
            try:
                oil_data_raw = sh.worksheet("オイル入力").get_all_values()
                oil_data = oil_data_raw[2:] if len(oil_data_raw) > 2 else []
                st.session_state.oil_data = oil_data
            except:
                pass
        if oil_data:
            latest = oil_data[-1]
            for i in range(1, 19):
                l_idx = i * 2
                v_idx = i * 2 + 1
                st.session_state[f"oil_in_len_{i}"] = latest[l_idx] if len(latest) > l_idx else ""
                st.session_state[f"oil_in_vol_{i}"] = latest[v_idx] if len(latest) > v_idx else ""

    def clear_all():
        for i in range(1, 19):
            st.session_state[f"oil_in_len_{i}"] = ""
            st.session_state[f"oil_in_vol_{i}"] = ""

    def clear_lane(lane):
        st.session_state[f"oil_in_len_{lane}"] = ""
        st.session_state[f"oil_in_vol_{lane}"] = ""

    def fill_same_as_above(target_lane):
        source_lane = None
        for i in range(target_lane - 1, 0, -1):
            if st.session_state[f"oil_in_len_{i}"] != "" or st.session_state[f"oil_in_vol_{i}"] != "":
                source_lane = i
                break
        if source_lane:
            src_len = st.session_state[f"oil_in_len_{source_lane}"]
            src_vol = st.session_state[f"oil_in_vol_{source_lane}"]
            for i in range(source_lane + 1, target_lane + 1):
                st.session_state[f"oil_in_len_{i}"] = src_len
                st.session_state[f"oil_in_vol_{i}"] = src_vol

    # 上部コントロールパネル（前回コピー ＆ ④オールクリア）
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        st.button("前回データをコピー (最新履歴)", on_click=copy_latest_oil, use_container_width=True)
    with c_btn2:
        st.button("オールクリア", on_click=clear_all, type="secondary", use_container_width=True)
    
    st.markdown("---")
    
    # ③ 日時をセレクトボックスで選択（0-24時、00-59分）
    st.markdown("<div style='color: silver; font-weight: bold; margin-bottom: 5px;'>日時設定</div>", unsafe_allow_html=True)
    c_date, c_hour, c_min, c_dummy = st.columns([2, 1, 1, 2])
    with c_date:
        st.text_input("日付 (YY/MM/DD)", key="oil_input_date")
    with c_hour:
        hours = [f"{i:02d}" for i in range(24)]
        st.selectbox("時", hours, key="oil_input_hour")
    with c_min:
        minutes = [f"{i:02d}" for i in range(60)]
        st.selectbox("分", minutes, key="oil_input_minute")
        
    render_section_title("各レーンのオイル設定")
    
    # ⑤ 画面を横並び9列でスタイリッシュに見せるための専用CSS
    st.markdown("""
    <style>
    div[data-testid="column"] { padding: 0 4px !important; }
    div[data-testid="column"] input { text-align: center; font-size: 14px !important; padding: 4px !important; }
    div[data-testid="column"] button { padding: 2px 0px !important; font-size: 11px !important; min-height: 26px !important; margin-top: 2px !important; }
    .lane-title { text-align:center; font-weight:900; font-size:16px; color:#c9a44e; margin-bottom:4px; border-bottom: 1px solid #c9a44e; }
    .input-label { text-align:center; font-size:11px; color:silver; margin-bottom:2px; }
    </style>
    """, unsafe_allow_html=True)

    # ⑤ 1〜9レーン、10〜18レーンを横並びにするレイアウト関数
    def render_lane_block(start_lane, end_lane):
        cols = st.columns(9)
        for i in range(start_lane, end_lane + 1):
            idx = i - start_lane
            with cols[idx]:
                st.markdown(f"<div class='lane-title'>{i}L</div>", unsafe_allow_html=True)
                st.markdown("<div class='input-label'>長さ(ft)</div>", unsafe_allow_html=True)
                # ② 単位表記と入力文字数制限（整数2桁）
                st.text_input(f"{i}L長さ", key=f"oil_in_len_{i}", label_visibility="collapsed", placeholder="ft", max_chars=2)
                
                st.markdown("<div class='input-label'>量(ml)</div>", unsafe_allow_html=True)
                # ② 単位表記と入力文字数制限（小数込4桁：例 25.5）
                st.text_input(f"{i}L量", key=f"oil_in_vol_{i}", label_visibility="collapsed", placeholder="ml", max_chars=4)
                
                if i > 1:
                    st.button("←同じ", key=f"btn_same_{i}", on_click=fill_same_as_above, args=(i,), use_container_width=True)
                else:
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # 1Lのズレ防止用
                    
                # ④ 個別のクリアボタン
                st.button("クリア", key=f"btn_clear_{i}", on_click=clear_lane, args=(i,), type="secondary", use_container_width=True)

    # 1〜9レーンの描画
    st.markdown("<div style='background: #1c1c1e; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #333;'>", unsafe_allow_html=True)
    render_lane_block(1, 9)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 10〜18レーンの描画
    st.markdown("<div style='background: #1c1c1e; padding: 15px; border-radius: 8px; border: 1px solid #333;'>", unsafe_allow_html=True)
    render_lane_block(10, 18)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # 登録処理
    if st.button("オイル情報を登録する", type="primary", use_container_width=True):
        error_msg = ""
        oil_time = f"{st.session_state.oil_input_hour}:{st.session_state.oil_input_minute}"
        row_to_add = [st.session_state.oil_input_date, oil_time]
        
        # ② 入力値の取得とバリデーション（長さは整数2桁、量は小数第1位）
        for i in range(1, 19):
            l_val = str(st.session_state[f"oil_in_len_{i}"]).strip()
            v_val = str(st.session_state[f"oil_in_vol_{i}"]).strip()
            
            if l_val:
                try:
                    l_int = int(float(l_val))
                    if not (0 <= l_int <= 99): raise ValueError
                    l_val = str(l_int)
                except:
                    error_msg = f"{i}レーンの「長さ」は2桁までの整数で入力してください。"
                    break
            if v_val:
                try:
                    v_float = float(v_val)
                    v_val = f"{v_float:.1f}"
                except:
                    error_msg = f"{i}レーンの「量」は数値で入力してください。"
                    break
                    
            row_to_add.extend([l_val, v_val])
            
        if error_msg:
            st.error(error_msg)
        else:
            with st.spinner("SPSに登録中..."):
                try:
                    if sh:
                        oil_sheet = sh.worksheet("オイル入力")
                        oil_sheet.append_row(row_to_add)
                        
                        # アプリ内キャッシュも更新
                        if "oil_data" in st.session_state:
                            st.session_state.oil_data.append(row_to_add)
                            
                        st.success("オイル情報の登録が完了しました！")
                    else:
                        st.error("データベースに接続できませんでした。")
                except Exception as e:
                    st.error(f"登録に失敗しました: {e}")
                    
    # ⚠️ オイル入力モードの時は、これより下のコードを読み込まずにストップする
    st.stop()


# =========================================================
# 【新機能】プレイヤー分析ダッシュボード
# =========================================================
if app_mode == "プレイヤー分析":
    import plotly.graph_objects as go
    import plotly.express as px
    import gspread
    import math
    import json
    from google.oauth2 import service_account
    from googleapiclient.discovery import build



    # 🎯 ダーツライブ準拠：レーティング＆フライト計算関数
    def calc_rating_flight(recent_scores):
        if not recent_scores: return 0.0, "UNRATED", 0.0
        
        a = sum(recent_scores) / len(recent_scores)
        
        if a >= 230: rt = 18 + (a - 230) * (3 / 20)
        elif a >= 210: rt = 15 + (a - 210) * (3 / 20)
        elif a >= 190: rt = 12 + (a - 190) * (3 / 20)
        elif a >= 170: rt = 9 + (a - 170) * (3 / 20)
        elif a >= 145: rt = 6 + (a - 145) * (3 / 25)
        elif a >= 95: rt = 1 + (a - 95) * 0.1
        else: rt = a / 95
        
        rt = round(max(1.0, rt), 2)
        
        if rt >= 16: flight = "SA ROLLER"
        elif rt >= 13: flight = "AA ROLLER"
        elif rt >= 10: flight = "A ROLLER"
        elif rt >= 8: flight = "BB ROLLER"
        elif rt >= 6: flight = "B ROLLER"
        elif rt >= 4: flight = "CC ROLLER"
        else: flight = "C ROLLER"
        
        return rt, flight, round(a, 1)

    # SPSからデータを取得（スピナー表示）
    with st.spinner("SPSから最新の分析データを取得中..."):
        try:
            creds_json_str = st.secrets["google_credentials"]
            creds_info = json.loads(creds_json_str, strict=False)
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
            gc = gspread.authorize(creds)
            drive_service = build('drive', 'v3', credentials=creds)

            query = "name = 'EagleBowl_ROLLERS' and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
            results = drive_service.files().list(q=query, fields="files(id, name)").execute()
            sheets = results.get('files', [])
            
            if not sheets:
                st.error("エラー: スプレッドシートが見つかりません。")
                st.stop()
                
            sh = gc.open_by_key(sheets[0]['id'])

            # 🌟【変更】プレイヤーリスト取得と公開・友達設定によるフィルタリング
            settings_data = sh.worksheet("プレイヤー設定").get_all_values()
            
            players = []
            for row in settings_data[1:]:
                if len(row) >= 4 and row[1]:
                    p_email = row[0]
                    p_name = row[1]
                    p_public = row[2]
                    p_friends = row[5] if len(row) > 5 else ""
                    p_friends_list = [f.strip() for f in p_friends.split(",")] if p_friends else []
                    
                    # 管理者・開発者は全員表示。ユーザは条件付き。
                    if st.session_state.user_role in ["開発者", "管理者"]:
                        players.append(p_name)
                    else:
                        if p_name == st.session_state.user_name: # 自分自身
                            players.append(p_name)
                        elif p_public == "公開": # 全体公開
                            players.append(p_name)
                        elif p_public == "友達限定公開": # 相手のF列（友達リスト）に自分がいるか判定
                            if st.session_state.user_email in p_friends_list:
                                players.append(p_name)

            # 先にマスターデータを取得し、全員の現在のレーティングを計算する
            master_data = sh.worksheet("マスター").get_all_values()
            
            player_options = [""]
            player_name_map = {}
            
            for p_name in players:
                p_games = []
                for row in master_data[1:]:
                    if len(row) >= 53 and row[1] == p_name:
                        is_710_game = (len(row) > 54 and str(row[54]).strip().upper() == "TRUE")
                        if not is_710_game:
                            try:
                                score = int(row[52])
                                p_games.append({"date": row[2], "time": row[3], "score": score})
                            except ValueError:
                                pass
                p_games.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
                tmp_recent_50 = [g["score"] for g in p_games[:50]]
                rt_val, _, _ = calc_rating_flight(tmp_recent_50)
                
                # ドロップダウン用の表示名を作成（レーティング数値を付与）
                rt_str = f"{rt_val:.2f}" if rt_val > 0 else "---"
                display_name = f"{p_name}  〔RT: {rt_str}〕"
                player_options.append(display_name)
                # 選択された表示名から、元のプレイヤー名を逆引きできるようにマッピング
                player_name_map[display_name] = p_name

            st.markdown("""
            <style>
            div[data-testid="stSelectbox"] > div > div {
                border: 2px solid #c9a44e !important;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.session_state.get("kiosk_mode"):
                selected_player = st.session_state.kiosk_user
            else:
                # 選択肢をレーティング付きの表示名にする
                selected_display = st.selectbox(" ", player_options, label_visibility="collapsed")
                selected_player = player_name_map.get(selected_display, "")

            # ▼ 追加：プレイヤー未選択時（初期画面）の表示処理 ▼
            if not selected_player:
                st.info("上部のドロップダウンからプレイヤーを選択してください。")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ① お知らせの表示
                announcement = get_announcement_data(sh) if 'get_announcement_data' in globals() else "現在、お知らせはありません。"
                st.markdown("### 📢 お知らせ")
                st.markdown(f'<div style="background-color:#2a2a2e;padding:20px;border-radius:10px;border-left:5px solid #00FFFF;margin-bottom:20px;"><p style="color:white;font-size:16px;white-space:pre-wrap;margin:0;">{announcement}</p></div>', unsafe_allow_html=True)

                st.markdown("<hr style='border:1px solid #444; margin: 30px 0;'>", unsafe_allow_html=True)

                # ② 本日のイベント表示（SPSから読込 ＋ 派手なUI）
                ev_result = get_today_event_from_sps(sh) if 'get_today_event_from_sps' in globals() else ("イベント予定なし", "")
                if isinstance(ev_result, tuple):
                    ev_name, ev_desc = ev_result
                else:
                    ev_name, ev_desc = ev_result, ""
                    
                if ev_name and ev_name != "イベント予定なし":
                    st.markdown("""
                    <style>
                    @keyframes neon { 0%,100% { text-shadow: 0 0 10px #FF107A, 0 0 20px #FF107A; } 50% { text-shadow: 0 0 5px #FF107A, 0 0 10px #FF107A; } }
                    @keyframes bounce { 0%,20%,50%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-10px); } 60% { transform: translateY(-5px); } }
                    .ev-box { background: linear-gradient(145deg, #1a1a1c, #2a1020); border: 2px solid #FF107A; border-radius: 15px; padding: 40px; text-align: center; box-shadow: 0 0 20px rgba(255,16,122,0.4); margin-bottom: 20px; }
                    .ev-main { font-size: 48px; font-weight: 900; color: white; animation: neon 2s infinite; margin: 15px 0; }
                    .ev-desc { font-size: 16px; color: #E0E0E0; margin: 10px 0; padding: 15px; background: rgba(0,0,0,0.5); border-radius: 8px; text-align: left; white-space: pre-wrap; }
                    </style>
                    """, unsafe_allow_html=True)

                    desc_html = f'<div class="ev-desc">{ev_desc}</div>' if ev_desc else ""

                    st.markdown(f"""
                    <div class="ev-box">
                        <p style="color:#FFD700;font-size:20px;font-weight:bold;margin:0;">🎳 TODAY's EVENT 🎳</p>
                        <p class="ev-main">{ev_name}</p>
                        {desc_html}
                        <p style="color:#bbb;font-size:16px;margin-top:20px;">詳細はカレンダーをチェック！</p>
                        <p style="color:#00FFFF;font-size:36px;animation:bounce 2s infinite;margin-top:10px;">☟</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📅 カレンダー原本で詳細を確認する"):
                        try:
                            import datetime
                            import base64
                            now = datetime.datetime.now()
                            f_query = "name = 'イベントスケジュール' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                            folders = drive_service.files().list(q=f_query).execute().get('files', [])
                            if folders:
                                p_query = f"'{folders[0]['id']}' in parents and name contains '{now.month}月' and mimeType = 'application/pdf'"
                                files = drive_service.files().list(q=p_query, fields="files(id)").execute().get('files', [])
                                if files:
                                    pdf_content = drive_service.files().get_media(fileId=files[0]['id']).execute()
                                    base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
                                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                                    st.markdown(pdf_display, unsafe_allow_html=True)
                                else:
                                    st.info("今月のスケジュールPDFが見つかりません。")
                            else:
                                st.info("「イベントスケジュール」フォルダが見つかりません。")
                        except Exception as e:
                            st.error(f"カレンダーの読み込みに失敗しました: {e}")
                else:
                    st.markdown("### 🗓 本日のイベント\n今日はイベントの予定はありません。通常営業でお待ちしております！")
            # ▲ 追加ここまで ▲

            if selected_player:
                # 1. マスターシートから選択されたプレイヤーの「直近50ゲーム」と「7-10G」を抽出
                player_games = []
                player_710_rows = [] 
                for row in master_data[1:]:
                    if len(row) >= 53 and row[1] == selected_player:
                        is_710_game = (len(row) > 54 and str(row[54]).strip().upper() == "TRUE")
                        if is_710_game:
                            player_710_rows.append(row)
                        else:
                            try:
                                score = int(row[52])
                                # ストライク・スペア集計用に "row" も保持するように追加
                                player_games.append({"date": row[2], "time": row[3], "score": score, "row": row})
                            except ValueError:
                                pass
                            
                # 日付・時間で降順ソートし、直近50件を抽出
                player_games.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
                recent_50 = [g["score"] for g in player_games[:50]]
                # 2. レーティング計算
                rt, flight, ave = calc_rating_flight(recent_50)

                # 3. 統計データの取得とタブの作成
                try:
                    award_data = sh.worksheet("AWARD").get_all_values()
                    p_awards = {row[3]: row[6] for row in award_data if len(row) >= 7 and row[1] == selected_player}
                except Exception:
                    p_awards = {}

                # =========================================================
                # ▼▼▼ ダッシュボード表示レイアウト設定 ▼▼▼
                # =========================================================
                # ここのリスト内の項目名（"01_rating_card"など）を別のタブに移動させたり、
                # 順番を上下に入れ替えるだけで、画面の表示順が自動的に変わります。
                # タブの名前（"🏠 HOME"など）も自由に変更・追加・削除可能です。
                dashboard_layout = {
                    "🎳 STATS": [
                        "01_rating_card",
                        "02_score_trend",
                        "14_top10_scores",
                        "16_rating_trend",
                    ],                    
                    "🏆 AWARDS": [                                             
                        "07_high_scores",
                        "08_split_make",
                    ],
                    "📍 MONTHLY": [
                        "15_monthly_stats",
                    ],
                    "📊 ANALYSIS": [
                        "04_first_pitch_pins",
                        "03_seven_ten",                        
                        "05_consecutive",
                        "12_lane_data",
                        "17_env_scatter",
                    ],
                    "🎳 7-10GAME": [
                        "13_seven_ten_game"
                    ]
                }

                # =========================================================
                # ▼▼▼ 各分析項目のコード本体 ▼▼▼
                # （各項目の見た目やロジックは一切変更していません）
                # =========================================================

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【01】 HOME：レーティングバッジ・ステータス
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_01_rating_card():
                    # 通算ストライク率・スペア率の取得
                    all_st_rate = p_awards.get("②1投目ストライク率", "0.0")
                    all_sp_rate = p_awards.get("③2投目スペア率", "0.0")
                    
                    # 通算AVEの計算
                    all_games_scores = [g["score"] for g in player_games]
                    all_ave = round(sum(all_games_scores) / len(all_games_scores), 1) if all_games_scores else 0.0

                    # 直近50ゲームのストライク率・スペア率の計算
                    recent_50_st_chances = 0
                    recent_50_st_success = 0
                    recent_50_sp_chances = 0
                    recent_50_sp_success = 0

                    for g in player_games[:50]:
                        r = g["row"]
                        # 1〜9フレーム
                        for f in range(9):
                            res1 = str(r[10 + f*4]).strip().upper()
                            res2 = str(r[12 + f*4]).strip().upper()
                            
                            recent_50_st_chances += 1
                            if "X" in res1:
                                recent_50_st_success += 1
                            else:
                                recent_50_sp_chances += 1
                                if "/" in res2:
                                    recent_50_sp_success += 1
                                    
                        # 10フレーム
                        res10_1 = str(r[46]).strip().upper() if len(r) > 46 else ""
                        res10_2 = str(r[48]).strip().upper() if len(r) > 48 else ""
                        res10_3 = str(r[50]).strip().upper() if len(r) > 50 else ""
                        
                        recent_50_st_chances += 1
                        if "X" in res10_1:
                            recent_50_st_success += 1
                            recent_50_st_chances += 1
                            if "X" in res10_2:
                                recent_50_st_success += 1
                                recent_50_st_chances += 1
                                if "X" in res10_3:
                                    recent_50_st_success += 1
                            else:
                                recent_50_sp_chances += 1
                                if "/" in res10_3:
                                    recent_50_sp_success += 1
                        else:
                            recent_50_sp_chances += 1
                            if "/" in res10_2:
                                recent_50_sp_success += 1
                                recent_50_st_chances += 1
                                if "X" in res10_3:
                                    recent_50_st_success += 1
                            
                    st_rate = round((recent_50_st_success / recent_50_st_chances) * 100, 1) if recent_50_st_chances > 0 else 0.0
                    sp_rate = round((recent_50_sp_success / recent_50_sp_chances) * 100, 1) if recent_50_sp_chances > 0 else 0.0

                    # ゲージの進捗パーセンテージ計算（MAXレーティングを18と仮定）
                    gauge_pct = min(100, max(0, int((rt / 18.0) * 100)))
                    
                    # 7時(210度)からスタートし、5時(150度)で終わるため、全体の可動域は300度
                    total_deg = 300
                    current_deg = int((gauge_pct / 100) * total_deg)

                    # 水色(#00bcd4) → 緑(#34a853) → 黄色(#fbbc04) → オレンジ(#ff6600) → 赤(#ff3b30) の計算
                    if gauge_pct <= 25:
                        p = gauge_pct / 25
                        r, g, b = int(0 + (52-0)*p), int(188 + (168-188)*p), int(212 + (83-212)*p)
                        conic_bg = f"conic-gradient(from 210deg, #00bcd4 0deg, rgb({r},{g},{b}) {current_deg}deg, #333 {current_deg}deg, #333 300deg, #1a1a1c 300deg, #1a1a1c 360deg)"
                    elif gauge_pct <= 50:
                        p = (gauge_pct - 25) / 25
                        r, g, b = int(52 + (251-52)*p), int(168 + (188-168)*p), int(83 + (4-83)*p)
                        conic_bg = f"conic-gradient(from 210deg, #00bcd4 0deg, #34a853 75deg, rgb({r},{g},{b}) {current_deg}deg, #333 {current_deg}deg, #333 300deg, #1a1a1c 300deg, #1a1a1c 360deg)"
                    elif gauge_pct <= 75:
                        p = (gauge_pct - 50) / 25
                        r, g, b = int(251 + (255-251)*p), int(188 + (102-188)*p), int(4 + (0-4)*p)
                        conic_bg = f"conic-gradient(from 210deg, #00bcd4 0deg, #34a853 75deg, #fbbc04 150deg, rgb({r},{g},{b}) {current_deg}deg, #333 {current_deg}deg, #333 300deg, #1a1a1c 300deg, #1a1a1c 360deg)"
                    else:
                        p = (gauge_pct - 75) / 25
                        r, g, b = int(255 + (255-255)*p), int(102 + (59-102)*p), int(0 + (48-0)*p)
                        conic_bg = f"conic-gradient(from 210deg, #00bcd4 0deg, #34a853 75deg, #fbbc04 150deg, #ff6600 225deg, rgb({r},{g},{b}) {current_deg}deg, #333 {current_deg}deg, #333 300deg, #1a1a1c 300deg, #1a1a1c 360deg)"
                    
                    current_color = f"rgb({r},{g},{b})"

                    # バッジ用の短い称号
                    flight_short = flight.replace(" ROLLER", "")

                    # ダーツライブアプリ風 UIカード (直径1.3倍=325px, 太さ1.2倍のため内径277pxに設定)
                    html_card = f"""
<div style="background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 35px 10px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); border: 1px solid #333; overflow: hidden;">
  <div style="position: relative; width: 325px; height: 325px; margin: 0 auto; border-radius: 50%; background: {conic_bg}; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 25px rgba({r},{g},{b},0.3);">
    <div style="width: 277px; height: 277px; background-color: #1a1a1c; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: inset 0 0 20px rgba(0,0,0,0.9);">
      <div style="color: white; font-size: 80px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif; line-height: 1; text-shadow: 0 0 30px {current_color}, 0 0 60px {current_color};">{rt}</div>
    </div>
  </div>
  <div style="text-align: center; margin-top: -45px; position: relative; z-index: 10; filter: drop-shadow(0 10px 10px rgba(0,0,0,0.8));">
    <div style="display: inline-flex; align-items: flex-start; justify-content: center; box-sizing: border-box; width: 99px; height: 110px; padding-top: 15px; background: linear-gradient(135deg, #bf953f 0%, #fcf6ba 25%, #b38728 50%, #fbf5b7 75%, #aa771c 100%); clip-path: polygon(0% 0%, 100% 0%, 100% 75%, 50% 100%, 0% 75%); border-radius: 2px;">
      <div style="color: {current_color}; font-size: 40px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.5), -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff; letter-spacing: 1px;">{flight_short}</div>
    </div>
  </div>
  
  <div style="text-align: center; margin-top: 40px; margin-bottom: 20px;">
    <div style="color: silver; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(192,192,192,0.6);">LAST 50 GAMES DATA</div>
  </div>
  <div style="display: flex; justify-content: space-around; align-items: center;">
    <div style="text-align: center;">
      <div style="color: #ff3b30; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(255,59,48,0.6);">AVE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{ave}</div>
    </div>
    <div style="text-align: center;">
      <div style="color: #4285f4; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(66,133,244,0.6);">STRIKE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{st_rate}<span style="font-size: 18px;">%</span></div>
    </div>
    <div style="text-align: center;">
      <div style="color: #34a853; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(52,168,83,0.6);">SPARE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{sp_rate}<span style="font-size: 18px;">%</span></div>
    </div>
  </div>

  <hr style="border-top: 1px solid #444; margin: 25px 20px;">

  <div style="text-align: center; margin-bottom: 20px;">
    <div style="color: silver; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(192,192,192,0.6);">ALL DATA</div>
  </div>
  <div style="display: flex; justify-content: space-around; margin-bottom: 10px; align-items: center;">
    <div style="text-align: center;">
      <div style="color: #ff3b30; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(255,59,48,0.6);">AVE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{all_ave}</div>
    </div>
    <div style="text-align: center;">
      <div style="color: #4285f4; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(66,133,244,0.6);">STRIKE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{all_st_rate}<span style="font-size: 18px;">%</span></div>
    </div>
    <div style="text-align: center;">
      <div style="color: #34a853; font-size: 14px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(52,168,83,0.6);">SPARE</div>
      <div style="color: white; font-size: 32px; font-weight: 900; font-family: 'Arial Black', Impact, sans-serif;">{all_sp_rate}<span style="font-size: 18px;">%</span></div>
    </div>
  </div>
</div>
"""
                    st.markdown(html_card, unsafe_allow_html=True)


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【02】 HOME：スコア推移グラフ
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_02_score_trend():
                    if player_games:
                        # 古い順に並び替えて折れ線グラフ化
                        chrono_games = list(reversed(player_games[:50]))
                        
                        # 横軸を「〇ゲーム前（50, 49... 1）」のカウントに変更
                        x_vals = list(range(len(chrono_games), 0, -1))
                        y_vals = [g['score'] for g in chrono_games]

                        # --- 新機能：ストライク・スペア数の集計 ---
                        st_vals = []
                        sp_vals = []
                        for g in chrono_games:
                            r = g.get('row', [])
                            st_count = 0
                            sp_count = 0
                            if r:
                                # 1〜9フレーム
                                for f in range(9):
                                    t1 = str(r[10+f*4]).upper()
                                    t2 = str(r[12+f*4]).upper()
                                    if 'X' in t1: st_count += 1
                                    elif '/' in t2: sp_count += 1
                                
                                # 10フレーム
                                t10_1 = str(r[46]).upper() if len(r)>46 else ""
                                t10_2 = str(r[48]).upper() if len(r)>48 else ""
                                t10_3 = str(r[50]).upper() if len(r)>50 else ""
                                if 'X' in t10_1: st_count += 1
                                if 'X' in t10_2: st_count += 1
                                elif '/' in t10_2: sp_count += 1
                                if 'X' in t10_3: st_count += 1
                                elif '/' in t10_3: sp_count += 1
                            
                            st_vals.append(st_count)
                            sp_vals.append(sp_count)

                        # ▼ 1つ目のグラフ：スコア推移
                        st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 5px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>RECENT 50 GAMES SCORE TREND</div>", unsafe_allow_html=True)

                        fig_trend = px.line(x=x_vals, y=y_vals, markers=True)

                        # アプリ風にオレンジ色のグラフとダークテーマに設定
                        fig_trend.update_traces(line_color='#ff6600', marker=dict(color='#ff6600', size=6, line=dict(color='white', width=1)))
                        fig_trend.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="", range=[50, 0], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=5, color='gray', fixedrange=True),
                            yaxis=dict(title="", range=[0, 300], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=50, color='gray', fixedrange=True),
                            height=300 if st.session_state.get("kiosk_mode") else 280,
                            margin=dict(l=30, r=30, t=10, b=10) # 左右の余白を少し増やして視覚的に中央へ寄せる
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

                        if not st.session_state.get("kiosk_mode"):
                            # ▼ 2つ目のグラフ：ストライク・スペア推移
                            st.markdown("<hr style='border-top: 1px solid #444; margin: 20px 0px;'>", unsafe_allow_html=True)
                            st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 5px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>STRIKES & SPARES TREND</div>", unsafe_allow_html=True)
    
                            fig_st_sp = go.Figure()
                            # ★modeを 'lines' に変更し、markerの設定を削除
                            fig_st_sp.add_trace(go.Scatter(x=x_vals, y=st_vals, mode='lines', name='STRIKE', line=dict(color='#4285f4', width=2)))
                            fig_st_sp.add_trace(go.Scatter(x=x_vals, y=sp_vals, mode='lines', name='SPARE', line=dict(color='#34a853', width=2)))
                            
                            fig_st_sp.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(title="", range=[50, 0], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=5, color='gray', fixedrange=True),
                                yaxis=dict(title="", range=[0, 13], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=2, color='gray', fixedrange=True),
                                height=280,
                                margin=dict(l=30, r=30, t=10, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white'))
                            )
                            st.plotly_chart(fig_st_sp, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("データがありません。")

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【03】 STATS：SEVEN-TEN カバー率
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_03_seven_ten():
                    st.markdown("### <span style='color: silver;'>🎳 SEVEN-TEN カバー率</span>", unsafe_allow_html=True)
                    c_7, c_10 = st.columns(2)
                    
                    # 7番ピンカバー率（ドーナツチャート）
                    rate_7 = float(p_awards.get("④7番ピン", "0"))
                    fig_7 = go.Figure(data=[go.Pie(labels=['Cover', 'Miss'], values=[rate_7, max(0, 100-rate_7)], hole=.7, marker_colors=['#00CC96', '#333333'])])
                    fig_7.update_layout(title_text="7番ピン", title_x=0.5, showlegend=False, margin=dict(t=30, b=10, l=10, r=10), height=200)
                    fig_7.add_annotation(text=f"{rate_7}%", x=0.5, y=0.5, font_size=20, showarrow=False)
                    c_7.plotly_chart(fig_7, use_container_width=True, config={'displayModeBar': False})
                    
                    # 10番ピンカバー率（ドーナツチャート）
                    rate_10 = float(p_awards.get("⑤10番ピン", "0"))
                    fig_10 = go.Figure(data=[go.Pie(labels=['Cover', 'Miss'], values=[rate_10, max(0, 100-rate_10)], hole=.7, marker_colors=['#AB63FA', '#333333'])])
                    fig_10.update_layout(title_text="10番ピン", title_x=0.5, showlegend=False, margin=dict(t=30, b=10, l=10, r=10), height=200)
                    fig_10.add_annotation(text=f"{rate_10}%", x=0.5, y=0.5, font_size=20, showarrow=False)
                    c_10.plotly_chart(fig_10, use_container_width=True, config={'displayModeBar': False})


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【04】 STATS：1投目 残ピン率
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_04_first_pitch_pins():
                    st.markdown("### <span style='color: silver;'>🎳 1投目 残ピン率</span>", unsafe_allow_html=True)
                    
                    #--- 円グラフをHTML/CSSで直接描画する内部関数（既存用） ---
                    def draw_pin_pie(pin_num):
                        rate_str = p_awards.get(f"⑬{pin_num}番ピン残存率", "0")
                        try:
                            rate = float(rate_str)
                        except ValueError:
                            rate = 0.0
                        
                        html = f"""<div style="width: 22%; max-width: 70px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
<div style="position: relative; width: 100%; aspect-ratio: 1 / 1; border-radius: 50%; background: conic-gradient(#EF553B 0% {rate}%, #555555 {rate}% 100%); display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 4px rgba(0,0,0,0.3), 0 2px 5px rgba(0,0,0,0.5); border: 1px solid #222;">
<span style="color: white; font-size: 12px; font-weight: bold; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px black, -1px -1px 2px black, 1px -1px 2px black, -1px 1px 2px black;">{rate}%</span>
</div>
</div>"""
                        return html

                    #--- 円グラフをHTML/CSSで直接描画する内部関数（新規計算用） ---
                    def draw_custom_pin_pie(rate):
                        html = f"""<div style="width: 22%; max-width: 70px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
<div style="position: relative; width: 100%; aspect-ratio: 1 / 1; border-radius: 50%; background: conic-gradient(#EF553B 0% {rate}%, #555555 {rate}% 100%); display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 4px rgba(0,0,0,0.3), 0 2px 5px rgba(0,0,0,0.5); border: 1px solid #222;">
<span style="color: white; font-size: 12px; font-weight: bold; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px black, -1px -1px 2px black, 1px -1px 2px black, -1px 1px 2px black;">{rate:.1f}%</span>
</div>
</div>"""
                        return html

                    # ▼① 今の図（奥・手前の文字を削除）
                    row4 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_pin_pie(7)}{draw_pin_pie(8)}{draw_pin_pie(9)}{draw_pin_pie(10)}</div>"
                    row3 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_pin_pie(4)}{draw_pin_pie(5)}{draw_pin_pie(6)}</div>"
                    row2 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_pin_pie(2)}{draw_pin_pie(3)}</div>"
                    row1 = f"<div style='display: flex; justify-content: center; margin-bottom: 30px;'>{draw_pin_pie(1)}</div>"
                    
                    st.markdown(row4 + row3 + row2 + row1, unsafe_allow_html=True)
                    
                    # ▼ 新機能：ヘッドピン条件別のデータ集計処理
                    head_hit_total = 0
                    head_hit_pins = {i: 0 for i in range(1, 11)}
                    head_miss_total = 0
                    head_miss_pins = {i: 0 for i in range(1, 11)}

                    import re
                    def process_pitch(res_val, pin_val):
                        nonlocal head_hit_total, head_miss_total, head_hit_pins, head_miss_pins
                        res = str(res_val).strip().upper()
                        left_pins = []
                        if "X" not in res: 
                            left_pins = [int(p) for p in re.findall(r'\d+', str(pin_val)) if 1 <= int(p) <= 10]
                        
                        # 1番ピンが残っていない（ヘッドピンが倒れた）
                        if 1 not in left_pins:
                            head_hit_total += 1
                            for p in left_pins:
                                head_hit_pins[p] += 1
                        # 1番ピンが残っている（ヘッドピンを外した）
                        else:
                            head_miss_total += 1
                            for p in left_pins:
                                head_miss_pins[p] += 1

                    # 全ゲームの投球履歴から、すべての「ラックリセット時の1投目」を抽出して集計
                    for g in player_games:
                        r = g['row']
                        # 1〜9フレーム
                        for f in range(9):
                            process_pitch(r[10+f*4], r[11+f*4])
                            
                        # 10フレーム
                        res10_1 = str(r[46]).strip().upper() if len(r) > 46 else ""
                        pin10_1 = str(r[47]).strip() if len(r) > 47 else ""
                        process_pitch(res10_1, pin10_1)
                        
                        # 1投目がストライクなら2投目も新たなラックでの1投目として集計
                        if "X" in res10_1:
                            res10_2 = str(r[48]).strip().upper() if len(r) > 48 else ""
                            pin10_2 = str(r[49]).strip() if len(r) > 49 else ""
                            process_pitch(res10_2, pin10_2)
                            # 2投目もストライクなら3投目も新たなラックでの1投目として集計
                            if "X" in res10_2:
                                res10_3 = str(r[50]).strip().upper() if len(r) > 50 else ""
                                pin10_3 = str(r[51]).strip() if len(r) > 51 else ""
                                process_pitch(res10_3, pin10_3)

                    # 確率を計算する関数
                    def get_hit_rate(p):
                        return (head_hit_pins[p] / head_hit_total * 100) if head_hit_total > 0 else 0.0
                    def get_miss_rate(p):
                        return (head_miss_pins[p] / head_miss_total * 100) if head_miss_total > 0 else 0.0

                    # ▼② ヘッドピンが倒れた場合の残ピン率
                    st.markdown("<hr style='border-top: 1px dashed #444; margin: 20px 0;'>", unsafe_allow_html=True)
                    st.markdown("<div style='color: #aaaaaa; font-size: 12px; text-align: center; margin-bottom: 15px;'>ヘッドピンが倒れた場合の残ピン率</div>", unsafe_allow_html=True)
                    
                    h_row4 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_hit_rate(7))}{draw_custom_pin_pie(get_hit_rate(8))}{draw_custom_pin_pie(get_hit_rate(9))}{draw_custom_pin_pie(get_hit_rate(10))}</div>"
                    h_row3 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_hit_rate(4))}{draw_custom_pin_pie(get_hit_rate(5))}{draw_custom_pin_pie(get_hit_rate(6))}</div>"
                    h_row2 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_hit_rate(2))}{draw_custom_pin_pie(get_hit_rate(3))}</div>"
                    h_row1 = f"<div style='display: flex; justify-content: center; margin-bottom: 30px;'>{draw_custom_pin_pie(get_hit_rate(1))}</div>"
                    
                    st.markdown(h_row4 + h_row3 + h_row2 + h_row1, unsafe_allow_html=True)

                    # ▼③ ヘッドピンを外した場合の残ピン率
                    st.markdown("<hr style='border-top: 1px dashed #444; margin: 20px 0;'>", unsafe_allow_html=True)
                    st.markdown("<div style='color: #aaaaaa; font-size: 12px; text-align: center; margin-bottom: 15px;'>ヘッドピンを外した場合の残ピン率</div>", unsafe_allow_html=True)

                    m_row4 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_miss_rate(7))}{draw_custom_pin_pie(get_miss_rate(8))}{draw_custom_pin_pie(get_miss_rate(9))}{draw_custom_pin_pie(get_miss_rate(10))}</div>"
                    m_row3 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_miss_rate(4))}{draw_custom_pin_pie(get_miss_rate(5))}{draw_custom_pin_pie(get_miss_rate(6))}</div>"
                    m_row2 = f"<div style='display: flex; justify-content: center; gap: 4%; margin-bottom: 12px;'>{draw_custom_pin_pie(get_miss_rate(2))}{draw_custom_pin_pie(get_miss_rate(3))}</div>"
                    m_row1 = f"<div style='display: flex; justify-content: center; margin-bottom: 10px;'>{draw_custom_pin_pie(get_miss_rate(1))}</div>"

                    st.markdown(m_row4 + m_row3 + m_row2 + m_row1, unsafe_allow_html=True)

                
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【05】 STATS：持続力・適応力 分析（50G）
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_05_consecutive():
                    st.markdown("### <span style='color: silver;'>🎳 持続力・適応力 分析（50G）</span>", unsafe_allow_html=True)
                    
                    if not player_games:
                        st.info("データがありません。")
                        return

                    # 最新50ゲームを取得
                    recent_50_games = player_games[:50]

                    # --- 1. ストライク率の計算 ---
                    st_chances = 0
                    strikes = 0
                    chances_after_db = 0
                    st_after_db = 0
                    chances_after_tk = 0
                    st_after_tk = 0

                    for g in recent_50_games:
                        r = g['row']
                        full_rack_shots = []
                        
                        # 1〜9フレーム
                        for f in range(9):
                            t1 = str(r[10+f*4]).strip().upper()
                            full_rack_shots.append('X' if 'X' in t1 else '-')

                        # 10フレーム
                        t10_1 = str(r[46]).strip().upper() if len(r) > 46 else ""
                        t10_2 = str(r[48]).strip().upper() if len(r) > 48 else ""
                        t10_3 = str(r[50]).strip().upper() if len(r) > 50 else ""

                        full_rack_shots.append('X' if 'X' in t10_1 else '-')
                        if 'X' in t10_1:
                            full_rack_shots.append('X' if 'X' in t10_2 else '-')
                            if 'X' in t10_2:
                                full_rack_shots.append('X' if 'X' in t10_3 else '-')
                        elif '/' in t10_2:
                            full_rack_shots.append('X' if 'X' in t10_3 else '-')

                        for i in range(len(full_rack_shots)):
                            st_chances += 1
                            if full_rack_shots[i] == 'X':
                                strikes += 1

                            if i > 1 and full_rack_shots[i-1] == 'X' and full_rack_shots[i-2] == 'X':
                                chances_after_db += 1
                                if full_rack_shots[i] == 'X':
                                    st_after_db += 1

                            if i > 2 and full_rack_shots[i-1] == 'X' and full_rack_shots[i-2] == 'X' and full_rack_shots[i-3] == 'X':
                                chances_after_tk += 1
                                if full_rack_shots[i] == 'X':
                                    st_after_tk += 1

                    st_rate = (strikes / st_chances * 100) if st_chances > 0 else 0
                    db_st_rate = (st_after_db / chances_after_db * 100) if chances_after_db > 0 else 0
                    tk_st_rate = (st_after_tk / chances_after_tk * 100) if chances_after_tk > 0 else 0

                    # --- 2. アベレージの計算 ---
                    scores_50 = [g['score'] for g in recent_50_games]
                    scores_g1 = [g['score'] for g in recent_50_games if str(g['row'][6]).strip().upper().replace("G", "") == '1']
                    scores_g2 = [g['score'] for g in recent_50_games if str(g['row'][6]).strip().upper().replace("G", "") == '2']
                    scores_g3 = [g['score'] for g in recent_50_games if str(g['row'][6]).strip().upper().replace("G", "") == '3']

                    ave_50 = sum(scores_50) / len(scores_50) if scores_50 else 0
                    ave_g1 = sum(scores_g1) / len(scores_g1) if scores_g1 else 0
                    ave_g2 = sum(scores_g2) / len(scores_g2) if scores_g2 else 0
                    ave_g3 = sum(scores_g3) / len(scores_g3) if scores_g3 else 0

                    diff_1_2 = ave_g2 - ave_g1
                    diff_2_3 = ave_g3 - ave_g2

                    # --- UI描画（上下に配置） ---
                    
                    # ▼ 1つ目のグラフ（ストライク持続率）
                    st.markdown("<div style='color: #E2DCC8; font-weight: 900; margin-bottom: 5px; margin-top: 10px; font-size: 16px;'>☕ ストライク持続率</div>", unsafe_allow_html=True)
                    
                    # ご指定の項目名に完全統一
                    labels_st = ['ターキー後の次投ストライク率', 'ダブル後の次投ストライク率', 'ストライク率']
                    values_st = [tk_st_rate, db_st_rate, st_rate]

                    # カフェ風カラー：エスプレッソ、モカ、ラテのグラデーション
                    colors_st = ['#5C4033', '#8B5A2B', '#C19A6B']

                    fig_st = go.Figure(go.Bar(
                        x=values_st,
                        y=labels_st,
                        orientation='h',
                        text=[f"{v:.1f}%" for v in values_st],
                        textposition='inside',
                        marker=dict(color=colors_st)
                    ))
                    fig_st.update_layout(
                        xaxis=dict(range=[0, 80], showgrid=True, gridcolor='#444'),
                        yaxis=dict(showgrid=False, color='silver', tickfont=dict(weight='bold')),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=200
                    )
                    st.plotly_chart(fig_st, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})


                    # ▼ 2つ目のグラフ（レーンアジャスト指数）
                    st.markdown("<div style='color: #E2DCC8; font-weight: 900; margin-bottom: 5px; margin-top: 30px; font-size: 16px;'>🧭 レーンアジャスト指数</div>", unsafe_allow_html=True)
                    
                    # ご指定の項目名に完全統一
                    labels_ave = ['３G目Ave', '２G目Ave', '１G目Ave', '50G Ave']
                    values_ave = [ave_g3, ave_g2, ave_g1, ave_50]
                    
                    # カフェ風カラー：1~3G目は温かみのあるブラウン、50G Aveは落ち着いたグレージュ
                    colors_ave = ['#A07855', '#A07855', '#A07855', '#8C8179']

                    fig_ave = go.Figure(go.Bar(
                        x=values_ave,
                        y=labels_ave,
                        orientation='h',
                        text=[f"{v:.1f}" for v in values_ave],
                        textposition='inside',
                        marker=dict(color=colors_ave)
                    ))

                    max_val = max(values_ave) if max(values_ave) > 0 else 150
                    annot_x = max_val + 40  # アジャスト差分のバッジが綺麗に収まるように余白を確保

                    # 1G -> 2G の差分（カフェ風の抹茶色 or テラコッタ色）
                    color_1_2 = '#6B8E23' if diff_1_2 >= 0 else '#CD5C5C'
                    sign_1_2 = "▲ +" if diff_1_2 > 0 else "▼ "
                    
                    fig_ave.add_annotation(
                        x=annot_x, y=1.5,
                        text=f"<b style='font-size:11px;'>１G ➔ ２G</b><br><b style='font-size:16px;'>{sign_1_2}{diff_1_2:.1f}</b>",
                        showarrow=False,
                        font=dict(color='white', family="Arial"),
                        align="center",
                        bgcolor=color_1_2,
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=6
                    )

                    # 2G -> 3G の差分
                    color_2_3 = '#6B8E23' if diff_2_3 >= 0 else '#CD5C5C'
                    sign_2_3 = "▲ +" if diff_2_3 > 0 else "▼ "
                    
                    fig_ave.add_annotation(
                        x=annot_x, y=0.5,
                        text=f"<b style='font-size:11px;'>２G ➔ ３G</b><br><b style='font-size:16px;'>{sign_2_3}{diff_2_3:.1f}</b>",
                        showarrow=False,
                        font=dict(color='white', family="Arial"),
                        align="center",
                        bgcolor=color_2_3,
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=6
                    )

                    fig_ave.update_layout(
                        xaxis=dict(range=[0, annot_x + 35], showgrid=True, gridcolor='#444'),
                        yaxis=dict(showgrid=False, color='silver', tickfont=dict(weight='bold')),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=250
                    )
                    st.plotly_chart(fig_ave, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【06】 AWARDS：TOTAL & MONTHLY AWARDS
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_06_total_monthly():
                    # --- 🎯 新機能：ダーツライブ風 トータル＆月別アワード集計 ---
                    player_full_games = []
                    for r in master_data[1:]:
                        if len(r) >= 53 and r[1] == selected_player:
                            is_710 = (len(r) > 54 and str(r[54]).strip().upper() == "TRUE")
                            if not is_710:
                                try:
                                    date_str = str(r[2]).strip()
                                    parts = date_str.split('/')
                                    if len(parts) == 3:
                                        yy = int(parts[0])
                                        yyyy = 2000 + yy if yy < 100 else yy
                                        mm = int(parts[1])
                                        dd = int(parts[2])
                                        month_key = f"{yyyy:04d}/{mm:02d}"
                                    else:
                                        continue
                                        
                                    score = int(r[52])
                                    st_count = 0
                                    sp_count = 0
                                    
                                    # 1〜9フレームのストライク・スペア判定
                                    for f in range(9):
                                        t1 = str(r[10+f*4]).upper()
                                        t2 = str(r[12+f*4]).upper()
                                        if 'X' in t1: st_count += 1
                                        elif '/' in t2: sp_count += 1
                                    
                                    # 10フレーム目の判定
                                    t10_1 = str(r[46]).upper() if len(r)>46 else ""
                                    t10_2 = str(r[48]).upper() if len(r)>48 else ""
                                    t10_3 = str(r[50]).upper() if len(r)>50 else ""
                                    if 'X' in t10_1: st_count += 1
                                    if 'X' in t10_2: st_count += 1
                                    elif '/' in t10_2: sp_count += 1
                                    if 'X' in t10_3: st_count += 1
                                    elif '/' in t10_3: sp_count += 1

                                    player_full_games.append({
                                        "month_key": month_key,
                                        "score": score,
                                        "strikes": st_count,
                                        "spares": sp_count,
                                        "sort_key": f"{yyyy:04d}/{mm:02d}/{dd:02d}_{str(r[3]).strip()}_{str(r[6]).strip().zfill(3)}"
                                    })
                                except ValueError:
                                    pass
                                    
                    # 古い順にソート（過去50ゲームのレーティング計算のため）
                    player_full_games.sort(key=lambda x: x["sort_key"])
                    
                    total_g = 0
                    total_st = 0
                    total_sp = 0
                    monthly_stats = {}
                    history_scores = []
                    
                    for g in player_full_games:
                        mk = g["month_key"]
                        if mk not in monthly_stats:
                            monthly_stats[mk] = {"g": 0, "st": 0, "sp": 0, "rt": 0.0}
                        
                        monthly_stats[mk]["g"] += 1
                        monthly_stats[mk]["st"] += g["strikes"]
                        monthly_stats[mk]["sp"] += g["spares"]
                        
                        total_g += 1
                        total_st += g["strikes"]
                        total_sp += g["spares"]
                        
                        history_scores.append(g["score"])
                        # その時点の直近50ゲームを取得
                        recent_50 = history_scores[-50:]
                        rt_val, _, _ = calc_rating_flight(recent_50)
                        # 月が変わるまで上書きされ続けるため、最終的に「その月の最終ゲーム終了時点のレーティング」になる
                        monthly_stats[mk]["rt"] = rt_val
                        
                    # ダーツライブ風 UI描画
                    st.markdown("### 🎳 TOTAL AWARDS")
                    
                    total_html = f"""
                    <div style="background-color: #1a1a1c; border-top: 2px solid #333; border-bottom: 2px solid #333; padding: 15px; margin-bottom: 30px;">
                        <div style="color: #bf953f; font-size: 14px; font-weight: bold; margin-bottom: 10px;">TOTAL</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 8px; margin-bottom: 8px;">
                            <span style="color: white; font-size: 16px;">PLAY COUNT</span>
                            <span style="color: white; font-size: 20px; font-weight: bold;">{total_g}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 8px; margin-bottom: 8px;">
                            <span style="color: white; font-size: 16px;">STRIKE</span>
                            <span style="color: white; font-size: 20px; font-weight: bold;">{total_st}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: white; font-size: 16px;">SPARE</span>
                            <span style="color: white; font-size: 20px; font-weight: bold;">{total_sp}</span>
                        </div>
                    </div>
                    """
                    st.markdown(total_html, unsafe_allow_html=True)
                    
                    st.markdown("### 📅 MONTHLY AWARDS")
                    sorted_months = sorted(monthly_stats.keys(), reverse=True)
                    for mk in sorted_months:
                        m_data = monthly_stats[mk]
                        month_html = f"""
                        <div style="background-color: #2a2a2e; border-radius: 8px; padding: 15px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="color: white; font-size: 18px; font-weight: bold;">{mk}</span>
                                <span style="color: #bf953f; font-size: 14px; font-weight: bold;">RATING <span style="color: white; font-size: 22px; margin-left: 5px;">{m_data['rt']:.2f}</span></span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 5px;">
                                <span style="color: #ccc; font-size: 14px;">PLAY COUNT</span>
                                <span style="color: white; font-size: 16px; font-weight: bold;">{m_data['g']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 5px;">
                                <span style="color: #ccc; font-size: 14px;">STRIKE</span>
                                <span style="color: white; font-size: 16px; font-weight: bold;">{m_data['st']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #ccc; font-size: 14px;">SPARE</span>
                                <span style="color: white; font-size: 16px; font-weight: bold;">{m_data['sp']}</span>
                            </div>
                        </div>
                        """
                        st.markdown(month_html, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【07】 AWARDS：ハイスコア & レコード (ROLLERS RECORD)
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_07_high_scores():
                    if player_games:
                        total_g = len(player_games)
                        score_100 = score_150 = score_200 = score_225 = score_250 = score_275 = score_300 = 0
                        strike_lengths = []
                        nomiss_count = 0

                        for g in player_games:
                            score = g['score']
                            if score >= 100: score_100 += 1
                            if score >= 150: score_150 += 1
                            if score >= 200: score_200 += 1
                            if score >= 225: score_225 += 1
                            if score >= 250: score_250 += 1
                            if score >= 275: score_275 += 1
                            if score == 300: score_300 += 1

                            r = g['row']
                            
                            # ▼ ノーミス判定 (オープンフレームがないかチェック)
                            is_nomiss = True
                            for f in range(9):
                                t1 = str(r[10+f*4]).strip().upper()
                                t2 = str(r[12+f*4]).strip().upper()
                                if 'X' not in t1 and '/' not in t2:
                                    is_nomiss = False
                                    break
                            if is_nomiss:
                                t10_1 = str(r[46]).strip().upper() if len(r) > 46 else ""
                                t10_2 = str(r[48]).strip().upper() if len(r) > 48 else ""
                                if 'X' not in t10_1 and '/' not in t10_2:
                                    is_nomiss = False
                            if is_nomiss:
                                nomiss_count += 1
                                
                            # ▼ 連続ストライク判定
                            seq_len = 0
                            # 1〜9フレーム
                            for f in range(9):
                                t1 = str(r[10+f*4]).strip().upper()
                                if 'X' in t1:
                                    seq_len += 1
                                else:
                                    if seq_len > 0:
                                        strike_lengths.append(seq_len)
                                        seq_len = 0
                            # 10フレーム
                            t10_1 = str(r[46]).strip().upper() if len(r) > 46 else ""
                            t10_2 = str(r[48]).strip().upper() if len(r) > 48 else ""
                            t10_3 = str(r[50]).strip().upper() if len(r) > 50 else ""

                            if 'X' in t10_1:
                                seq_len += 1
                                if 'X' in t10_2:
                                    seq_len += 1
                                    if 'X' in t10_3:
                                        seq_len += 1
                                    else:
                                        if seq_len > 0:
                                            strike_lengths.append(seq_len)
                                            seq_len = 0
                                else:
                                    if seq_len > 0:
                                        strike_lengths.append(seq_len)
                                        seq_len = 0
                            else:
                                if seq_len > 0:
                                    strike_lengths.append(seq_len)
                                    seq_len = 0

                            if seq_len > 0:
                                strike_lengths.append(seq_len)

                        # ストライクが発生した回数（群の数）を母数とする
                        strike_base = len(strike_lengths)

                        # パーセンテージ計算用の内部関数
                        def fmt_pct(count, base):
                            return f"{(count/base*100):.1f}" if base > 0 else "0.0"

                        # 画面表示用のHTML（ダークコンテナUI）
                        html = f"""<div style="background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); border: 1px solid #333; margin-bottom: 20px;">
<div style='color: silver; font-weight: 900; margin-bottom: 20px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>ROLLERS RECORD</div>
<div style="color: #bf953f; font-weight: 900; font-size: 16px; letter-spacing: 1px; margin-bottom: 12px; border-bottom: 2px solid #444; padding-bottom: 6px; display: flex; align-items: center;">
<span style="font-size: 20px; margin-right: 8px;">🎳</span> TOTAL SCORE ACHIEVEMENTS
</div>
<div style="margin-left: 5px; margin-bottom: 25px;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">100 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_100}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_100, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">150 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_150}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_150, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">200 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_200}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_200, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">225 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_225}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_225, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">250 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_250}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_250, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #444; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">275 UP</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_275}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_275, total_g)}％)</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0;">
<span style="color: #ff3b30; font-size: 14px; font-weight: bold;">PERFECT 300</span>
<span style="color: white; font-size: 14px;"><span style="color: #ff6600; font-weight: bold; font-size: 16px;">{score_300}</span> G <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(score_300, total_g)}％)</span></span>
</div>
</div>
<div style="color: #bf953f; font-weight: 900; font-size: 16px; letter-spacing: 1px; margin-bottom: 12px; border-bottom: 2px solid #444; padding-bottom: 6px; display: flex; align-items: center;">
<span style="font-size: 20px; margin-right: 8px;">🔥</span> CONSECUTIVE STRIKES
</div>
<div style="margin-left: 5px; margin-bottom: 25px;">"""
                        
                        for i in range(2, 13):
                            cnt = len([l for l in strike_lengths if l >= i])
                            label_str = ""
                            if i == 2: label_str = " (DOUBLE)"
                            elif i == 3: label_str = " (TURKEY)"
                            elif i == 12: label_str = " (PERFECT)"
                            
                            border_style = "border-bottom: 1px dashed #444;" if i < 12 else ""
                            
                            html += f"""<div style="display: flex; justify-content: space-between; align-items: center; {border_style} padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">連続 {i}回{label_str}</span>
<span style="color: white; font-size: 14px;"><span style="color: #4285f4; font-weight: bold; font-size: 16px;">{cnt}</span> 回 <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(cnt, strike_base)}％)</span></span>
</div>"""

                        html += f"""</div>
<div style="color: #bf953f; font-weight: 900; font-size: 16px; letter-spacing: 1px; margin-bottom: 12px; border-bottom: 2px solid #444; padding-bottom: 6px; display: flex; align-items: center;">
<span style="font-size: 20px; margin-right: 8px;">✨</span> NO-MISS GAMES
</div>
<div style="margin-left: 5px;">
<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0;">
<span style="color: silver; font-size: 14px; font-weight: bold;">ノーミスゲーム</span>
<span style="color: white; font-size: 14px;"><span style="color: #34a853; font-weight: bold; font-size: 16px;">{nomiss_count}</span> 回 <span style="color: gray; font-size: 12px; margin-left: 5px;">({fmt_pct(nomiss_count, total_g)}％)</span></span>
</div>
</div>
</div>"""
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.info("データがありません。")
                




                
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【08】 AWARDS：スプリット・メイク
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_08_split_make():
                    st.markdown("### <span style='color: silver;'>🎳 SPLIT MAKE DATA</span>", unsafe_allow_html=True)
                    
                    split_records = []
                    
                    for row in award_data:
                        if len(row) >= 7 and row[1] == selected_player and "⑥" in row[3]:
                            name_part = row[3].replace("⑥", "")
                            # name_part は "スネークアイ (7-10)" のような形式なので、名前とピン配置に分割
                            if " (" in name_part and ")" in name_part:
                                s_name = name_part.split(" (")[0]
                                s_pins = name_part.split(" (")[1].replace(")", "")
                            else:
                                s_name = name_part
                                s_pins = ""
                                
                            try:
                                chances = int(row[4])
                                success = int(row[5])
                                rate = float(row[6])
                            except ValueError:
                                continue
                            
                            if s_name != "Others":
                                split_records.append({"name": s_name, "pins": s_pins, "chances": chances, "success": success, "rate": rate})
                    
                    if not split_records:
                        st.info("スプリットの記録がありません。")
                        return

                    def get_split_tier(split_name):
                        if any(x in split_name for x in ["ベビースプリット", "ダイムストア", "2ピン"]):
                            return 1 # 簡単（緑）
                        elif any(x in split_name for x in ["リリー", "ビッグディボット", "フォーシックス", "クリスマスツリー", "ムース", "3ピン"]):
                            return 2 # 中くらい（黄）
                        elif any(x in split_name for x in ["スネークアイ", "ビッグフォー", "グリークチャーチ", "ワシントン条約", "マイティマイト", "4ピン", "5ピン"]):
                            return 3 # 難しい（赤オレンジ）
                        else:
                            return 2

                    split_records.sort(key=lambda x: get_split_tier(x["name"]))

                    def get_split_color(split_name):
                        if get_split_tier(split_name) == 3:
                            return "#ff5722" 
                        elif get_split_tier(split_name) == 2:
                            return "#fbbc04" 
                        elif get_split_tier(split_name) == 1:
                            return "#34a853" 
                        else:
                            return "white"

                    # ツールチップ内でピン配置図を描画する関数
                    def draw_tooltip_pins(pins_str):
                        if not pins_str: return ""
                        active_pins = pins_str.split("-")
                        
                        def pin_html(p_num):
                            bg_color = "#ff2d55" if str(p_num) in active_pins else "#333333"
                            border_color = "#ffaaaa" if str(p_num) in active_pins else "#555555"
                            return f'<div style="width: 14px; height: 14px; border-radius: 50%; background-color: {bg_color}; border: 1px solid {border_color}; margin: 2px;"></div>'

                        html = f"""<div style="display: flex; flex-direction: column; align-items: center; background-color: #1a1a1c; padding: 10px; border-radius: 8px; border: 1px solid #444;">
<div style="color: silver; font-size: 10px; margin-bottom: 5px; font-weight: bold;">{pins_str}</div>
<div style="display: flex; justify-content: center;">{pin_html(7)}{pin_html(8)}{pin_html(9)}{pin_html(10)}</div>
<div style="display: flex; justify-content: center;">{pin_html(4)}{pin_html(5)}{pin_html(6)}</div>
<div style="display: flex; justify-content: center;">{pin_html(2)}{pin_html(3)}</div>
<div style="display: flex; justify-content: center;">{pin_html(1)}</div>
</div>"""
                        return html

                    # CSSでツールチップの動きを定義
                    html = """<style>
.tooltip-container { position: relative; display: inline-block; cursor: pointer; margin-left: 5px; }
.tooltip-container .tooltip-content { visibility: hidden; width: max-content; background-color: transparent; text-align: center; border-radius: 8px; position: absolute; z-index: 100; bottom: 125%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s; pointer-events: none; }
.tooltip-container:hover .tooltip-content, .tooltip-container:active .tooltip-content { visibility: visible; opacity: 1; }
</style>
<div style="background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); border: 1px solid #333; margin-bottom: 20px;">
<table style="width: 100%; text-align: left; border-collapse: collapse;">
<thead>
<tr style="border-bottom: 1px solid #555; color: silver; font-size: 12px;">
<th style="padding: 5px; font-weight: normal;">スプリット名称</th>
<th style="padding: 5px; text-align: center; font-weight: normal;">遭遇</th>
<th style="padding: 5px; text-align: center; font-weight: normal;">メイク</th>
<th style="padding: 5px; text-align: right; font-weight: normal;">メイク率</th>
</tr>
</thead>
<tbody>"""
                    
                    for rec in split_records:
                        color = get_split_color(rec["name"])
                        tooltip_html = draw_tooltip_pins(rec['pins'])
                        # ★ 名前とピン番号を結合して表示用の文字列を作成
                        display_name = f"{rec['name']} ({rec['pins']})" if rec['pins'] else rec['name']
                        
                        html += f"""
<tr style="border-bottom: 1px dashed #444;">
<td style="padding: 7px 5px; color: {color}; font-weight: bold; font-size: 14px; display: flex; align-items: center;">
{display_name}
<div class="tooltip-container">
<span style="filter: hue-rotate(300deg) saturate(200%); font-size: 14px;">🎳</span>
<div class="tooltip-content">{tooltip_html}</div>
</div>
</td>
<td style="padding: 7px 5px; color: white; text-align: center; font-size: 14px;">{rec['chances']}</td>
<td style="padding: 7px 5px; color: white; text-align: center; font-size: 14px;">{rec['success']}</td>
<td style="padding: 7px 5px; color: white; text-align: right; font-size: 14px; font-weight: bold;">{rec['rate']:.1f}%</td>
</tr>"""
                        
                    html += """
</tbody>
</table>
</div>"""
                    
                    st.markdown(html, unsafe_allow_html=True)
                

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【09】 ENVIRONMENT：投球方式 適性
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_09_play_style():
                    st.markdown("### <span style='color: silver;'>🎳 投球方式 適性</span>", unsafe_allow_html=True)
                    euro_ave = float(p_awards.get("⑨1レーン", "0"))
                    am_ave = float(p_awards.get("⑨2レーン", "0"))
                    fig_style = px.bar(
                        x=["ヨーロピアン (1レーン)", "アメリカン (2レーン)"], 
                        y=[euro_ave, am_ave],
                        labels={"x": "投球方式", "y": "アベレージ"},
                        text=[f"{euro_ave}", f"{am_ave}"],
                        color_discrete_sequence=['#FFA15A']
                    )
                    fig_style.update_traces(textposition='outside')
                    fig_style.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(range=[0, max(euro_ave, am_ave, 150) * 1.2]))
                    st.plotly_chart(fig_style, use_container_width=True)


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【10】 ENVIRONMENT：オイル長 適性
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_10_oil_length():
                    st.markdown("### <span style='color: silver;'>📏 オイル長 (Length) 適性</span>", unsafe_allow_html=True)
                    len_keys, len_aves = [], []
                    for row in award_data:
                        if len(row) >= 7 and row[1] == selected_player and "⑪" in row[3]:
                            try:
                                if float(row[4]) > 0: # プレイ回数が1回以上のものだけ抽出
                                    len_keys.append(row[3].replace("⑪", ""))
                                    len_aves.append(float(row[6]))
                            except ValueError:
                                pass
                    if len_keys:
                        fig_len = px.line(x=len_keys, y=len_aves, markers=True, labels={"x": "オイル長 (ft)", "y": "アベレージ"})
                        fig_len.update_traces(line_color='#00CC96')
                        fig_len.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(fig_len, use_container_width=True)
                    else:
                        st.info("オイル長のプレイデータがありません。")


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【11】 ENVIRONMENT：オイル量 適性
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_11_oil_volume():
                    st.markdown("### <span style='color: silver;'>💧 オイル量 (Volume) 適性</span>", unsafe_allow_html=True)
                    vol_keys, vol_aves = [], []
                    for row in award_data:
                        if len(row) >= 7 and row[1] == selected_player and "⑫" in row[3]:
                            try:
                                if float(row[4]) > 0: # プレイ回数が1回以上のものだけ抽出
                                    vol_keys.append(row[3].replace("⑫", ""))
                                    vol_aves.append(float(row[6]))
                            except ValueError:
                                pass
                    if vol_keys:
                        fig_vol = px.line(x=vol_keys, y=vol_aves, markers=True, labels={"x": "オイル量 (ml)", "y": "アベレージ"})
                        fig_vol.update_traces(line_color='#AB63FA')
                        fig_vol.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(fig_vol, use_container_width=True)
                    else:
                        st.info("オイル量のプレイデータがありません。")


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【12】 STATS：レーン相性分析
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_12_lane_data():
                    st.markdown("<div style='color: #E2DCC8; font-weight: 900; margin-bottom: 15px; margin-top: 10px; font-size: 16px;'>🧭 レーン相性分析</div>", unsafe_allow_html=True)
                    
                    if not player_games:
                        st.info("データがありません。")
                        return

                    import plotly.graph_objects as go

                    # 横軸のレーンリスト（1, 2, 1-2, 3, 4, 3-4 ... 17-18 まで固定で生成）
                    target_lanes = []
                    for i in range(1, 18, 2):
                        target_lanes.extend([str(i), str(i+1), f"{i}-{i+1}"])

                    # レーンごとのスコアを格納する辞書
                    lane_scores = {lane: [] for lane in target_lanes}

                    for g in player_games:
                        try:
                            # マスターデータの「レーン」列はインデックス 5
                            lane_raw = str(g['row'][5]).strip().upper()
                        except:
                            continue
                        
                        if not lane_raw:
                            continue

                        # "2-1" などを "1-2" に自動統合
                        if "-" in lane_raw:
                            parts = lane_raw.split("-")
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                p1, p2 = int(parts[0]), int(parts[1])
                                lane_val = f"{min(p1, p2)}-{max(p1, p2)}"
                            else:
                                lane_val = lane_raw
                        else:
                            lane_val = lane_raw

                        if lane_val in lane_scores:
                            try:
                                score = int(g['score'])
                                lane_scores[lane_val].append(score)
                            except:
                                pass

                    # 各レーンの「最新50G」のアベレージとゲーム数を計算
                    averages = []
                    game_counts = []
                    for lane in target_lanes:
                        recent_50_scores = lane_scores[lane][:50]
                        count = len(recent_50_scores)
                        game_counts.append(count)
                        if count > 0:
                            avg = sum(recent_50_scores) / count
                            averages.append(avg)
                        else:
                            averages.append(0)

                    # 色分け（ヨーロピアン: カフェブラウン, アメリカン: マスタードゴールド）
                    colors = []
                    tick_texts = []
                    for lane in target_lanes:
                        if "-" in lane:
                            colors.append("#D4AF37")  # アメリカン
                            tick_texts.append(f"<b style='color: #D4AF37;'>{lane}</b>")
                        else:
                            colors.append("#A07855")  # ヨーロピアン
                            tick_texts.append(f"<b style='color: #A07855;'>{lane}</b>")

                    st.markdown("<hr style='border-top: 1px solid #444; margin: 20px 0px;'>", unsafe_allow_html=True)
                    st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 5px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>LANE AFFINITY (RECENT 50G AVE)</div>", unsafe_allow_html=True)

                    bar_texts = []
                    for val, cnt in zip(averages, game_counts):
                        if cnt > 0:
                            # HTMLタグで太字化を維持
                            bar_texts.append(f"<b>{val:.1f}({cnt}G)</b>")
                        else:
                            bar_texts.append("")

                    fig = go.Figure(go.Bar(
                        x=target_lanes,
                        y=averages,
                        marker=dict(color=colors),
                        text=bar_texts,
                        textposition='outside',
                        textangle=-90,
                        textfont=dict(size=12, color='#cccccc'), # ★ 文字色を白(white)から薄いグレー(#cccccc)に変更
                        cliponaxis=False
                    ))

                    # グラフを隙間なく詰めて、スクロールなしで1画面に収める設定
                    fig.update_layout(
                        uniformtext=dict(minsize=12, mode='show'),
                        bargap=0.15,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            type='category',
                            categoryorder='array',
                            categoryarray=target_lanes,
                            tickmode='array',
                            tickvals=target_lanes,
                            ticktext=tick_texts,
                            tickangle=-90,
                            showgrid=False,
                            fixedrange=True,
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            range=[0, 360],
                            color='silver',
                            gridcolor='#444',
                            fixedrange=True,
                            tickmode='array',
                            tickvals=[0, 50, 100, 150, 200, 250, 300]
                        ),
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【13】 7-10G：7-10 GAME 分析
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_13_seven_ten_game():
                    # ★ タイトルをシルバーに統一し、ダーツから2本のピン(🎳🎳)のイメージに変更
                    st.markdown("### <span style='color: silver;'>🎳 7-10 GAME 分析</span>", unsafe_allow_html=True)
                    
                    if not player_710_rows:
                        st.info("7-10 GAME のプレイデータがありません。")
                        return

                    g_count = len(player_710_rows)
                    f_count = 0
                    success_c = 0
                    nearpin_c = 0
                    ponkotsu_c = 0

                    def val(t, t_prev=0):
                        t_str = str(t).strip().upper()
                        if t_str == 'X': return 10
                        if t_str == '/': return 10 - t_prev
                        if t_str in ['G', '-', '']: return 0
                        try: return int(t_str)
                        except: return 0

                    scores_710 = []

                    for r in player_710_rows:
                        try:
                            s = int(r[52])
                            scores_710.append({"date": r[2], "score": s})
                        except:
                            pass
                            
                        for f in range(10):
                            f_count += 1
                            if f < 9:
                                t1 = r[10 + f*4]
                                t2 = r[12 + f*4]
                            else:
                                t1 = r[46]
                                t2 = r[48]
                            
                            v1 = val(t1)
                            v2 = val(t2, v1)
                            
                            if v1 == 1 and v2 == 1:
                                success_c += 1
                            elif (v1 == 1 and v2 == 2) or (v1 == 2 and v2 == 1):
                                nearpin_c += 1
                            elif (v1 + v2) >= 8:
                                ponkotsu_c += 1
                    
                    def fmt_pct(num, den):
                        return f"{(num/den)*100:.1f}%" if den > 0 else "0.0%"
                    
                    scores_710.sort(key=lambda x: x["score"])

                    # ＝＝＝ UI描画 ＝＝＝

                    # ① サマリーリボン
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-around; background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);'>
                        <div style='text-align: center;'>
                            <div style='color: #A07855; font-size: 13px; font-weight: bold;'>プレイゲーム数</div>
                            <div style='color: white; font-size: 28px; font-weight: bold;'>{g_count}<span style='font-size: 16px; color: silver;'> G</span></div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='color: #A07855; font-size: 13px; font-weight: bold;'>総チャレンジ数 (フレーム)</div>
                            <div style='color: white; font-size: 28px; font-weight: bold;'>{f_count}<span style='font-size: 16px; color: silver;'> 回</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ② 詳細カード（横並びから、おしゃれな配色のまま縦に3列並べるレイアウトに変更）
                    st.markdown(f"""
                    <div style='display: flex; flex-direction: column; gap: 12px; margin-bottom: 25px;'>
                        <div style='background: rgba(74, 144, 226, 0.1); border-left: 5px solid #4A90E2; padding: 12px 15px; border-radius: 5px;'>
                            <div style='color: #4A90E2; font-size: 13px; font-weight: bold;'>🏆 成功 (1本 - 1本)</div>
                            <div style='color: white; font-size: 22px; font-weight: bold; margin-top: 5px;'>{success_c} <span style='font-size: 14px; color: silver;'>回</span> <span style='float: right; color: #4A90E2;'>{fmt_pct(success_c, f_count)}</span></div>
                        </div>
                        <div style='background: rgba(138, 180, 248, 0.1); border-left: 5px solid #8AB4F8; padding: 12px 15px; border-radius: 5px;'>
                            <div style='color: #8AB4F8; font-size: 13px; font-weight: bold;'>惜しい! ニアピン（1本-2本、2本- 1本）</div>
                            <div style='color: white; font-size: 22px; font-weight: bold; margin-top: 5px;'>{nearpin_c} <span style='font-size: 14px; color: silver;'>回</span> <span style='float: right; color: #8AB4F8;'>{fmt_pct(nearpin_c, f_count)}</span></div>
                        </div>
                        <div style='background: rgba(224, 102, 102, 0.1); border-left: 5px solid #E06666; padding: 12px 15px; border-radius: 5px;'>
                            <div style='color: #E06666; font-size: 13px; font-weight: bold;'>💥 ポンコツ (計8本以上)</div>
                            <div style='color: white; font-size: 22px; font-weight: bold; margin-top: 5px;'>{ponkotsu_c} <span style='font-size: 14px; color: silver;'>回</span> <span style='float: right; color: #E06666;'>{fmt_pct(ponkotsu_c, f_count)}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ③ MINI SCORE RANKING
                    st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 10px; margin-top: 25px; font-size: 15px;'>📉 LOWER SCORE TOP3</div>", unsafe_allow_html=True)
                    
                    def get_mini_html(rank, color, score_data):
                        if not score_data:
                            return f"<div style='flex: 1; background: #1e1e1e; border-top: 3px solid #444; padding: 12px; text-align: center; border-radius: 6px;'><div style='color: gray; font-size: 12px; font-weight: bold;'>LOWER {rank}</div><div style='color: #555; font-size: 20px; font-weight: bold; margin: 5px 0;'>-</div><div style='color: transparent; font-size: 11px;'>-</div></div>"
                        return f"<div style='flex: 1; background: #2a2a2e; border-top: 3px solid {color}; padding: 12px; text-align: center; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'><div style='color: silver; font-size: 12px; font-weight: bold;'>LOWER {rank}</div><div style='color: white; font-size: 22px; font-weight: bold; margin: 5px 0;'>{score_data['score']}<span style='font-size: 12px; color: gray;'> 点</span></div><div style='color: #A07855; font-size: 11px;'>{score_data['date']}</div></div>"

                    m1 = get_mini_html(1, '#E06666', scores_710[0] if len(scores_710) > 0 else None)
                    m2 = get_mini_html(2, '#D4AF37', scores_710[1] if len(scores_710) > 1 else None)
                    m3 = get_mini_html(3, '#A07855', scores_710[2] if len(scores_710) > 2 else None)

                    st.markdown(f"<div style='display: flex; gap: 15px; margin-bottom: 25px;'>{m1}{m2}{m3}</div>", unsafe_allow_html=True)

                    # ④ 用語解説エリア
                    st.markdown("""
                    <div style='background: #1c1c1e; padding: 15px; border-radius: 8px; border: 1px dashed #555;'>
                        <div style='color: #A07855; font-weight: bold; font-size: 13px; margin-bottom: 10px;'>📖 7-10 GAME 用語解説</div>
                        <div style='font-size: 11px; color: #aaa; line-height: 1.6; gap: 10px;'>
                            <div>
                                <b>① 7-10ゲーム数：</b>7-10 GAMEとして登録された総G数<br>
                                <b>② 7-10総チャレンジ数：</b>プレイした全フレーム数<br>
                                <b>③ 7-10成功数：</b>1投目も2投目もスコアが「1」だった回数<br>
                                <b>④ 7-10成功率：</b>チャレンジ数に対する成功数の割合<br>
                                <b>⑤ 7-10ニアピン数：</b>1投目「1」で2投目「2」（逆も含む）の回数<br>
                                <b>⑥ 7-10ニアピン率：</b>チャレンジ数に対するニアピンの割合<br>
                                <b>⑦ 7-10ポンコツ数：</b>2投の合計スコアが8本以上の回数<br>
                                <b>⑧ 7-10ポンコツ率：</b>チャレンジ数に対するポンコツの割合<br>
                                <b>⑨ MINI_1~3：</b>過去の7-10 GAMEで最も低かったスコア（1位〜3位）
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【14】 AWARDS：歴代スコアトップ10
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_14_top10_scores():
                    if player_games:
                        from collections import Counter
                        
                        # 全ゲームのスコアを取得
                        all_scores = [g['score'] for g in player_games]
                        
                        # スコアの出現回数をカウント
                        score_counts = Counter(all_scores)
                        
                        # スコアを降順（高い順）にソートしてトップ10種類を抽出
                        sorted_unique_scores = sorted(score_counts.keys(), reverse=True)
                        top_10_scores = sorted_unique_scores[:10]
                        
                        y_vals = []
                        x_vals = []
                        text_vals = []
                        
                        for i, score in enumerate(top_10_scores):
                            y_vals.append(f"{i+1}位 ")
                            x_vals.append(score)
                            count = score_counts[score]
                            if count > 1:
                                text_vals.append(f"{score} ({count}回)")
                            else:
                                text_vals.append(f"{score}")

                        # ★ 空のボックス（ダークコンテナ）を削除し、グラフ間と同じグレーの横線を挿入
                        st.markdown("<hr style='border-top: 1px solid #444; margin: 20px 0px;'>", unsafe_allow_html=True)
                        st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 5px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>ALL-TIME TOP 10 SCORES</div>", unsafe_allow_html=True)

                        # 横向きの棒グラフ作成
                        fig_top10 = px.bar(x=x_vals, y=y_vals, orientation='h', text=text_vals)

                        fig_top10.update_traces(
                            marker_color='#ff6600', 
                            textposition='inside',
                            insidetextanchor='middle',
                            textfont=dict(color='white', size=14, family='Arial Black')
                        )
                        
                        fig_top10.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="", range=[0, 300], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=50, color='gray', fixedrange=True),
                            yaxis=dict(title="", autorange="reversed", color='silver', tickfont=dict(size=14, family='Arial, sans-serif'), fixedrange=True),
                            height=350,
                            margin=dict(l=40, r=30, t=10, b=10)
                        )
                        
                        st.plotly_chart(fig_top10, use_container_width=True, config={'displayModeBar': False})
                        # ★ 末尾にあった </div> の閉じタグも削除済み
                    else:
                        st.info("データがありません。")

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【16】 STATS：レーティング推移（直近50ヶ月）
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_16_rating_trend():
                    if player_games:
                        import plotly.graph_objects as go
                        
                        # 古い順のゲームリスト
                        chrono_games = list(reversed(player_games))
                        
                        monthly_rt = {}
                        history_scores = []
                        
                        for g in chrono_games:
                            date_str = str(g["date"]).strip()
                            parts = date_str.split('/')
                            if len(parts) >= 2:
                                yy = int(parts[0])
                                yyyy = 2000 + yy if yy < 100 else yy
                                mm = int(parts[1])
                                month_key = f"{yyyy:04d}/{mm:02d}"
                            else:
                                continue
                                
                            history_scores.append(g["score"])
                            recent_50 = history_scores[-50:]
                            rt_val, _, _ = calc_rating_flight(recent_50)
                            
                            # 月が変わるまで上書きされ続けるため、その月の最終レーティングになる
                            monthly_rt[month_key] = rt_val
                            
                        # 月別レーティングを新しい順にソートして、最大50ヶ月分を取得
                        sorted_months = sorted(monthly_rt.keys(), reverse=True)
                        recent_50_months = sorted_months[:50]
                        
                        # グラフ描画用に、古い順（左から右へ）に戻す
                        recent_50_months.reverse()
                        
                        y_vals = [monthly_rt[m] for m in recent_50_months]
                        
                        # 横軸（ヶ月前）。最新月を0とする。
                        num_months = len(recent_50_months)
                        x_vals = list(range(num_months - 1, -1, -1))
                        
                        # 空欄の元凶だった背景付きdivを削除し、横線のみでスマートに区切る
                        st.markdown("<hr style='border-top: 1px solid #444; margin: 20px 0px;'>", unsafe_allow_html=True)
                        st.markdown("<div style='color: silver; font-weight: 900; margin-bottom: 5px; font-size: 16px; font-family: Arial, sans-serif; text-align: center;'>RECENT 50 MONTHS RATING TREND</div>", unsafe_allow_html=True)

                        fig_rt = go.Figure()

                        # ★ HOME画面のレーティングカードと【全く同じグラデーション計算】を行う関数
                        def get_rt_color(rt):
                            gauge_pct = min(100, max(0, int((rt / 18.0) * 100)))
                            if gauge_pct <= 25:
                                p = gauge_pct / 25
                                r, g, b = int(0 + (52-0)*p), int(188 + (168-188)*p), int(212 + (83-212)*p)
                            elif gauge_pct <= 50:
                                p = (gauge_pct - 25) / 25
                                r, g, b = int(52 + (251-52)*p), int(168 + (188-168)*p), int(83 + (4-83)*p)
                            elif gauge_pct <= 75:
                                p = (gauge_pct - 50) / 25
                                r, g, b = int(251 + (255-251)*p), int(188 + (102-188)*p), int(4 + (0-4)*p)
                            else:
                                p = (gauge_pct - 75) / 25
                                r, g, b = int(255 + (255-255)*p), int(102 + (59-102)*p), int(0 + (48-0)*p)
                            return f"rgb({r},{g},{b})"

                        # 線分ごとに、その月のレーティングカラーをHOME画面と完全一致させて描画
                        for i in range(len(x_vals) - 1):
                            x0, x1 = x_vals[i], x_vals[i+1]
                            y0, y1 = y_vals[i], y_vals[i+1]
                            
                            color = get_rt_color(y1)
                            
                            fig_rt.add_trace(go.Scatter(
                                x=[x0, x1],
                                y=[y0, y1],
                                mode='lines',  # ★ 点を消して線のみ
                                line=dict(color=color, width=3),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                        # タップ時に数値を表示するための透明なカバーレイヤー
                        fig_rt.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines',
                            line=dict(color='rgba(0,0,0,0)', width=0),
                            showlegend=False,
                            hovertemplate="<b>%{x}ヶ月前</b><br>Rt: %{y:.2f}<extra></extra>"
                        ))

                        fig_rt.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="（ヶ月前）", range=[50, 0], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=5, color='gray', fixedrange=True),
                            yaxis=dict(title="Rt", range=[0.0, 18.0], showgrid=True, gridcolor='#444', tickmode='linear', tick0=0, dtick=2, color='gray', fixedrange=True),
                            height=300 if st.session_state.get("kiosk_mode") else 280,
                            margin=dict(l=30, r=30, t=10, b=10)
                        )
                        st.plotly_chart(fig_rt, use_container_width=True, config={'displayModeBar': False})

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【17】 ANALYSIS：オイル長さ・オイル量 相性分析（点グラフ）
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_17_env_scatter():
                    st.markdown("<hr style='border-top: 1px solid #444; margin: 30px 0px 20px 0px;'>", unsafe_allow_html=True)
                    # ★ タイトルを変更
                    st.markdown("### <span style='color: #E2DCC8;'>🌍 オイル長さ・オイル量 相性分析</span>", unsafe_allow_html=True)
                    
                    if not player_games:
                        st.info("データがありません。")
                        return

                    import plotly.graph_objects as go

                    len_x = []
                    len_y = []
                    vol_x = []
                    vol_y = []

                    # データの抽出とフィルタリング
                    for g in player_games:
                        score = g.get('score', 0)
                        row = g.get('row', [])
                        
                        # オイル長はインデックス 7、オイル量はインデックス 8 に変更
                        if len(row) > 7:
                            try:
                                l_str = str(row[7]).replace('ft', '').replace('フィート', '').strip()
                                if l_str:
                                    l_val = float(l_str)
                                    if 30 <= l_val <= 45:
                                        len_x.append(l_val)
                                        len_y.append(score)
                            except:
                                pass
                                
                        if len(row) > 8:
                            try:
                                v_str = str(row[8]).replace('ml', '').replace('cc', '').strip()
                                if v_str:
                                    v_val = float(v_str)
                                    if 20 <= v_val <= 35:
                                        vol_x.append(v_val)
                                        vol_y.append(score)
                            except:
                                pass

                    # ▼ 1つ目のグラフ：オイル長さ 適性
                    fig_len = go.Figure()
                    fig_len.add_trace(go.Scatter(
                        x=len_x, y=len_y,
                        mode='markers',
                        # ★ 点のサイズを3.3(約1/3)にし、白い枠線(line)の指定を削除
                        marker=dict(color='#00E5FF', size=3.3, opacity=0.6),
                        name='オイル長さ',
                        hovertemplate="オイル長さ: %{x}ft<br>スコア: %{y}<extra></extra>"
                    ))
                    fig_len.update_layout(
                        title=dict(text="オイル長さとスコアの関係", font=dict(color='silver', size=14)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        # fixedrange=True でズームやスクロールを完全に禁止
                        xaxis=dict(title="オイル長さ (ft)", range=[29, 46], color='silver', gridcolor='#444', dtick=1, fixedrange=True),
                        yaxis=dict(title="Score", range=[0, 320], color='silver', gridcolor='#444', dtick=50, fixedrange=True),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=350
                    )

                    # ▼ 2つ目のグラフ：オイル量 適性
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(
                        x=vol_x, y=vol_y,
                        mode='markers',
                        # ★ 点のサイズを3.3(約1/3)にし、白い枠線(line)の指定を削除
                        marker=dict(color='#FF007F', size=3.3, opacity=0.6),
                        name='オイル量',
                        hovertemplate="オイル量: %{x}ml<br>スコア: %{y}<extra></extra>"
                    ))
                    fig_vol.update_layout(
                        title=dict(text="オイル量とスコアの関係", font=dict(color='silver', size=14)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        # fixedrange=True でズームやスクロールを完全に禁止
                        xaxis=dict(title="オイル量 (ml)", range=[19, 36], color='silver', gridcolor='#444', dtick=1, fixedrange=True),
                        yaxis=dict(title="Score", range=[0, 320], color='silver', gridcolor='#444', dtick=50, fixedrange=True),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=350
                    )

                    # 描画 (タップ時のポップアップを残しつつ、操作メニューを非表示に)
                    st.plotly_chart(fig_len, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("<hr style='border-top: 1px dashed #444; margin: 10px 0px;'>", unsafe_allow_html=True)
                    st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})

                

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【14】月別集計（MONTHLY STATS）機能
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_monthly_stats():
                    st.markdown("### <span style='color: silver;'>📅 MONTHLY STATS</span>", unsafe_allow_html=True)

                    named_splits = {
                        "7-10": ("2P", "スネークアイ"),
                        "2-7": ("2P", "ベビースプリット"),
                        "3-10": ("2P", "ベビースプリット"),
                        "4-6": ("2P", "フォーシックス"),
                        "4-9": ("2P", "ビッグディボット"),
                        "6-8": ("2P", "ビッグディボット"),
                        "5-7": ("2P", "ダイムストア"),
                        "5-10": ("2P", "ダイムストア"),
                        "7-9": ("2P", "ムース"),
                        "8-10": ("2P", "ムース"),
                        "5-7-10": ("3P", "リリー"),
                        "2-7-10": ("3P", "クリスマスツリー"),
                        "3-7-10": ("3P", "クリスマスツリー"),
                        "4-7-9": ("3P", "マイティマイト"),
                        "6-7-10": ("3P", "マイティマイト"),
                        "4-6-7-10": ("4_5P", "ビッグフォー"),
                        "4-6-7-8-10": ("4_5P", "グリークチャーチ"),
                        "4-6-7-9-10": ("4_5P", "ワシントン条約")
                    }

                    monthly_data = {}
                    
                    def get_pins(val):
                        v = str(val).replace("R:", "").strip().upper()
                        if v == "X": return 10
                        if v.isdigit(): return int(v)
                        return 0

                    def is_strike(val):
                        return str(val).replace("R:", "").strip().upper() == "X"
                        
                    def is_spare(val):
                        return "/" in str(val).upper()

                    # データ集計ループ
                    for row in master_data:
                        if len(row) < 55 or str(row[1]).strip() != selected_player:
                            continue
                            
                        date_str = str(row[2]).strip()
                        if len(date_str) < 5: continue
                        month_key = "/".join(date_str.split("/")[:2]) # "YY/MM" の形式
                        
                        if month_key not in monthly_data:
                            monthly_data[month_key] = {
                                "games": 0, "pitches": 0, "pin_falls": 0, "total_score": 0,
                                "high_score": 0, "score_300": 0, "score_275": 0, "score_250": 0,
                                "score_225": 0, "score_200": 0, "no_miss_games": 0,
                                "strikes": 0, "strike_chances": 0, "spares": 0, "spare_chances": 0,
                                "no_head": 0, "no_head_chances": 0,
                                "pin7_s": 0, "pin7_c": 0, "pin10_s": 0, "pin10_c": 0,
                                "split_s": 0, "split_c": 0,
                                "streak": {i: 0 for i in range(3, 13)},
                                "games_by_num": {i: {"score": 0, "count": 0} for i in range(1, 14)},
                                "game_710_c": 0, "game_710_s": 0
                            }
                        
                        m_stat = monthly_data[month_key]
                        
                        # 7-10ゲーム判定
                        is_710 = str(row[54]).strip().upper() == "TRUE"
                        if is_710:
                            m_stat["game_710_c"] += 1
                            if str(row[52]).strip() == "10":
                                m_stat["game_710_s"] += 1
                            continue
                            
                        # ゲーム数・スコア
                        m_stat["games"] += 1
                        try:
                            score = int(row[52])
                        except:
                            score = 0
                        m_stat["total_score"] += score
                        if score > m_stat["high_score"]: m_stat["high_score"] = score
                        if score == 300: m_stat["score_300"] += 1
                        if score >= 275: m_stat["score_275"] += 1
                        if score >= 250: m_stat["score_250"] += 1
                        if score >= 225: m_stat["score_225"] += 1
                        if score >= 200: m_stat["score_200"] += 1
                        
                        # Nゲーム目判定
                        g_num_str = str(row[6]).replace("G", "").strip()
                        if g_num_str.isdigit():
                            g_num = int(g_num_str)
                            target_g = g_num if g_num <= 12 else 13
                            m_stat["games_by_num"][target_g]["score"] += score
                            m_stat["games_by_num"][target_g]["count"] += 1

                        game_seq = []
                        miss_flag = False
                        
                        # 1〜9フレーム処理
                        for f in range(9):
                            base_idx = 10 + f * 4
                            res1 = str(row[base_idx])
                            rem1 = str(row[base_idx + 1])
                            res2 = str(row[base_idx + 2])
                            
                            m_stat["pitches"] += 1
                            m_stat["strike_chances"] += 1
                            m_stat["no_head_chances"] += 1
                            
                            if "1" in rem1.split(","):
                                m_stat["no_head"] += 1
                            
                            if is_strike(res1):
                                m_stat["strikes"] += 1
                                m_stat["pin_falls"] += 10
                                game_seq.append("X")
                            else:
                                game_seq.append("-")
                                m_stat["pitches"] += 1
                                m_stat["spare_chances"] += 1
                                p1 = get_pins(res1)
                                
                                if is_spare(res2):
                                    m_stat["spares"] += 1
                                    m_stat["pin_falls"] += 10
                                else:
                                    miss_flag = True
                                    m_stat["pin_falls"] += (p1 + get_pins(res2))
                                    
                                is_split = False
                                rem1_sorted = ",".join(sorted(rem1.replace(" ", "").split(",")))
                                if rem1_sorted in named_splits:
                                    m_stat["split_c"] += 1
                                    if is_spare(res2): m_stat["split_s"] += 1
                                    
                                if "7" == rem1.strip():
                                    m_stat["pin7_c"] += 1
                                    if is_spare(res2): m_stat["pin7_s"] += 1
                                elif "10" == rem1.strip():
                                    m_stat["pin10_c"] += 1
                                    if is_spare(res2): m_stat["pin10_s"] += 1
                                    
                        # 10フレーム処理
                        res10_1 = str(row[46])
                        rem10_1 = str(row[47])
                        res10_2 = str(row[48])
                        res10_2_rem = str(row[49])
                        res10_3 = str(row[50])
                        
                        m_stat["pitches"] += 1
                        m_stat["strike_chances"] += 1
                        m_stat["no_head_chances"] += 1
                        if "1" in rem10_1.split(","): m_stat["no_head"] += 1
                        
                        if is_strike(res10_1):
                            m_stat["strikes"] += 1
                            m_stat["pin_falls"] += 10
                            game_seq.append("X")
                            m_stat["pitches"] += 1
                            m_stat["strike_chances"] += 1
                            
                            if is_strike(res10_2):
                                m_stat["strikes"] += 1
                                m_stat["pin_falls"] += 10
                                game_seq.append("X")
                                m_stat["pitches"] += 1
                                m_stat["strike_chances"] += 1
                                
                                if is_strike(res10_3):
                                    m_stat["strikes"] += 1
                                    m_stat["pin_falls"] += 10
                                    game_seq.append("X")
                                else:
                                    game_seq.append("-")
                                    m_stat["pin_falls"] += get_pins(res10_3)
                                    if not is_spare(res10_3): miss_flag = True
                            else:
                                game_seq.append("-")
                                m_stat["pitches"] += 1
                                m_stat["spare_chances"] += 1
                                if is_spare(res10_3):
                                    m_stat["spares"] += 1
                                    m_stat["pin_falls"] += 10
                                else:
                                    miss_flag = True
                                    m_stat["pin_falls"] += (get_pins(res10_2) + get_pins(res10_3))
                        else:
                            game_seq.append("-")
                            m_stat["pitches"] += 1
                            m_stat["spare_chances"] += 1
                            p1 = get_pins(res10_1)
                            if is_spare(res10_2):
                                m_stat["spares"] += 1
                                m_stat["pin_falls"] += 10
                                m_stat["pitches"] += 1
                                m_stat["strike_chances"] += 1
                                if is_strike(res10_3):
                                    m_stat["strikes"] += 1
                                    m_stat["pin_falls"] += 10
                                    game_seq.append("X")
                                else:
                                    game_seq.append("-")
                                    m_stat["pin_falls"] += get_pins(res10_3)
                                    if not is_spare(res10_3): miss_flag = True
                            else:
                                miss_flag = True
                                m_stat["pin_falls"] += (p1 + get_pins(res10_2))
                                
                        if not miss_flag:
                            m_stat["no_miss_games"] += 1

                        current_streak = 0
                        for pitch in game_seq:
                            if pitch == "X":
                                current_streak += 1
                            else:
                                if current_streak >= 3:
                                    rec_streak = min(current_streak, 12)
                                    m_stat["streak"][rec_streak] += 1
                                current_streak = 0
                        if current_streak >= 3:
                            rec_streak = min(current_streak, 12)
                            m_stat["streak"][rec_streak] += 1

                    if not monthly_data:
                        st.info("集計可能な月別データがありません。")
                        return

                    # 全期間の合計を算出するためのデータセット作成
                    total_data = {
                        "games": 0, "pitches": 0, "pin_falls": 0, "total_score": 0,
                        "high_score": 0, "score_300": 0, "score_275": 0, "score_250": 0,
                        "score_225": 0, "score_200": 0, "no_miss_games": 0,
                        "strikes": 0, "strike_chances": 0, "spares": 0, "spare_chances": 0,
                        "no_head": 0, "no_head_chances": 0,
                        "pin7_s": 0, "pin7_c": 0, "pin10_s": 0, "pin10_c": 0,
                        "split_s": 0, "split_c": 0,
                        "streak": {i: 0 for i in range(3, 13)},
                        "games_by_num": {i: {"score": 0, "count": 0} for i in range(1, 14)},
                        "game_710_c": 0, "game_710_s": 0
                    }
                    
                    for m_stat in monthly_data.values():
                        total_data["games"] += m_stat["games"]
                        total_data["pitches"] += m_stat["pitches"]
                        total_data["pin_falls"] += m_stat["pin_falls"]
                        total_data["total_score"] += m_stat["total_score"]
                        total_data["high_score"] = max(total_data["high_score"], m_stat["high_score"])
                        total_data["score_300"] += m_stat["score_300"]
                        total_data["score_275"] += m_stat["score_275"]
                        total_data["score_250"] += m_stat["score_250"]
                        total_data["score_225"] += m_stat["score_225"]
                        total_data["score_200"] += m_stat["score_200"]
                        total_data["no_miss_games"] += m_stat["no_miss_games"]
                        total_data["strikes"] += m_stat["strikes"]
                        total_data["strike_chances"] += m_stat["strike_chances"]
                        total_data["spares"] += m_stat["spares"]
                        total_data["spare_chances"] += m_stat["spare_chances"]
                        total_data["no_head"] += m_stat["no_head"]
                        total_data["no_head_chances"] += m_stat["no_head_chances"]
                        total_data["pin7_s"] += m_stat["pin7_s"]
                        total_data["pin7_c"] += m_stat["pin7_c"]
                        total_data["pin10_s"] += m_stat["pin10_s"]
                        total_data["pin10_c"] += m_stat["pin10_c"]
                        total_data["split_s"] += m_stat["split_s"]
                        total_data["split_c"] += m_stat["split_c"]
                        for i in range(3, 13):
                            total_data["streak"][i] += m_stat["streak"][i]
                        for i in range(1, 14):
                            total_data["games_by_num"][i]["score"] += m_stat["games_by_num"][i]["score"]
                            total_data["games_by_num"][i]["count"] += m_stat["games_by_num"][i]["count"]
                        total_data["game_710_c"] += m_stat["game_710_c"]
                        total_data["game_710_s"] += m_stat["game_710_s"]

                    # ① 年月を降順（新しい順）に並べ替え
                    sorted_months = sorted(monthly_data.keys(), reverse=True)
                    
                    def safe_rate(s, c): return f"{(s/c*100):.1f}%" if c > 0 else "-"
                    def safe_ave(s, c): return f"{(s/c):.1f}" if c > 0 else "-"

                    # ③ 名称を短くコンパクトに
                    rows = [
                        {"label": "G数", "key": "games", "fmt": lambda x: f"{x['games']}"},
                        {"label": "投球数", "key": "pitches", "fmt": lambda x: f"{x['pitches']}"},
                        {"label": "倒ピン", "key": "pin_falls", "fmt": lambda x: f"{x['pin_falls']}"},
                        {"label": "AVE", "key": "score_ave", "fmt": lambda x: safe_ave(x['total_score'], x['games'])},
                        {"label": "HIGH", "key": "high_score", "fmt": lambda x: f"{x['high_score']}"},
                        {"label": "300", "key": "score_300", "fmt": lambda x: f"{x['score_300']}"},
                        {"label": "275+", "key": "score_275", "fmt": lambda x: f"{x['score_275']}"},
                        {"label": "250+", "key": "score_250", "fmt": lambda x: f"{x['score_250']}"},
                        {"label": "225+", "key": "score_225", "fmt": lambda x: f"{x['score_225']}"},
                        {"label": "200+", "key": "score_200", "fmt": lambda x: f"{x['score_200']}"},
                        {"label": "ノーミス", "key": "no_miss_games", "fmt": lambda x: f"{x['no_miss_games']}"},
                        {"label": "ST数", "key": "strikes", "fmt": lambda x: f"{x['strikes']}"},
                        {"label": "ST率", "key": "strike_rate", "fmt": lambda x: safe_rate(x['strikes'], x['strike_chances'])},
                        {"label": "SP数", "key": "spares", "fmt": lambda x: f"{x['spares']}"},
                        {"label": "SP率", "key": "spare_rate", "fmt": lambda x: safe_rate(x['spares'], x['spare_chances'])},
                        {"label": "NO HEAD", "key": "no_head", "fmt": lambda x: f"{x['no_head']}"},
                        {"label": "NO HEAD率", "key": "no_head_rate", "fmt": lambda x: safe_rate(x['no_head'], x['no_head_chances'])},
                        {"label": "7ピン率", "key": "pin7_rate", "fmt": lambda x: safe_rate(x['pin7_s'], x['pin7_c'])},
                        {"label": "10ピン率", "key": "pin10_rate", "fmt": lambda x: safe_rate(x['pin10_s'], x['pin10_c'])},
                    ]
                    
                    for i in range(3, 13):
                        rows.append({"label": f"{i}連ST", "key": f"streak_{i}", "fmt": lambda x, idx=i: f"{x['streak'][idx]}"})
                        
                    for i in range(1, 13):
                        rows.append({"label": f"G{i} AVE", "key": f"g{i}_ave", "fmt": lambda x, idx=i: safe_ave(x['games_by_num'][idx]['score'], x['games_by_num'][idx]['count'])})
                    rows.append({"label": "G13+ AVE", "key": "g13_ave", "fmt": lambda x: safe_ave(x['games_by_num'][13]['score'], x['games_by_num'][13]['count'])})
                    
                    rows.append({"label": "7-10 G数", "key": "game_710_c", "fmt": lambda x: f"{x['game_710_c']}"})
                    rows.append({"label": "7-10 MAKE", "key": "game_710_s", "fmt": lambda x: f"{x['game_710_s']}"})

                    # ② 項目とTOTALを左側に固定（CSSで追従）させる
                    html = """<div style="background: #1e1e1e; padding: 15px; border-radius: 10px; overflow-x: auto; margin-bottom: 20px;">
<table style="width: 100%; border-collapse: separate; border-spacing: 0; white-space: nowrap; font-size: 13px;">
<thead>
<tr style="color: silver;">
<th style="padding: 4px 8px; position: sticky; left: 0; background: #2a2a2e; z-index: 3; border-bottom: 2px solid #555; width: 85px; min-width: 85px; max-width: 85px;">項目</th>
<th style="padding: 4px 8px; position: sticky; left: 85px; background: #2a2a2e; z-index: 3; border-bottom: 2px solid #555; border-right: 2px solid #555; width: 65px; min-width: 65px; max-width: 65px; text-align: right; color: #bf953f;">TOTAL</th>"""
                    
                    for m in sorted_months:
                        html += f'<th style="padding: 4px 10px; text-align: right; border-bottom: 2px solid #555;">{m}</th>'
                    html += "</tr></thead><tbody>"

                    for r in rows:
                        html += f'<tr>'
                        # 固定列1：項目名
                        html += f'<td style="padding: 4px 8px; position: sticky; left: 0; background: #2a2a2e; z-index: 2; font-weight: bold; color: #ddd; border-bottom: 1px dashed #444;">{r["label"]}</td>'
                        # 固定列2：TOTALデータ
                        val_total = r["fmt"](total_data)
                        html += f'<td style="padding: 4px 8px; position: sticky; left: 85px; background: #2a2a2e; z-index: 2; text-align: right; font-weight: bold; color: #bf953f; border-bottom: 1px dashed #444; border-right: 2px solid #555;">{val_total}</td>'
                        
                        # スクロール列：月別データ
                        for m in sorted_months:
                            val = r["fmt"](monthly_data[m])
                            html += f'<td style="padding: 4px 10px; text-align: right; color: white; border-bottom: 1px dashed #444;">{val}</td>'
                        html += "</tr>"
                    html += "</tbody></table></div>"
                    
                    st.markdown(html, unsafe_allow_html=True)

                    # グラフ展開エリア
                    st.markdown("<h4 style='color: silver; margin-top: 10px;'>📊 項目別グラフ</h4>", unsafe_allow_html=True)
                    graph_options = [r["label"] for r in rows]
                    
                    if "monthly_graph_sel" not in st.session_state:
                        st.session_state.monthly_graph_sel = graph_options[0]

                    # ★ポップオーバーの代わりにダイアログ（ポップアップ）機能を使用
                    @st.dialog("📊 グラフ化する項目を選択")
                    def open_graph_selector():
                        # スマホでもスクロールが減るように2列に分けて配置
                        cols = st.columns(2)
                        for i, opt in enumerate(graph_options):
                            label = f"✅ {opt}" if st.session_state.monthly_graph_sel == opt else opt
                            if cols[i % 2].button(label, key=f"dlg_btn_{opt}", use_container_width=True):
                                st.session_state.monthly_graph_sel = opt
                                st.rerun()

                    # 窓を開くためのメインボタン
                    if st.button(f"🔽 グラフ表示を変更： {st.session_state.monthly_graph_sel}", use_container_width=True):
                        open_graph_selector()

                    selected_graph = st.session_state.monthly_graph_sel
                    
                    if selected_graph:
                        import plotly.graph_objects as go
                        target_row = next(r for r in rows if r["label"] == selected_graph)
                        
                        # グラフ用の横軸は時系列（古い→新しい）に再ソートして表示
                        x_vals_graph = sorted(monthly_data.keys())
                        y_vals = []
                        for m in x_vals_graph:
                            val_str = target_row["fmt"](monthly_data[m]).replace("%", "")
                            try:
                                y_vals.append(float(val_str))
                            except:
                                y_vals.append(0.0)
                                
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_vals_graph, y=y_vals, mode='lines+markers', line=dict(color='#bf953f', width=3), marker=dict(size=8, color='white')))
                        fig.update_layout(
                            title=dict(text=f"{selected_graph} の推移", font=dict(color='white')),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                            xaxis=dict(color='silver', showgrid=False),
                            yaxis=dict(color='silver', gridcolor='#444'),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=350
                        )
                        # ★ config={'displayModeBar': False} を追加して右上のアイコンを全消去
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # =========================================================
                # ▼▼▼ 設定に従って画面を描画する処理（ここは変更不要） ▼▼▼
                # =========================================================
                
                # 全関数を辞書に登録
                render_functions = {
                    "01_rating_card": render_01_rating_card,
                    "02_score_trend": render_02_score_trend,
                    "03_seven_ten": render_03_seven_ten,
                    "04_first_pitch_pins": render_04_first_pitch_pins,
                    "05_consecutive": render_05_consecutive,
                    "06_total_monthly": render_06_total_monthly,
                    "07_high_scores": render_07_high_scores,
                    "08_split_make": render_08_split_make,
                    "09_play_style": render_09_play_style,
                    "10_oil_length": render_10_oil_length,
                    "11_oil_volume": render_11_oil_volume,
                    "12_lane_data": render_12_lane_data,
                    "13_seven_ten_game": render_13_seven_ten_game,
                    "14_top10_scores": render_14_top10_scores,
                    "15_monthly_stats": render_monthly_stats,
                    "16_rating_trend": render_16_rating_trend,
                    "17_env_scatter": render_17_env_scatter
                }

                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # 【キオスク専用】 今月の簡易サマリー計算・描画
                # ＃★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def render_kiosk_monthly_summary():
                    if not player_games:
                        return
                    
                    # 1. 登録データの中で最新の月を探す
                    latest_month = ""
                    for g in player_games:
                        try:
                            parts = str(g["date"]).strip().split('/')
                            if len(parts) >= 2:
                                yy = int(parts[0])
                                yyyy = 2000 + yy if yy < 100 else yy
                                mm = int(parts[1])
                                m_key = f"{yyyy:04d}/{mm:02d}"
                                if m_key > latest_month:
                                    latest_month = m_key
                        except:
                            pass
                    
                    if not latest_month:
                        return

                    # 2. その月のデータを集計
                    g_count = 0
                    total_score = 0
                    h_score = 0
                    pitches = 0
                    pin_falls = 0

                    def get_pins(val):
                        v = str(val).replace("R:", "").strip().upper()
                        if v == "X": return 10
                        if v.isdigit(): return int(v)
                        return 0

                    for g in player_games:
                        try:
                            parts = str(g["date"]).strip().split('/')
                            if len(parts) < 2: continue
                            yy = int(parts[0])
                            yyyy = 2000 + yy if yy < 100 else yy
                            mm = int(parts[1])
                            m_key = f"{yyyy:04d}/{mm:02d}"
                            
                            if m_key == latest_month:
                                g_count += 1
                                score = g["score"]
                                total_score += score
                                if score > h_score: h_score = score
                                
                                r = g["row"]
                                for f in range(9):
                                    res1 = str(r[10 + f * 4]).upper()
                                    res2 = str(r[12 + f * 4]).upper()
                                    pitches += 1
                                    if "X" in res1:
                                        pin_falls += 10
                                    else:
                                        pitches += 1
                                        if "/" in res2:
                                            pin_falls += 10
                                        else:
                                            pin_falls += get_pins(res1) + get_pins(res2)
                                
                                res10_1 = str(r[46]).upper() if len(r) > 46 else ""
                                res10_2 = str(r[48]).upper() if len(r) > 48 else ""
                                res10_3 = str(r[50]).upper() if len(r) > 50 else ""
                                
                                pitches += 1
                                if "X" in res10_1:
                                    pin_falls += 10
                                    pitches += 1
                                    if "X" in res10_2:
                                        pin_falls += 10
                                        pitches += 1
                                        if "X" in res10_3: pin_falls += 10
                                        else: pin_falls += get_pins(res10_3)
                                    else:
                                        if "/" in res10_3: pin_falls += 10
                                        else: pin_falls += get_pins(res10_2) + get_pins(res10_3)
                                else:
                                    pitches += 1
                                    if "/" in res10_2:
                                        pin_falls += 10
                                        pitches += 1
                                        if "X" in res10_3: pin_falls += 10
                                        else: pin_falls += get_pins(res10_3)
                                    else:
                                        pin_falls += get_pins(res10_1) + get_pins(res10_2)
                        except:
                            pass

                    ave = round(total_score / g_count, 1) if g_count > 0 else 0.0

                    # 3. HTMLでコンパクトに描画（修正版）
                    html = f"""
<div style="background: linear-gradient(145deg, #2a2a2e, #1c1c1e); padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); border: 1px solid #333; margin-bottom: 20px;">
<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 20px;">
<div style="text-align: center; padding: 10px 30px; background: #1a1a1c; border-radius: 10px; border: 1px solid #444; width: 100%; max-width: 250px; margin-bottom: 5px;">
<div style="color: silver; font-size: 14px; font-weight: bold; letter-spacing: 1px;">LATEST MONTH</div>
<div style="color: #bf953f; font-size: 28px; font-weight: 900;">{latest_month}</div>
</div>
<div style="display: flex; justify-content: space-between; width: 100%; text-align: center;">
<div style="flex: 1;">
<div style="color: gray; font-size: 14px; font-weight: bold;">G数</div>
<div style="color: white; font-size: 26px; font-weight: bold;">{g_count}</div>
</div>
<div style="flex: 1;">
<div style="color: gray; font-size: 14px; font-weight: bold;">投球数</div>
<div style="color: white; font-size: 26px; font-weight: bold;">{pitches}</div>
</div>
<div style="flex: 1;">
<div style="color: gray; font-size: 14px; font-weight: bold;">倒ピン</div>
<div style="color: white; font-size: 26px; font-weight: bold;">{pin_falls}</div>
</div>
<div style="flex: 1;">
<div style="color: #ff3b30; font-size: 14px; font-weight: bold;">AVE</div>
<div style="color: white; font-size: 26px; font-weight: bold;">{ave}</div>
</div>
<div style="flex: 1;">
<div style="color: #4285f4; font-size: 14px; font-weight: bold;">HIGH</div>
<div style="color: white; font-size: 26px; font-weight: bold;">{h_score}</div>
</div>
</div>
</div>
</div>
"""
                    st.markdown(html, unsafe_allow_html=True)

                if st.session_state.get("kiosk_mode"):
                    st.markdown("<h1 class='premium-header' style='font-size: 24px; margin-top: 0px; margin-bottom: 10px;'>REGISTRATION COMPLETE</h1>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        # 左側：レーティングカード
                        render_01_rating_card()
                        
                        # 左側下段：横並びのボタン（装飾CSSを利用して目立たせる）
                        b1, b2 = st.columns(2)
                        with b1:
                            st.markdown("<div class='red-btn-marker' style='display: none;'></div>", unsafe_allow_html=True)
                            if st.button("CHECK-IN画面へ", use_container_width=True):
                                st.session_state.kiosk_step = "auth"
                                st.session_state.kiosk_user = None
                                st.rerun()
                        with b2:
                            st.markdown("<div class='gold-btn-marker' style='display: none;'></div>", unsafe_allow_html=True)
                            if st.button("解析を続ける", use_container_width=True):
                                st.session_state.kiosk_step = "register"
                                st.rerun()
                    
                    with col2:
                        # 右側：3段分割のコンテンツ
                        render_kiosk_monthly_summary()
                        render_02_score_trend()
                        render_16_rating_trend()
                else:
                    # レイアウト辞書のキー（タブ名）からタブを生成
                    tab_titles = list(dashboard_layout.keys())
                    tabs = st.tabs(tab_titles)
                    
                    # 各タブの中に、指定された順序で関数を呼び出して描画
                    for i, tab_title in enumerate(tab_titles):
                        with tabs[i]:
                            for item_key in dashboard_layout[tab_title]:
                                if item_key in render_functions:
                                    render_functions[item_key]()

        except Exception as e:
            st.error(f"データ取得エラー: {e}")

    # ⚠️ 【重要】分析モードの時は、これより下の「登録画面のコード」を読み込まずにストップする
    st.stop()


# ⚠️ AIモデル設定：Flash版を除外し、Proモデルに限定（存在しないモデル名によるエラーを防止）
fallback_models = [
    'gemini-2.5-pro',
]

# =========================================================
# 【新機能】データ比較
# =========================================================
if app_mode == "データ比較":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
    import re

    # ▼ INITIATE ANALYSIS ボタンを目立たせる専用CSS
    st.markdown("""
    <style>
    /* initiate-marker を含んだコンテナの「次のコンテナ」にあるボタンを装飾 */
    div[data-testid="stElementContainer"]:has(.initiate-marker) + div[data-testid="stElementContainer"] button,
    div.element-container:has(.initiate-marker) + div.element-container button {
        background: linear-gradient(145deg, #bf953f, #aa771c) !important;
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 900 !important;
        border: 2px solid #fcf6ba !important;
        box-shadow: 0 0 15px rgba(191, 149, 63, 0.6) !important;
        height: 70px !important;
        border-radius: 12px !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
    }
    /* Streamlitの<p>タグによる文字色上書きを防ぐ */
    div[data-testid="stElementContainer"]:has(.initiate-marker) + div[data-testid="stElementContainer"] button p,
    div.element-container:has(.initiate-marker) + div.element-container button p {
        color: #ffffff !important;
        font-weight: 900 !important;
    }
    div[data-testid="stElementContainer"]:has(.initiate-marker) + div[data-testid="stElementContainer"] button:hover,
    div.element-container:has(.initiate-marker) + div.element-container button:hover {
        background: linear-gradient(145deg, #fcf6ba, #bf953f) !important;
        box-shadow: 0 0 25px rgba(191, 149, 63, 0.9) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    div[data-testid="stElementContainer"]:has(.initiate-marker) + div[data-testid="stElementContainer"] button:hover p,
    div.element-container:has(.initiate-marker) + div.element-container button:hover p {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    render_section_title("データ比較")

    with st.spinner("データベースに接続中... 高度解析エンジンを起動しています..."):
        try:
            sh = get_gspread_client()
            if not sh:
                st.error("データベースに接続できません。")
                st.stop()
            master_data_raw = sh.worksheet("マスター").get_all_values()
            settings_data = sh.worksheet("プレイヤー設定").get_all_values()
            all_players = [row[1] for row in settings_data[1:] if len(row) >= 2 and row[1]]
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.stop()

    def calc_rt_val(scores):
        if not scores: return 0.0
        a = sum(scores) / len(scores)
        if a >= 230: rt = 18 + (a - 230) * (3 / 20)
        elif a >= 210: rt = 15 + (a - 210) * (3 / 20)
        elif a >= 190: rt = 12 + (a - 190) * (3 / 20)
        elif a >= 170: rt = 9 + (a - 170) * (3 / 20)
        elif a >= 145: rt = 6 + (a - 145) * (3 / 25)
        elif a >= 95: rt = 1 + (a - 95) * 0.1
        else: rt = a / 95
        return round(max(1.0, rt), 2)

    def parse_master_data(data):
        parsed = []
        for row in data[1:]:
            if len(row) < 53 or not row[1]: continue
            if len(row) > 54 and str(row[54]).strip().upper() == "TRUE": continue
            try:
                date_str = str(row[2]).strip()
                score = int(row[52])
                parts = date_str.split('/')
                
                if len(parts) >= 2:
                    y_str = parts[0][-2:] if len(parts[0]) >= 2 else parts[0].zfill(2)
                    y_full = f"20{y_str}"
                    m_full = parts[1].zfill(2)
                    month_key = f"{y_full}/{m_full}"
                else:
                    month_key = "不明"

                dt_val = pd.NaT
                if len(parts) >= 3:
                    try:
                        d_full = parts[2].zfill(2)
                        t_str = str(row[3]).strip() if len(row) > 3 else "00:00"
                        if not t_str: t_str = "00:00"
                        if len(t_str.split(':')) == 2:
                            dt_val = datetime.strptime(f"{y_str}/{m_full}/{d_full} {t_str}", "%y/%m/%d %H:%M")
                        else:
                            dt_val = datetime.strptime(f"{y_str}/{m_full}/{d_full}", "%y/%m/%d")
                    except:
                        pass
                
                cond1 = str(row[55]).strip() if len(row) > 55 else ""
                cond2 = str(row[56]).strip() if len(row) > 56 else ""
                cond3 = str(row[57]).strip() if len(row) > 57 else ""
                ball = str(row[9]).strip()
                lane = str(row[5]).strip()
                oil_len = str(row[7]).strip()
                oil_vol = str(row[8]).strip()

                st_c = 0; st_s = 0; sp_c = 0; sp_s = 0
                pin7_c = 0; pin7_s = 0; pin10_c = 0; pin10_s = 0
                split_c = 0; split_s = 0
                open_frames = 0
                pin_left = {str(i): 0 for i in range(1, 11)}
                first_pitch_pins = 0; first_pitch_count = 0; no_head = 0
                game_seq = []

                def clean_res(v): return str(v).strip().upper()
                def get_pins(p_str): return [str(p) for p in re.findall(r'\d+', str(p_str)) if 1 <= int(p) <= 10]
                
                for f in range(9):
                    res1 = clean_res(row[10+f*4]); pin1 = str(row[11+f*4]); res2 = clean_res(row[12+f*4])
                    st_c += 1; first_pitch_count += 1
                    if "X" in res1:
                        st_s += 1; first_pitch_pins += 10; game_seq.append("X")
                    else:
                        sp_c += 1; game_seq.append("-")
                        left_arr = get_pins(pin1)
                        first_pitch_pins += (10 - len(left_arr))
                        if "1" in left_arr: no_head += 1
                        for p in left_arr: pin_left[p] += 1
                        if "7" in left_arr and len(left_arr) == 1: pin7_c += 1
                        if "10" in left_arr and len(left_arr) == 1: pin10_c += 1
                        if len(left_arr) >= 2: split_c += 1
                        if "/" in res2:
                            sp_s += 1
                            if "7" in left_arr and len(left_arr) == 1: pin7_s += 1
                            if "10" in left_arr and len(left_arr) == 1: pin10_s += 1
                            if len(left_arr) >= 2: split_s += 1
                        else:
                            open_frames += 1

                r10_1 = clean_res(row[46]) if len(row)>46 else ""; p10_1 = str(row[47]) if len(row)>47 else ""
                r10_2 = clean_res(row[48]) if len(row)>48 else ""; p10_2 = str(row[49]) if len(row)>49 else ""
                r10_3 = clean_res(row[50]) if len(row)>50 else ""; p10_3 = str(row[51]) if len(row)>51 else ""

                st_c += 1; first_pitch_count += 1
                if "X" in r10_1:
                    st_s += 1; first_pitch_pins += 10; game_seq.append("X")
                    st_c += 1
                    if "X" in r10_2:
                        st_s += 1; game_seq.append("X")
                        st_c += 1
                        if "X" in r10_3: st_s += 1; game_seq.append("X")
                        else:
                            game_seq.append("-")
                            for p in get_pins(p10_3): pin_left[p] += 1
                            if r10_3 != "/": open_frames += 1
                    else:
                        sp_c += 1; game_seq.append("-")
                        left_arr = get_pins(p10_2)
                        for p in left_arr: pin_left[p] += 1
                        if "/" in r10_3: sp_s += 1
                        else: open_frames += 1
                else:
                    sp_c += 1; game_seq.append("-")
                    left_arr = get_pins(p10_1)
                    first_pitch_pins += (10 - len(left_arr))
                    if "1" in left_arr: no_head += 1
                    for p in left_arr: pin_left[p] += 1
                    if "/" in r10_2:
                        sp_s += 1; st_c += 1
                        if "X" in r10_3: st_s += 1; game_seq.append("X")
                        else:
                            game_seq.append("-")
                            for p in get_pins(p10_3): pin_left[p] += 1
                    else:
                        open_frames += 1

                db_c = 0; db_s = 0; tk_c = 0; tk_s = 0; current_streak = 0; max_streak = 0
                for i in range(len(game_seq)):
                    if game_seq[i] == "X":
                        current_streak += 1
                        if current_streak > max_streak: max_streak = current_streak
                    else: current_streak = 0
                    if i > 1 and game_seq[i-1] == "X" and game_seq[i-2] == "X":
                        db_c += 1
                        if game_seq[i] == "X": db_s += 1
                    if i > 2 and game_seq[i-1] == "X" and game_seq[i-2] == "X" and game_seq[i-3] == "X":
                        tk_c += 1
                        if game_seq[i] == "X": tk_s += 1

                parsed_row = {
                    "player": row[1],
                    "datetime": dt_val,
                    "month_key": month_key, "score": score, "max_strike_streak": max_streak,
                    "st_c": st_c, "st_s": st_s, "sp_c": sp_c, "sp_s": sp_s,
                    "pin7_c": pin7_c, "pin7_s": pin7_s, "pin10_c": pin10_c, "pin10_s": pin10_s,
                    "split_c": split_c, "split_s": split_s,
                    "open_frames": open_frames, "is_nomiss": 1 if open_frames == 0 else 0,
                    "up200": 1 if score >= 200 else 0, "up225": 1 if score >= 225 else 0, "up250": 1 if score >= 250 else 0,
                    "first_pitch_pins": first_pitch_pins, "first_pitch_count": first_pitch_count, "no_head": no_head,
                    "db_c": db_c, "db_s": db_s, "tk_c": tk_c, "tk_s": tk_s,
                    "cond1": cond1, "cond2": cond2, "cond3": cond3,
                    "ball": ball, "lane": lane, "oil_len": oil_len, "oil_vol": oil_vol
                }
                for i in range(1, 11): parsed_row[f"pin{i}_left"] = pin_left[str(i)]
                parsed.append(parsed_row)
            except Exception:
                pass
        return parsed

    raw_df = pd.DataFrame(parse_master_data(master_data_raw))

    if not raw_df.empty:
        raw_df = raw_df.dropna(subset=["datetime"]).sort_values("datetime", ascending=True)
        raw_df["game_num"] = raw_df.groupby("player").cumcount() + 1
        raw_df["rating"] = raw_df.groupby("player")["score"].transform(
            lambda x: x.rolling(window=50, min_periods=1).apply(lambda s: calc_rt_val(s.dropna().tolist()), raw=False)
        )
        def safe_div(a, b): return (a / b * 100) if b > 0 else 0
        raw_df["st_rate"] = raw_df.apply(lambda r: safe_div(r["st_s"], r["st_c"]), axis=1)
        raw_df["sp_rate"] = raw_df.apply(lambda r: safe_div(r["sp_s"], r["sp_c"]), axis=1)
        raw_df["nomiss_pct"] = raw_df["is_nomiss"] * 100
        raw_df["fp_ave"] = raw_df.apply(lambda r: (r["first_pitch_pins"] / r["first_pitch_count"]) if r["first_pitch_count"] > 0 else 0, axis=1)
        raw_df["nohead_rate"] = raw_df.apply(lambda r: safe_div(r["no_head"], r["first_pitch_count"]), axis=1)
        raw_df["db_rate"] = raw_df.apply(lambda r: safe_div(r["db_s"], r["db_c"]), axis=1)
        raw_df["tk_rate"] = raw_df.apply(lambda r: safe_div(r["tk_s"], r["tk_c"]), axis=1)
        raw_df["split_enc_rate"] = raw_df.apply(lambda r: safe_div(r["split_c"], r["st_c"]), axis=1)
        raw_df["split_make_rate"] = raw_df.apply(lambda r: safe_div(r["split_s"], r["split_c"]), axis=1)
        raw_df["pin7_rate"] = raw_df.apply(lambda r: safe_div(r["pin7_s"], r["pin7_c"]), axis=1)
        raw_df["pin10_rate"] = raw_df.apply(lambda r: safe_div(r["pin10_s"], r["pin10_c"]), axis=1)
        for i in range(1, 11): raw_df[f"pin{i}_left_rate"] = raw_df.apply(lambda r, p=i: safe_div(r[f"pin{p}_left"], r["st_c"]), axis=1)

    Y_METRICS = {
        "レーティング": {"agg": lambda df: calc_rt_val(df.sort_values("datetime")["score"].tail(50).tolist()), "col": "rating"},
        "スコア (アベレージ)": {"agg": lambda df: df["score"].mean(), "col": "score"},
        "ハイスコア": {"agg": lambda df: df["score"].max(), "col": "score"},
        "ロースコア": {"agg": lambda df: df["score"].min(), "col": "score"},
        "合計総ピン数": {"agg": lambda df: df["score"].sum(), "col": "score"},
        "最大連続ストライク数": {"agg": lambda df: df["max_strike_streak"].max(), "col": "max_strike_streak"},
        "ストライク数 / G": {"agg": lambda df: df["st_s"].mean(), "col": "st_s"},
        "ストライク率 (%)": {"agg": lambda df: (df["st_s"].sum() / df["st_c"].sum() * 100) if df["st_c"].sum() > 0 else 0, "col": "st_rate"},
        "スペア数 / G": {"agg": lambda df: df["sp_s"].mean(), "col": "sp_s"},
        "スペア率 (%)": {"agg": lambda df: (df["sp_s"].sum() / df["sp_c"].sum() * 100) if df["sp_c"].sum() > 0 else 0, "col": "sp_rate"},
        "ノーミス率 (%)": {"agg": lambda df: (df["is_nomiss"].sum() / len(df) * 100) if len(df) > 0 else 0, "col": "nomiss_pct"},
        "オープンフレーム数 / G": {"agg": lambda df: df["open_frames"].mean(), "col": "open_frames"},
        "1投目 平均カウント (本)": {"agg": lambda df: (df["first_pitch_pins"].sum() / df["first_pitch_count"].sum()) if df["first_pitch_count"].sum() > 0 else 0, "col": "fp_ave"},
        "ノーヘッド率 (%)": {"agg": lambda df: (df["no_head"].sum() / df["first_pitch_count"].sum() * 100) if df["first_pitch_count"].sum() > 0 else 0, "col": "nohead_rate"},
        "ダブル成功率 (%)": {"agg": lambda df: (df["db_s"].sum() / df["db_c"].sum() * 100) if df["db_c"].sum() > 0 else 0, "col": "db_rate"},
        "ターキー成功率 (%)": {"agg": lambda df: (df["tk_s"].sum() / df["tk_c"].sum() * 100) if df["tk_c"].sum() > 0 else 0, "col": "tk_rate"},
        "スプリット遭遇率 (%)": {"agg": lambda df: (df["split_c"].sum() / df["st_c"].sum() * 100) if df["st_c"].sum() > 0 else 0, "col": "split_enc_rate"},
        "スプリットメイク率 (%)": {"agg": lambda df: (df["split_s"].sum() / df["split_c"].sum() * 100) if df["split_c"].sum() > 0 else 0, "col": "split_make_rate"},
        "7番ピン カバー率 (%)": {"agg": lambda df: (df["pin7_s"].sum() / df["pin7_c"].sum() * 100) if df["pin7_c"].sum() > 0 else 0, "col": "pin7_rate"},
        "10番ピン カバー率 (%)": {"agg": lambda df: (df["pin10_s"].sum() / df["pin10_c"].sum() * 100) if df["pin10_c"].sum() > 0 else 0, "col": "pin10_rate"},
        "200UP達成率 (%)": {"agg": lambda df: (df["up200"].sum() / len(df) * 100) if len(df) > 0 else 0, "col": "up200"},
        "225UP達成率 (%)": {"agg": lambda df: (df["up225"].sum() / len(df) * 100) if len(df) > 0 else 0, "col": "up225"},
        "250UP達成率 (%)": {"agg": lambda df: (df["up250"].sum() / len(df) * 100) if len(df) > 0 else 0, "col": "up250"},
    }
    for i in range(1, 11):
        Y_METRICS[f"{i}番ピン 残存率 (%)"] = {"agg": lambda df, p=i: (df[f"pin{p}_left"].sum() / df["st_c"].sum() * 100) if df["st_c"].sum() > 0 else 0, "col": f"pin{i}_left_rate"}

    X_AXIS_OPTIONS = {
        "プレイヤー": "player",
        "ゲーム順 (時系列)": "game_num",
        "月別推移 (YYYY/MM)": "month_key",
        "個別条件1": "cond1",
        "個別条件2": "cond2",
        "個別条件3": "cond3",
        "使用ボール": "ball",
        "使用レーン": "lane",
        "オイル長 (ft)": "oil_len",
        "オイル量 (ml)": "oil_vol"
    }

    # ▼ STEP 1
    st.markdown("<div style='color: #bf953f; font-weight: bold; border-bottom: 1px solid #444; margin-top: 20px; margin-bottom: 10px;'>[STEP 1] DATA EXTRACTION & FILTERS</div>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_players = st.multiselect("対象プレイヤー (複数選択可)", all_players, default=all_players[:1] if all_players else [])
    with col_f2:
        period_type = st.radio("期間指定", ["直近〇ゲーム", "カレンダー指定"], horizontal=True)
        if period_type == "カレンダー指定":
            c_d1, c_d2 = st.columns(2)
            with c_d1: start_d = st.date_input("開始日", value=None)
            with c_d2: end_d = st.date_input("終了日", value=None)
            recent_g = 0
        else:
            recent_g = st.number_input("直近ゲーム数", min_value=1, max_value=1000, value=50, step=10)
            start_d = end_d = None

    filter_opts = {k: v for k, v in X_AXIS_OPTIONS.items() if v not in ["player", "game_num"]}
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        filter_col_name = st.selectbox("追加フィルター (特定の条件のみ抽出する場合)", ["選択しない"] + list(filter_opts.keys()))
    with col_f4:
        filter_val = None
        if filter_col_name != "選択しない":
            target_field = filter_opts[filter_col_name]
            unique_vals = [v for v in raw_df[target_field].unique() if str(v).strip() != ""]
            if unique_vals:
                filter_val = st.selectbox(f"「{filter_col_name}」の絞り込み値", unique_vals)
            else:
                st.info("データがありません")

    # ▼ STEP 2
    st.markdown("<div style='color: #bf953f; font-weight: bold; border-bottom: 1px solid #444; margin-top: 20px; margin-bottom: 10px;'>[STEP 2] VISUALIZATION FORMAT</div>", unsafe_allow_html=True)
    sel_graph = st.radio("表示形式", ["データ表", "棒グラフ", "折れ線グラフ", "分布図 (箱ひげ+散布)", "レーダーチャート"], horizontal=True)

    # ▼ STEP 3
    st.markdown("<div style='color: #bf953f; font-weight: bold; border-bottom: 1px solid #444; margin-top: 20px; margin-bottom: 10px;'>[STEP 3] AXIS & METRICS SETUP</div>", unsafe_allow_html=True)
    if sel_graph == "レーダーチャート":
        sel_xaxis = st.selectbox("X軸 (グループ化・比較の基準)", list(X_AXIS_OPTIONS.keys()))
        sel_metrics = st.multiselect("比較するデータ (複数選択)", list(Y_METRICS.keys()), default=list(Y_METRICS.keys())[:4])
        sel_yaxis = None
    else:
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            sel_xaxis = st.selectbox("X軸 (横軸・グループ化の基準)", list(X_AXIS_OPTIONS.keys()))
        with col_a2:
            sel_yaxis = st.selectbox("Y軸 (縦軸・比較するデータ)", list(Y_METRICS.keys()))

    # ▼ ここに追加：ボタンを狙い撃ちするための見えないマーカー
    st.markdown("<div class='initiate-marker' style='display: none;'></div>", unsafe_allow_html=True)

    if st.button("INITIATE ANALYSIS", type="primary", use_container_width=True):
        df = raw_df[raw_df["player"].isin(sel_players)]
        
        if period_type == "カレンダー指定":
            if start_d: df = df[df["datetime"].dt.date >= start_d]
            if end_d: df = df[df["datetime"].dt.date <= end_d]
        else:
            df = df.sort_values("datetime", ascending=False).groupby("player").head(recent_g)

        if filter_val and filter_col_name != "選択しない":
            df = df[df[filter_opts[filter_col_name]] == filter_val]

        if df.empty:
            st.warning("指定された条件のデータが見つかりません。条件を変更してください。")
            st.stop()

        x_col = X_AXIS_OPTIONS[sel_xaxis]
        df[x_col] = df[x_col].replace("", "未設定")

        grp_cols = ["player"] if x_col == "player" else [x_col, "player"]

        res = []
        for grp_vals, pdf in df.groupby(grp_cols):
            if x_col == "player":
                val = grp_vals[0] if isinstance(grp_vals, tuple) else grp_vals
                x_val = val
                p_val = val
            else:
                x_val = grp_vals[0] if isinstance(grp_vals, tuple) else grp_vals
                p_val = grp_vals[1] if isinstance(grp_vals, tuple) else grp_vals
            
            row_data = {x_col: x_val, "player": p_val, "ゲーム数": len(pdf)}
            if sel_graph == "レーダーチャート":
                if sel_metrics:
                    for m in sel_metrics: row_data[m] = Y_METRICS[m]["agg"](pdf)
            else:
                row_data[sel_yaxis] = Y_METRICS[sel_yaxis]["agg"](pdf)
            res.append(row_data)

        res_df = pd.DataFrame(res)
        st.markdown("<hr style='border-color: #bf953f; box-shadow: 0 0 10px rgba(191, 149, 63, 0.3);'>", unsafe_allow_html=True)

        # グラフのカラー設定をAWARD風に調整
        award_colors = ['#bf953f', '#4285f4', '#34a853', '#ea4335', '#fbbc04']
        base_layout = dict(
            template="plotly_dark",
            plot_bgcolor='rgba(10, 15, 25, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='silver', family='Arial, sans-serif'),
            xaxis=dict(showgrid=True, gridcolor='#222', zerolinecolor='#444'),
            yaxis=dict(showgrid=True, gridcolor='#222', zerolinecolor='#444')
        )

        try:
            if sel_graph == "データ表":
                st.dataframe(res_df.style.format(precision=1).set_properties(**{'background-color': '#0d1117', 'color': 'white', 'border-color': '#333'}), use_container_width=True)
                
            elif sel_graph == "棒グラフ":
                res_df = res_df.sort_values(x_col)
                fig = px.bar(res_df, x=x_col, y=sel_yaxis, color="player", barmode="group", text_auto='.1f', color_discrete_sequence=award_colors)
                if x_col != "game_num": fig.update_xaxes(type='category', categoryorder='category ascending')
                fig.update_traces(marker_line_color='rgba(255,255,255,0.8)', marker_line_width=1.5, opacity=0.85)
                fig.update_layout(**base_layout)
                st.plotly_chart(fig, use_container_width=True)
                
            elif sel_graph == "折れ線グラフ":
                res_df = res_df.sort_values(x_col)
                fig = px.line(res_df, x=x_col, y=sel_yaxis, color="player", markers=True, color_discrete_sequence=award_colors)
                if x_col != "game_num": fig.update_xaxes(type='category', categoryorder='category ascending')
                fig.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=2, color='white')))
                fig.update_layout(**base_layout)
                st.plotly_chart(fig, use_container_width=True)
                
            elif sel_graph == "分布図 (箱ひげ+散布)":
                y_col_raw = Y_METRICS[sel_yaxis]["col"]
                df_sorted = df.sort_values(x_col)
                fig = px.box(df_sorted, x=x_col, y=y_col_raw, color="player", points="all", color_discrete_sequence=award_colors)
                if x_col != "game_num": fig.update_xaxes(type='category', categoryorder='category ascending')
                fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=1, color='white')))
                fig.update_layout(**base_layout)
                st.plotly_chart(fig, use_container_width=True)
                
            elif sel_graph == "レーダーチャート":
                if not sel_metrics:
                    st.warning("比較するデータを1つ以上選択してください。")
                else:
                    fig = go.Figure()
                    for idx, row in res_df.iterrows():
                        vals = []
                        for cat in sel_metrics:
                            max_v = res_df[cat].max()
                            vals.append((row[cat] / max_v * 100) if max_v > 0 else 0)
                        
                        name_label = f'{row["player"]} ({row[x_col]})' if x_col != "player" else row["player"]
                        color = award_colors[idx % len(award_colors)]
                        fig.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]], 
                            theta=sel_metrics + [sel_metrics[0]], 
                            fill='toself', 
                            name=name_label,
                            fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in color else f"{color}33",
                            line=dict(color=color, width=2)
                        ))
                    fig.update_layout(**base_layout)
                    fig.update_layout(polar=dict(radialaxis=dict(visible=False, gridcolor='#222'), angularaxis=dict(gridcolor='#333')))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"解析エラー: 選択された組み合わせではグラフを描画できません。(詳細: {e})")

    st.stop()


# =========================================================
# 📍 【ブロック 2】 状態管理とGoogleドライブからの画像取得
# =========================================================
if "app_state" not in st.session_state:
    st.session_state.app_state = "init"
if "raw_images_data" not in st.session_state:
    st.session_state.raw_images_data = []
if "analyzed_results" not in st.session_state:
    st.session_state.analyzed_results = None
if "downloaded_images" not in st.session_state:
    st.session_state.downloaded_images = []    
if "waiting_for_scan" not in st.session_state:
    st.session_state.waiting_for_scan = False
if "last_file_id_at_click" not in st.session_state:
    st.session_state.last_file_id_at_click = None

st.markdown("<h3 style='text-align: center;'>☟　☟　☟</h3>", unsafe_allow_html=True)

st.markdown("<div class='gold-btn-marker' style='display: none;'></div>", unsafe_allow_html=True)

# --- モード別のボタン表示制御 ---
if st.session_state.get("kiosk_mode"):
    if not st.session_state.waiting_for_scan:
        fetch_button = st.button("スキャン・解析を開始する", use_container_width=True)
    else:
        fetch_button = False
        st.button("解析をキャンセルする", on_click=lambda: st.session_state.update(waiting_for_scan=False), use_container_width=True)
else:
    fetch_button = st.button("スコアシート取込（1枚）", use_container_width=True)

with st.expander("残ピン判定の微調整"):
    st.markdown("<span style='font-size: 12px; color: silver;'>自動計算された残ピン判定の閾値に、この数値をプラスマイナスして一時的に調整します。<br>（ピンが反応しにくい場合はマイナスへ、過剰に反応する場合はプラスへ変更して再取込してください）</span>", unsafe_allow_html=True)
    
    if "pin_thresh_offset" not in st.session_state:
        st.session_state.pin_thresh_offset = 1.0

    def reset_thresh():
        st.session_state.pin_thresh_offset = 1.0

    col_sl, col_btn = st.columns([4, 1])
    with col_sl:
        # value=1.0 を明示的に追加して、リセット時や初回表示時に-20.0になるのを防ぐ
        st.slider("閾値の調整値（％）", min_value=-20.0, max_value=20.0, value=1.0, step=0.05, key="pin_thresh_offset")
    with col_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        st.button("初期値に戻す", on_click=reset_thresh, use_container_width=True)
# --- ボタン押下時の処理 ---
if fetch_button:
    try:
        creds_json_str = st.secrets["google_credentials"]
        creds_info = json.loads(creds_json_str, strict=False)
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        scopes = ['https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
        drive_service = build('drive', 'v3', credentials=creds)

        # フォルダIDを取得
        folder_query = "name = 'Bowling_App' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        f_res = drive_service.files().list(q=folder_query, fields="files(id)").execute()
        folder_id = f_res.get('files', [{}])[0].get('id')

        if st.session_state.get("kiosk_mode"):
            # 【重要】ボタンを押した時点での「最新のファイルID」を記録する（ベースライン）
            q = f"'{folder_id}' in parents and mimeType='image/jpeg' and trashed=false"
            res = drive_service.files().list(q=q, orderBy="createdTime desc", pageSize=1, fields="files(id)").execute()
            items = res.get('files', [])
            st.session_state.last_file_id_at_click = items[0]['id'] if items else "none"
            st.session_state.waiting_for_scan = True
            st.rerun()
        else:
            # 通常モード：最新を1枚だけ即時取得
            q = f"'{folder_id}' in parents and mimeType='image/jpeg' and trashed=false"
            res = drive_service.files().list(q=q, orderBy="createdTime desc", pageSize=1, fields="files(id, name)").execute()
            items = res.get('files', [])
            if not items:
                st.error("画像が見つかりません。")
            else:
                item = items[0]
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, drive_service.files().get_media(fileId=item['id']))
                done = False
                while not done: _, done = downloader.next_chunk()
                st.session_state.raw_images_data = [{"name": item['name'], "bytes": fh.getvalue(), "file_id": item['id']}]
                st.session_state.analyzed_results = None
                st.rerun()
    except Exception as e:
        st.error(f"エラー: {e}")

# --- 新規画像検知（自動解析）ロジック ---
if st.session_state.get("kiosk_mode") and st.session_state.get("waiting_for_scan"):
    st.info("🔄 スキャナーでスコアシートをスキャンしてください...\n（新しく追加された画像を自動で検知して解析を開始します）")
    
    with st.spinner("画像の追加を監視中... (最大2分)"):
        try:
            creds_json_str = st.secrets["google_credentials"]
            creds_info = json.loads(creds_json_str, strict=False)
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            drive_service = build('drive', 'v3', credentials=service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive']))

            folder_query = "name = 'Bowling_App' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            f_res = drive_service.files().list(q=folder_query, fields="files(id)").execute()
            folder_id = f_res.get('files', [{}])[0].get('id')
            
            query = f"'{folder_id}' in parents and mimeType='image/jpeg' and trashed=false"
            
            # 約2分間、3秒おきにチェック
            for _ in range(40):
                res = drive_service.files().list(q=query, orderBy="createdTime desc", pageSize=1, fields="files(id, name)").execute()
                items = res.get('files', [])
                
                if items:
                    current_id = items[0]['id']
                    # 記録していたベースラインIDと異なる（＝新しいファイルが追加された）かチェック
                    if current_id != st.session_state.last_file_id_at_click:
                        item = items[0]
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, drive_service.files().get_media(fileId=current_id))
                        done = False
                        while not done: _, done = downloader.next_chunk()
                        
                        st.session_state.raw_images_data = [{"name": item['name'], "bytes": fh.getvalue(), "file_id": current_id}]
                        st.session_state.analyzed_results = None
                        st.session_state.waiting_for_scan = False
                        st.session_state.last_file_id_at_click = None # リセット
                        st.success("新しい画像を検知しました！解析を開始します。")
                        time.sleep(1)
                        st.rerun()
                        break
                time.sleep(3)
            else:
                st.session_state.waiting_for_scan = False
                st.error("タイムアウトしました。もう一度実行してください。")
                st.rerun()
        except Exception as e:
            st.session_state.waiting_for_scan = False
            st.error(f"エラー: {e}")
            st.rerun()

if not st.session_state.raw_images_data:
    st.info("　")
    st.stop()

gemini_api_key = st.secrets.get("gemini_api_key", "")
if not gemini_api_key:
    st.error("StreamlitのSecretsに 'gemini_api_key' が設定されていません。")
    st.stop()

client = genai.Client(api_key=gemini_api_key)
status_text = st.empty()

# =========================================================
# 📍 【AIプロンプトの定義】
# =========================================================
prompt_metadata = """
画像はボウリングのスコアシートの全体写真です。
この画像から「日付」「最初のゲーム数」「全体の開始時刻」「全体の終了時刻」「レーン番号」「プレイヤーネーム」および「各ゲームの開始・終了時刻」を探し出し、以下のJSON形式で出力してください。

【ルール】
1. 日付: 中央上部にある黒い文字。「YY/MM/DD」の形式で "date" に出力。
2. 最初のゲーム数: 一番上のゲームのスコア欄の左端に記載。フレームという文字の下GAMEの下に改行されて数字を記載。GAME1, GAME7, GAME13, GAME19, GAME25のいずれか。「1」などの数値のみを "start_game_num" に出力。
3. 全体の開始時刻: 1枚のスコアシートの日付の右下に記載。1ゲーム目の開始時刻と終了時刻が左右に並んでいて、その左側の時刻が開始時刻。"HH:MM" 形式で "start_time" に出力。見つからなければ "時刻不明" にする。
4. 全体の終了時刻: 1枚のスコアシートの一番最後のゲームの9フレーム目のスコア欄の上部に記載。開始時刻と終了時刻が左右に並んでいて、その右側の時刻が終了時刻。"HH:MM" 形式で "end_time" に出力。見つからなければ "時刻不明" にする。
5. レーン番号: 「ゲーム日付」の右側にある「使用レーン」の右に記載されている数字。1から18までの単独の整数か、「1-2」「3-4」「5-6」「7-8」「9-10」「11-12」「13-14」「15-16」「17-18」または、その逆の「2-1」から「18-17」までの文字列を "lane" に出力。見つからなければ空文字にする。
6. プレイヤーネーム: 一番上のゲームのスコア欄の左上の「プレーヤ ネーム」の文字の右側に書かれている名前を "player_name" に出力。見つからなければ空文字にする。
7. 各ゲームの時刻: 画像の上から順に、各ゲームごとの開始時刻と終了時刻を読み取り、配列 "games_time" に出力してください。各ゲームの時刻はスコア欄の周辺（主に9フレーム目の上部など）に記載されています。
   【重要な自己検証ステップ】
   読み取った各時刻について、以下の論理チェックを必ず行ってください。
   a. 各ゲームの開始時刻・終了時刻が、全体の「開始時刻」と「終了時刻」の間に入っているか。
   b. 同一ゲーム内で「開始時刻 ＜ 終了時刻」となっているか（開始と終了がテレコになっていないか）。
   c. 「前のゲームの終了時刻 ≦ 次のゲームの開始時刻」となっているか。
   ※もし上記チェックに1つでも矛盾（NG）がある場合、推測で大幅に時刻を捏造するのではなく、間違っている箇所を特定し、その部分の画像をもう一度よく観察して正確な数字を読み直してください。
8. Markdownの記号などは一切含めず、純粋なJSON文字列だけを出力してください。

【出力フォーマット例】
{
  "date": "26/02/07",
  "start_game_num": 1,
  "start_time": "14:12",
  "end_time": "15:30",
  "lane": "1-2",
  "player_name": "TARO",
  "games_time": [
    {
      "game_index": 1,
      "start_time": "14:12",
      "end_time": "14:25"
    },
    {
      "game_index": 2,
      "start_time": "14:25",
      "end_time": "14:40"
    }
  ]
}
"""

prompt_score = """
あなたはプロのボウリングスコア記録員です。
画像はボウリングのスコアシートから、スコア部分だけを切り取って縦に並べたものです。
以下の【ルール】に従って、フレームごとの「累計トータルスコア」を正確に読み取り、JSON形式で出力してください。

【ルール】
1. 各ゲームの行の下段にある「累計トータルスコア」の数字だけを読み取り、必ず「10個の数字の配列」にしてください。上段の投球結果（Gや数字）は絶対に読まないでください。
2．「累計トータルスコア」の欄には、数字のみで、記号はありません。6,8,9など数字は似ていて間違いやすいため、間違わないように慎重に丁寧に読み取ってください。
3. 「累計トータルスコア」の「1」は、薄い縦線（｜）のように見えることがありますが、見落とさずに「1」として読み取ってください。
4. 絶対に引き算などの計算や、推測は行わないでください。画像解析AIが青色で書いた1投目の数字も計算に用いないでください。
5. 最終的なトータルスコア（"total"）は、必ず「10フレーム目の下段の累計トータルスコア」に書かれている数字を読み取ってください。
6. 複数のゲームが写っている場合は、写っているすべてのゲームのデータを配列 "games" に出力してください。
7. Markdownの記号などは一切含めず、純粋なJSON文字列だけを出力してください。

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
        file_id = img_data.get("file_id") 
        status_text.info(f"画像 {img_idx+1}/{len(st.session_state.raw_images_data)} 枚目 ({file_name}) を解析中...")

        image_bytes = np.frombuffer(img_data["bytes"], np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.warning(f"{file_name} の画像変換に失敗しました。スキップします。")
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
        thresh_ink = cv2.adaptiveThreshold(b_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

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
            st.warning(f"{file_name}: ゲームの行（水色線）が見つかりませんでした。スキップします。")
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
                    if real_x > img_resized.shape[1] * 0.015:
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
        
        offset = st.session_state.get("pin_thresh_offset", 1.0)
        dyn_thresh_empty = dyn_thresh_empty + offset

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
                    # ▼枠線の色を黄色からオレンジ色(0, 165, 255)に変更
                    cv2.polylines(output_img, [pts_y], isClosed=True, color=(0, 165, 255), thickness=1)

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
                        # ▼斜線の色を黄色からオレンジ色(0, 165, 255)に変更
                        cv2.line(output_img, p_top_left, p_bottom_right, (0, 165, 255), 2)
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
                row_crop = img_for_ai[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                USE_MASKING = False 
                
                if USE_MASKING:
                    h_crop, w_crop = row_crop.shape[:2]
                    mask_h = int(h_crop * 0.48) 
                    cv2.rectangle(row_crop, (0, 0), (w_crop, mask_h), (255, 255, 255), -1)

                score_crops.append(row_crop)

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

        img_pil_full = Image.fromarray(cv2.cvtColor(img_color_rotated, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------
        # 📍 【ブロック 9】 AIによるテキスト読み取り（スコア → 日時）
        # ---------------------------------------------------------
        status_text.info(f"画像 {img_idx+1}: AIがスコアを読み取り中...")
        time.sleep(5) 

        ai_score_data = {"lane": "", "games": []}
        success_score = False
        last_error = ""
        used_model = "FAILED"
        max_retries = 7  

        for attempt_model in fallback_models:
            for attempt in range(max_retries):
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
                    error_msg = last_error.lower()
                    if any(err in error_msg for err in ["429", "too many requests", "quota", "503", "unavailable", "high demand", "overloaded"]):
                        if attempt < max_retries - 1:
                            wait_sec = (2 ** (attempt + 1)) + random.uniform(0, 1)
                            status_text.warning(f"サーバー高負荷/制限。{wait_sec:.1f}秒待機して再試行します... ({attempt+1}/{max_retries})")
                            time.sleep(wait_sec)
                            status_text.info(f"画像 {img_idx+1}: AIがスコアを読み取り中... (再試行 {attempt+1})")
                            continue
                    break
            if success_score:
                break

        if not success_score:
            st.warning(f"{file_name}: AIのスコア読み取りに失敗しました。理由: {last_error}")

        status_text.info(f"画像 {img_idx+1}: AIが日付・時刻・ゲーム数を取得中...")
        time.sleep(5) 

        ai_meta_data = {"date": "日付不明", "start_time": "時刻不明", "end_time": "時刻不明", "start_game_num": 1, "lane": "", "player_name": ""}
        success_meta = False
        
        for attempt_model in fallback_models:
            for attempt in range(max_retries):
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
                    success_meta = True
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    if any(err in error_msg for err in ["429", "too many requests", "quota", "503", "unavailable", "high demand", "overloaded"]):
                        if attempt < max_retries - 1:
                            wait_sec = (2 ** (attempt + 1)) + random.uniform(0, 1)
                            status_text.warning(f"API制限/高負荷(日時取得)。{wait_sec:.1f}秒待機して再試行します... ({attempt+1}/{max_retries})")
                            time.sleep(wait_sec)
                            status_text.info(f"画像 {img_idx+1}: AIが日付・時刻・ゲーム数を取得中... (再試行 {attempt+1})")
                            continue
                    break
            if success_meta:
                break
                   

        # ---------------------------------------------------------
        # 📍 【ブロック 10】 解析結果の統合とデータ整形
        # ---------------------------------------------------------
        cv2.putText(output_img, f"AI Ver: {used_model}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        if isinstance(ai_score_data, list):
            ai_score_data = {"games": ai_score_data}
        elif not isinstance(ai_score_data, dict):
            ai_score_data = {"games": []}

        global_date = str(ai_meta_data.get("date", "日付不明")).replace("-", "/")
        global_start_time = str(ai_meta_data.get("start_time", "時刻不明"))
        global_end_time = str(ai_meta_data.get("end_time", "時刻不明"))
        games_time_list = ai_meta_data.get("games_time", [])
        
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

            # AIが取得した各ゲームの時刻を取得（取得できていない場合は全体の開始/終了時刻をフォールバックとして使用）
            g_start_time = global_start_time
            g_end_time = global_end_time
            if group_idx < len(games_time_list):
                g_time_info = games_time_list[group_idx]
                if g_time_info.get("start_time") and g_time_info.get("start_time") != "時刻不明":
                    g_start_time = str(g_time_info["start_time"])
                if g_time_info.get("end_time") and g_time_info.get("end_time") != "時刻不明":
                    g_end_time = str(g_time_info["end_time"])

            row_data = [""] * 52
            row_data[0] = global_date
            row_data[1] = g_start_time
            row_data[2] = g_end_time 
            row_data[3] = str(ai_meta_data.get("lane") or "")
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

            # ▼ 追加: AIが読み取った各フレームの累計トータルスコアの描画（高さを3mm上に微調整）
            for f_tot in range(10):
                val_tot = str(ai_frame_totals[f_tot])
                if val_tot and val_tot != "0":
                    # X座標: 1投目の列の左端から左へ約2mm（位置は維持）
                    tot_x = start_x_base + f_tot * box_w - 1.0 * current_scale
                    # Y座標: 前回の +7.5 から 3mm分（3.0）引き、+4.5 に調整（一番下の線の上に乗る高さ）
                    tot_y = py1_local + ph_full + 4.5 * current_scale
                    # 文字サイズ（0.8）と色は維持
                    put_rotated_text(output_img, val_tot, tot_x, tot_y, new_ref1[0], new_ref1[1], theta, (0, 220, 0), scale=0.6, thickness=1)

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
            "file_id": file_id, 
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
    st.success("全ての画像の解析が完了しました！")

    if "dynamic_player_list" not in st.session_state or "oil_data" not in st.session_state:
        with st.spinner("SPSから設定データを取得中..."):
            try:
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

                query = "name = 'EagleBowl_ROLLERS' and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
                results = drive_service_write.files().list(q=query, fields="files(id, name)").execute()
                sheets = results.get('files', [])
                
                fetched_players = []
                player_nickname_map = {} 
                oil_data_list = [] 
                if sheets:
                    sh = gc.open_by_key(sheets[0]['id'])
                    
                    settings_sheet = sh.worksheet("プレイヤー設定")
                    settings_data = settings_sheet.get_all_values()
                    
                    for row in settings_data[1:]:
                        if len(row) >= 2 and row[1].strip():
                            p_name = row[1].strip()
                            fetched_players.append(p_name)
                            for i in range(6, len(row)):
                                nickname = row[i].strip()
                                if nickname:
                                    player_nickname_map[nickname] = p_name
                                    
                    try:
                        oil_sheet = sh.worksheet("オイル入力")
                        oil_data_raw = oil_sheet.get_all_values()
                        oil_data_list = oil_data_raw[2:] if len(oil_data_raw) > 2 else []
                    except Exception as e:
                        st.warning(f"オイル入力シートの読み込みに失敗しました: {e}")
                
                if fetched_players:
                    st.session_state.dynamic_player_list = fetched_players
                    st.session_state.player_nickname_map = player_nickname_map
                else:
                    st.session_state.dynamic_player_list = ["999_ゲスト"] 
                    st.session_state.player_nickname_map = {}
                    
                st.session_state.oil_data = oil_data_list 
            except Exception as e:
                st.warning(f"設定の読み込みに失敗しました: {e}")
                st.session_state.dynamic_player_list = ["999_ゲスト"]
                st.session_state.player_nickname_map = {}
                st.session_state.oil_data = [] 

    
    player_list = st.session_state.dynamic_player_list
    
    default_player_index = 0
    ai_player_name = ""
    if st.session_state.analyzed_results and len(st.session_state.analyzed_results) > 0:
        ai_player_name = st.session_state.analyzed_results[0].get("meta_data", {}).get("player_name", "").strip()

    if ai_player_name and "player_nickname_map" in st.session_state:
        matched_player = st.session_state.player_nickname_map.get(ai_player_name)
        if not matched_player:
            for nick, p_name in st.session_state.player_nickname_map.items():
                if ai_player_name in nick or nick in ai_player_name:
                    matched_player = p_name
                    break
        
        if matched_player and matched_player in player_list:
            default_player_index = player_list.index(matched_player)

    current_analyzed_count = len(st.session_state.analyzed_results) if st.session_state.analyzed_results else 0
    if st.session_state.get("prev_analyzed_count") != current_analyzed_count:
        st.session_state.synced_player = player_list[default_player_index] if player_list else ""
        st.session_state.prev_analyzed_count = current_analyzed_count
    elif "synced_player" not in st.session_state or st.session_state.synced_player not in player_list:
        st.session_state.synced_player = player_list[default_player_index] if player_list else ""

    if st.session_state.get("kiosk_mode"):
        default_player_index = player_list.index(st.session_state.kiosk_user) if st.session_state.kiosk_user in player_list else 0
        selected_player = st.selectbox("プレイヤー選択", player_list, index=default_player_index, disabled=True)
    else:
        selected_player = st.selectbox("プレイヤー選択", player_list, index=default_player_index)

    st.markdown("---")
    
    if "register_all_check" not in st.session_state:
        st.session_state.register_all_check = True

    def uncheck_all_if_needed(key):
        if not st.session_state[key]:
            st.session_state.register_all_check = False

    register_all = st.checkbox("全てのゲームをマスターに登録する", key="register_all_check")
    st.markdown("---")
    
    def get_oil_info(target_date, target_time, target_lane):
        oil_data = st.session_state.get("oil_data", [])
        if not oil_data or not target_date or not target_time or not target_lane:
            return "", ""
        
        lane_str = str(target_lane).replace(" ", "")
        if "-" in lane_str:
            parts = lane_str.split("-")
            try:
                lane_num = min(int(parts[0]), int(parts[1]))
            except:
                return "", ""
        else:
            try:
                lane_num = int(lane_str)
            except:
                return "", ""
                
        if not (1 <= lane_num <= 18):
            return "", ""
            
        len_col = lane_num * 2
        vol_col = lane_num * 2 + 1
        
        def to_mins(t_str):
            try:
                h, m = map(int, str(t_str).replace("：", ":").split(":"))
                return h * 60 + m
            except:
                return -1
                
        tgt_mins = to_mins(target_time)
        if tgt_mins == -1:
            return "", ""

        best_match = None
        best_mins = -1
        
        for row in oil_data:
            if len(row) <= vol_col:
                continue
            r_date = str(row[0]).strip()
            r_time = str(row[1]).strip()
            
            if r_date == target_date:
                r_mins = to_mins(r_time)
                if r_mins != -1 and tgt_mins >= r_mins:
                    if best_match is None or r_mins > best_mins:
                        best_match = row
                        best_mins = r_mins
                        
        if best_match:
            return str(best_match[len_col]).strip(), str(best_match[vol_col]).strip()
        return "", ""

    game_checkboxes = []

    for img_idx, res in enumerate(st.session_state.analyzed_results):
        st.markdown(f"#### 画像 {img_idx+1}: {res['file_name']}")
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
                match_status = "計算一致"
            else:
                # 濃い黄色（オレンジ）背景と警告マークを採用
                match_status = ":orange-background[⚠️計算不一致]"
            
            check_key = f"check_{img_idx}_{local_idx}"
            if check_key not in st.session_state:
                st.session_state[check_key] = True

            # 上の段：【ゲーム数】日時 ｜ [レ] データ登録する
            chk_col1, chk_col2 = st.columns([2.5, 7.5])
            with chk_col1:
                # 全体CSSの設定を上書きするため、色指定に !important を追加
                st.markdown(f"<div style='margin-top: 5px;'><span style='color: #00FFFF !important; font-weight: bold;'>【{game_name}】</span>{date_str}_{start_time}_{end_time} ｜</div>", unsafe_allow_html=True)
            with chk_col2:
                is_checked = st.checkbox(
                    "データ登録する", 
                    key=check_key,
                    on_change=uncheck_all_if_needed,
                    args=(check_key,)
                )

            game_checkboxes.append({
                "is_checked": is_checked,
                "export_row": row,
                "date": date_str,
                "start": start_time,
                "end": end_time,
                "img_idx": img_idx,      
                "local_idx": local_idx   
            })

            edit_key = f"edit_{img_idx}_{local_idx}"
            close_flag_key = f"close_flag_{img_idx}_{local_idx}"

            if edit_key not in st.session_state:
                st.session_state[edit_key] = False

            if st.session_state.get(close_flag_key):
                st.session_state[edit_key] = False
                st.session_state[close_flag_key] = False

            # 下の段：ゲーム名 を手動修正 ｜ トータル:〇〇_計算一致/不一致（トグルスイッチ）
            toggle_text = f"{game_name} を手動修正 ｜ トータル:{ai_total_str}_{match_status}"

            if st.toggle(toggle_text, key=edit_key):

                c_date, c_start, c_end = st.columns(3)
                if st.session_state.get("kiosk_mode"):
                    with c_date:
                        new_date = render_tenkey("日付", f"tk_d_{img_idx}_{local_idx}", row[0], format_type="date")
                    with c_start:
                        new_start = render_tenkey("開始時刻", f"tk_s_{img_idx}_{local_idx}", row[1], format_type="time")
                    with c_end:
                        new_end = render_tenkey("終了時刻", f"tk_e_{img_idx}_{local_idx}", row[2], format_type="time")
                else:
                    with c_date:
                        new_date = st.text_input("日付", value=row[0], key=f"d_{img_idx}_{local_idx}")
                    with c_start:
                        new_start = st.text_input("開始時刻", value=row[1], key=f"s_{img_idx}_{local_idx}")
                    with c_end:
                        new_end = st.text_input("終了時刻", value=row[2], key=f"e_{img_idx}_{local_idx}")

                is_710_checked = st.checkbox("7-10G (セブン-テン ゲームとして登録)", value=bool(row[51]) if len(row) > 51 else False, key=f"710_{img_idx}_{local_idx}")
                state_key = f"edit_data_{img_idx}_{local_idx}"
                active_cell_key = f"active_cell_{img_idx}_{local_idx}"

                if state_key not in st.session_state:
                    init_throws = [str(row[throw_cols[i]]).replace("R:", "") for i in range(21)]
                    init_pins = []
                    for i in range(12):
                        p_str = str(row[target_indices[i]])
                        init_pins.append([int(p) for p in p_str.split(",")] if p_str else [])
                    st.session_state[state_key] = {"throws": init_throws, "pins": init_pins}
                    st.session_state[active_cell_key] = 0  

                curr_throws = st.session_state[state_key]["throws"]
                curr_pins = st.session_state[state_key]["pins"]
                active_idx = st.session_state[active_cell_key]

                try:
                    frame_totals = calculate_bowling_score(curr_throws)
                except Exception:
                    frame_totals = []
                while len(frame_totals) < 10:
                    frame_totals.append("")

                st.markdown("""
                <style>
                /* ① マス目の余白調整と文字色（白） */
                [data-testid="stHorizontalBlock"] div[data-testid="stBlock"] button {
                    margin: 1px 0 !important;
                    padding: 2px !important;
                }
                div[data-testid="stPopover"] button p,
                [data-testid="stHorizontalBlock"] div[data-testid="stBlock"] button p {
                    font-size: 15px !important;
                    font-weight: 900 !important;
                    color: #ffffff !important;
                }
                
                .section-header {
                    font-size: 15px;
                    font-weight: 700;
                    color: silver;
                    border-bottom: 2px solid #bf953f;
                    padding-bottom: 3px;
                    margin-top: 10px;
                    margin-bottom: 10px;
                }

                /* ⑥ ポップオーバー小窓内の基本設定 */
                div[data-testid="stPopoverBody"] button {
                    min-height: 35px !important;
                }
                div[data-testid="stPopoverBody"] button p {
                    color: #ffffff !important;
                    font-weight: bold !important;
                }

                /* ▼▼▼ メイン画面のマス目（1投目/2投目）の背景・枠色 ▼▼▼ */
                /* 1投目（濃い緑） */
                div[data-testid="stColumn"]:has(.pitch-1) div[data-testid="stPopover"] > button {
                    background-color: #1a4d2a !important;
                    border: 2px solid #2e8b57 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stColumn"]:has(.pitch-1) div[data-testid="stPopover"] > button p {
                    color: #ffffff !important;
                }
                
                /* 2投目・3投目（濃い青） */
                div[data-testid="stColumn"]:has(.pitch-2) div[data-testid="stPopover"] > button {
                    background-color: #102a43 !important;
                    border: 2px solid #1e90ff !important;
                    color: #ffffff !important;
                }
                div[data-testid="stColumn"]:has(.pitch-2) div[data-testid="stPopover"] > button p {
                    color: #ffffff !important;
                }

                /* ▼▼▼ 選択して開いているマスの赤枠強調（最優先で上書きされるよう配置） ▼▼▼ */
                div[data-testid="stColumn"] div[data-testid="stPopover"] > button[aria-expanded="true"] {
                    background-color: #3a1c24 !important;
                    border: 3px solid #ff2d55 !important;
                    box-shadow: 0 0 12px #ff2d55 !important;
                }

                /* ▼▼▼ ポップオーバー小窓：スコア入力エリア ▼▼▼ */
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.score-area-marker) button[kind="secondary"] {
                    background-color: #2c3e50 !important;
                    border: 1px solid #455a64 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.score-area-marker) button[kind="secondary"]:hover {
                    background-color: #bf953f !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.score-area-marker) button[kind="secondary"]:hover p {
                    color: #000000 !important;
                }

                /* ▼▼▼ ポップオーバー小窓：残ピン切替エリア ▼▼▼ */
                /* はみ出し防止（パディングと文字サイズの調整） */
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button {
                    padding: 0px !important;
                    min-height: 32px !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button p {
                    font-size: 13px !important;
                    margin: 0 !important;
                }

                /* 残ピン無い(ターコイズブルー) */
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="secondary"] {
                    background-color: #40E0D0 !important;
                    border: 1px solid #30B0A0 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="secondary"],
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="secondary"] p {
                    color: #1a1a1c !important; /* 背景が明るいので黒文字 */
                }

                /* 残ピン有る(ピンク) */
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="primary"] {
                    background-color: #ffaaaa !important;
                    border: 1px solid #ff8888 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="primary"],
                div[data-testid="stPopoverBody"] div[data-testid="stColumn"]:has(.pin-area-marker) button[kind="primary"] p {
                    color: #1a1a1c !important; /* 背景が明るいので黒文字 */
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f"<div class='section-header'>スコアシート (修正するマスをタップ)</div>", unsafe_allow_html=True)
                
                def update_score_and_pins(idx, choice):
                    st.session_state[state_key]["throws"][idx] = choice
                    if choice == "X":
                        # ▼ Xを選択したとき、自動で残ピン無し（リストを空）に切り替える
                        if idx <= 17: pin_idx = idx // 2
                        elif idx == 18: pin_idx = 9
                        elif idx == 19: pin_idx = 10
                        else: pin_idx = 11
                        st.session_state[state_key]["pins"][pin_idx] = []

                        if idx < 18 and idx % 2 == 0:
                            st.session_state[state_key]["throws"][idx+1] = ""
                        elif idx == 18:
                            st.session_state[state_key]["throws"][19] = ""
                            st.session_state[state_key]["throws"][20] = ""
                        elif idx == 19:
                            st.session_state[state_key]["throws"][20] = ""
                
                # 💡 確実な色付けを行うためのカスタム描画関数
                def render_score_popover(col_obj, idx, bg_color):
                    val = curr_throws[idx] if curr_throws[idx] else " "
                    
                    # 1投目か2投目かを判定するクラス名（10フレ1投目は1投目扱い、それ以外は2投目扱い）
                    pitch_class = "pitch-1" if (idx % 2 == 0 and idx < 18) or idx == 18 else "pitch-2"

                    with col_obj:
                        # CSSでこのマス目を確実・安全に狙い撃ちするための目印（不可視）
                        st.markdown(f"<div class='{pitch_class}' style='display:none;'></div>", unsafe_allow_html=True)
                        
                        with st.popover(label=f"{val}", use_container_width=True):
                            st.markdown(f"**スコアと残ピンの修正**")
                            
                            p_col1, p_col2 = st.columns([1, 1.2])
                            with p_col1:
                                st.markdown("<div class='score-area-marker' style='display:none;'></div>", unsafe_allow_html=True) # スコア列用の目印
                                st.markdown("<div style='font-size:12px; color:gray; text-align:center;'>スコア</div>", unsafe_allow_html=True)
                                
                                # ▼ 手動入力テンキーの配置をご要望の通りに変更
                                def draw_btn(choice, col_layout):
                                    display_choice = "空" if choice == "" else choice
                                    # / や 空文字 がキー名にそのまま使えないためエスケープ
                                    safe_key_suffix = "slash" if choice == "/" else ("empty" if choice == "" else choice)
                                    if col_layout.button(display_choice, key=f"sel_{img_idx}_{local_idx}_{idx}_{safe_key_suffix}", use_container_width=True):
                                        update_score_and_pins(idx, choice)
                                        st.rerun()

                                # 上の段に1, 2, 3
                                r1 = st.columns(3)
                                for i, c in enumerate(["1", "2", "3"]): draw_btn(c, r1[i])
                                # 次の段に4, 5, 6
                                r2 = st.columns(3)
                                for i, c in enumerate(["4", "5", "6"]): draw_btn(c, r2[i])
                                # 次の段に7, 8, 9
                                r3 = st.columns(3)
                                for i, c in enumerate(["7", "8", "9"]): draw_btn(c, r3[i])
                                # 次の段にX, /
                                r4 = st.columns(2)
                                for i, c in enumerate(["X", "/"]): draw_btn(c, r4[i])
                                # 次の段にG, -, 空
                                r5 = st.columns(3)
                                for i, c in enumerate(["G", "-", ""]): draw_btn(c, r5[i])
                            
                            with p_col2:
                                st.markdown("<div class='pin-area-marker' style='display:none;'></div>", unsafe_allow_html=True) # 残ピン列用の目印
                                st.markdown("<div style='font-size:12px; color:gray; text-align:center;'>残ピン切替</div>", unsafe_allow_html=True)
                                if idx <= 17: pin_idx = idx // 2
                                elif idx == 18: pin_idx = 9
                                elif idx == 19: pin_idx = 10
                                else: pin_idx = 11
                                
                                active_pins = curr_pins[pin_idx]
                                
                                def toggle_pin(p_num):
                                    if p_num in st.session_state[state_key]["pins"][pin_idx]:
                                        st.session_state[state_key]["pins"][pin_idx].remove(p_num)
                                    else:
                                        st.session_state[state_key]["pins"][pin_idx].append(p_num)
                                        st.session_state[state_key]["pins"][pin_idx].sort()
                                
                                r1 = st.columns(4)
                                for i, p in enumerate([7, 8, 9, 10]):
                                    is_act = p in active_pins
                                    if r1[i].button(str(p) if is_act else " ", key=f"p_{img_idx}_{local_idx}_{idx}_{p}", type="primary" if is_act else "secondary"):
                                        toggle_pin(p); st.rerun()
                                r2 = st.columns([0.5, 1, 1, 1, 0.5])
                                for i, p in enumerate([4, 5, 6]):
                                    is_act = p in active_pins
                                    if r2[i+1].button(str(p) if is_act else " ", key=f"p_{img_idx}_{local_idx}_{idx}_{p}", type="primary" if is_act else "secondary"):
                                        toggle_pin(p); st.rerun()
                                r3 = st.columns([1, 1, 1, 1])
                                for i, p in enumerate([2, 3]):
                                    is_act = p in active_pins
                                    if r3[i+1].button(str(p) if is_act else " ", key=f"p_{img_idx}_{local_idx}_{idx}_{p}", type="primary" if is_act else "secondary"):
                                        toggle_pin(p); st.rerun()
                                r4 = st.columns([1.5, 1, 1.5])
                                is_act = 1 in active_pins
                                if r4[1].button("1" if is_act else " ", key=f"p_{img_idx}_{local_idx}_{idx}_1", type="primary" if is_act else "secondary"):
                                    toggle_pin(1); st.rerun()

                # カラー定義（緑と紺）
                COLOR_1ST = "#1b4528" # 濃い目の緑
                COLOR_2ND = "#1a2c4c" # 濃い目の紺
                COLOR_TOTAL_BG = "#4a3424" # 濃い目の茶色

                sheet_cols = st.columns(10)
                
                for f in range(9):
                    with sheet_cols[f]:
                        st.markdown(f"<div style='text-align:center; font-size:12px; font-weight:700;'>{f+1}F</div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        render_score_popover(c1, f*2, COLOR_1ST)    # 1投目は緑
                        render_score_popover(c2, f*2+1, COLOR_2ND)  # 2投目は紺
                        tot = frame_totals[f] if f < len(frame_totals) else ""
                        # ④ トータルは濃い茶色、文字白
                        st.markdown(f"<div style='text-align:center; font-weight:900; font-size:14px; background-color:{COLOR_TOTAL_BG}; color:#ffffff; border:1px solid #6b4c35; border-radius:4px; padding:2px 0; margin-top:1px;'>{tot}</div>", unsafe_allow_html=True)

                with sheet_cols[9]:
                    st.markdown(f"<div style='text-align:center; font-size:12px; font-weight:700;'>10F</div>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    render_score_popover(c1, 18, COLOR_1ST)  # 10Fの1投目は緑
                    render_score_popover(c2, 19, COLOR_2ND)  # 10Fの2投目は紺
                    render_score_popover(c3, 20, COLOR_2ND)  # 10Fの3投目も紺
                    tot = frame_totals[9] if len(frame_totals) == 10 else ""
                    # ④⑤ 10Fのみ、背景は茶色のまま枠線を金色に指定
                    st.markdown(f"<div style='text-align:center; font-weight:900; font-size:15px; background-color:{COLOR_TOTAL_BG}; color:#ffffff; border: 2px solid #bf953f; border-radius:4px; padding:2px 0; margin-top:1px;'>{tot}</div>", unsafe_allow_html=True)
                    
                st.markdown("---")
                
                st.markdown("<div class='gold-btn-marker' style='display: none;'></div>", unsafe_allow_html=True)
                if st.button("修正を完了して閉じる", key=f"update_{img_idx}_{local_idx}", type="secondary", use_container_width=True):
                    row[0] = new_date
                    row[1] = new_start
                    row[2] = new_end
                    if len(row) > 51:
                        row[51] = is_710_checked
                    else:
                        row.append(is_710_checked)
                    
                    for i in range(21):
                        row[throw_cols[i]] = curr_throws[i]
                    for i in range(12):
                        row[target_indices[i]] = ",".join(map(str, curr_pins[i]))
                    
                    if frame_totals and str(frame_totals[-1]).isdigit():
                        row[50] = str(frame_totals[-1])
                    
                    st.session_state[close_flag_key] = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### レーン・オイル・ボール")
    input_data = {}

    LANE_OPTIONS = [""] + [str(i) for i in range(1, 19)] + [f"{i}-{i+1}" for i in range(1, 19, 2)] + [f"{i+1}-{i}" for i in range(1, 19, 2)]

    games_by_img = {}
    for item in game_checkboxes:
        idx = item["img_idx"]
        if idx not in games_by_img:
            games_by_img[idx] = []
        games_by_img[idx].append(item)
        
    for img_idx, items in games_by_img.items():
        st.markdown(f"**画像 {img_idx+1} の設定**")
        
        ai_lane = items[0]["export_row"][3]
        default_lane_index = LANE_OPTIONS.index(ai_lane) if ai_lane in LANE_OPTIONS else 0
        
        c_lane, c_len, c_vol = st.columns([1.5, 1, 1])
        with c_lane:
            common_lane = st.selectbox("レーン番号", LANE_OPTIONS, index=default_lane_index, key=f"c_lane_{img_idx}")
            
        t_date = items[0]["export_row"][0]
        t_time = items[0]["export_row"][1]
        default_len, default_vol = get_oil_info(t_date, t_time, common_lane)

        with c_len:
            if st.session_state.get("kiosk_mode"):
                common_len = render_tenkey("オイル長 (ft)", f"tk_c_len_{img_idx}", default_len, format_type="none")
            else:
                common_len = st.text_input("オイル長 (ft)", value=default_len, key=f"c_len_{img_idx}", placeholder="例: 42")
        with c_vol:
            if st.session_state.get("kiosk_mode"):
                common_vol = render_tenkey("オイル量 (ml)", f"tk_c_vol_{img_idx}", default_vol, format_type="none")
            else:
                common_vol = st.text_input("オイル量 (ml)", value=default_vol, key=f"c_vol_{img_idx}", placeholder="例: 25.5")
            
        if st.session_state.get("kiosk_mode"):
            ball_options = ["", "ソリッド", "パール", "ハイブリッド", "ウレタン", "ハウスボール", "ポリエステル (スペア用)"]
            common_ball = st.selectbox("使用ボール", options=ball_options, index=0, key=f"c_ball_{img_idx}")
        else:
            common_ball = st.text_input("使用ボール", key=f"c_ball_{img_idx}", placeholder="例: ツアーダイナミクス")
            
        with st.expander(f"画像 {img_idx+1} のゲームごとの個別設定"):
            for item in items:
                l_idx = item["local_idx"]
                g_name = item["export_row"][4]
                
                st.markdown(f"**{g_name}**")
                i_c1, i_c2 = st.columns(2)
                with i_c1:
                    if st.session_state.get("kiosk_mode"):
                        i_len = render_tenkey(f"{g_name} オイル長", f"tk_i_len_{img_idx}_{l_idx}", "", format_type="none")
                    else:
                        i_len = st.text_input(f"{g_name} オイル長", key=f"i_len_{img_idx}_{l_idx}", placeholder="共通を適用")
                with i_c2:
                    if st.session_state.get("kiosk_mode"):
                        i_vol = render_tenkey(f"{g_name} オイル量", f"tk_i_vol_{img_idx}_{l_idx}", "", format_type="none")
                    else:
                        i_vol = st.text_input(f"{g_name} オイル量", key=f"i_vol_{img_idx}_{l_idx}", placeholder="共通を適用")
                
                if st.session_state.get("kiosk_mode"):
                    ball_options = ["", "ソリッド", "パール", "ハイブリッド", "ウレタン", "ポリエステル (スペア用)"]
                    i_ball = st.selectbox(f"{g_name} 使用ボール (空白は共通を適用)", options=ball_options, index=0, key=f"i_ball_{img_idx}_{l_idx}")
                else:
                    i_ball = st.text_input(f"{g_name} 使用ボール", key=f"i_ball_{img_idx}_{l_idx}", placeholder="共通を適用")
                
                final_len = i_len if i_len.strip() else common_len
                final_vol = i_vol if i_vol.strip() else common_vol
                final_ball = i_ball if i_ball.strip() else common_ball
                
                input_data[(img_idx, l_idx)] = (common_lane, final_len, final_vol, final_ball)

    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left;'>☟　☟　☟　☟　☟　☟</h3>", unsafe_allow_html=True)
    
    st.markdown(f"<h4 style='text-align: left;'>プレイヤー名：{selected_player}</h4>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    def move_images_to_processed(is_discard=False):
        msg = "画像を「取込済み画像」フォルダへ移動中..." if not is_discard else "解析を破棄し、画像を移動中..."
        with st.spinner(msg):
            try:
                creds_json_str = st.secrets["google_credentials"]
                creds_info = json.loads(creds_json_str, strict=False)
                if "private_key" in creds_info:
                    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
                scopes = ['https://www.googleapis.com/auth/drive']
                creds_move = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
                drive_service_move = build('drive', 'v3', credentials=creds_move)
                
                SOURCE_FOLDER_ID = "1PjzUPZNZYl2vKBnJjG0YVSh3NRyxlbEX"
                PROCESSED_FOLDER_NAME = "取込済み画像"

                query = f"'{SOURCE_FOLDER_ID}' in parents and name = '{PROCESSED_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                results = drive_service_move.files().list(q=query, fields="files(id, name)").execute()
                folders = results.get('files', [])
                
                if not folders:
                    folder_metadata = {
                        'name': PROCESSED_FOLDER_NAME,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [SOURCE_FOLDER_ID]
                    }
                    created_folder = drive_service_move.files().create(body=folder_metadata, fields='id').execute()
                    dest_folder_id = created_folder.get('id')
                else:
                    dest_folder_id = folders[0]['id']

                move_count = 0
                for res in st.session_state.analyzed_results:
                    fid = res.get("file_id")
                    if fid:
                        file_obj = drive_service_move.files().get(fileId=fid, fields='parents').execute()
                        previous_parents = ",".join(file_obj.get('parents', []))
                        
                        drive_service_move.files().update(
                            fileId=fid,
                            addParents=dest_folder_id,
                            removeParents=previous_parents,
                            fields='id, parents'
                        ).execute()
                        move_count += 1
                        
                if is_discard:
                    st.warning(f"解析を破棄し、{move_count}枚の画像を「{PROCESSED_FOLDER_NAME}」へ移動しました。")
                else:
                    st.success(f"スコアを登録し、{move_count}枚の画像を「{PROCESSED_FOLDER_NAME}」へ移動しました！")
                
                st.session_state.analyzed_results = None
                st.session_state.raw_images_data = []
                st.session_state.sps_registered = False

                keys_to_delete = []
                for key in st.session_state.keys():
                    if "_" in key and any(char.isdigit() for char in key):
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    del st.session_state[key]

                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"移動エラーが発生しました: {e}")

    col_reg, col_discard = st.columns([6, 1])
    
    with col_discard:
        btn_discard = st.button("破棄", help="SPSに登録せず画像を取込済みへ移動します")
    
    with col_reg:
        st.markdown("<div class='red-btn-marker' style='display: none;'></div>", unsafe_allow_html=True)
        btn_register = st.button("スコアデータを登録する", use_container_width=True, type="primary")

    if btn_discard:
        move_images_to_processed(is_discard=True)

    if btn_register:
        
        has_invalid_datetime = False
        for item in game_checkboxes:
            is_target = True if register_all else item["is_checked"]
            if is_target:
                chk_d = str(item["export_row"][0]).strip()
                chk_s = str(item["export_row"][1]).strip()
                chk_e = str(item["export_row"][2]).strip()
                
                if not chk_d or chk_d == "日付不明" or not chk_s or chk_s == "時刻不明" or not chk_e or chk_e == "時刻不明":
                    has_invalid_datetime = True
                    break
        
        if has_invalid_datetime:
            st.error("【登録エラー】 日付や開始・終了時刻が「不明」または空欄のままのデータが含まれています。対象データの「手動修正」を開き、正しい日時を入力してから再度登録ボタンを押してください。")
            st.stop() 

        with st.spinner("データを登録中..."):
            try:
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

                query = "name = 'EagleBowl_ROLLERS' and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
                results = drive_service_write.files().list(q=query, fields="files(id, name)").execute()
                sheets = results.get('files', [])
                if not sheets:
                    st.error("エラー: Googleドライブ内に「EagleBowl_ROLLERS」が見つかりません。")
                    st.stop()
                
                sheet_id = sheets[0]['id']
                sh = gc.open_by_key(sheet_id)

                BACKUP_FOLDER_ID = "1ONqsfeWmt6mT248fD7OuMhUdiqdQuoLa"

                if BACKUP_FOLDER_ID != "ここにバックアップフォルダのIDを入力":
                    try:
                        drive_service_backup = build('drive', 'v3', credentials=creds_write)
                        
                        backup_ws = sh.worksheet("バックアップ管理")
                        last_backup = backup_ws.acell('A1').value
                        next_num_str = backup_ws.acell('B1').value
                        
                        from datetime import datetime
                        now = datetime.now()
                        current_week = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
                        
                        if last_backup != current_week:
                            try:
                                next_num = int(next_num_str) if next_num_str else 1
                            except:
                                next_num = 1
                                
                            if next_num > 32:
                                next_num = 1
                                
                            target_name = f"EagleBowl_ROLLERS バックアップ{next_num:02d}"
                            
                            query = f"name = '{target_name}' and '{BACKUP_FOLDER_ID}' in parents and trashed = false"
                            results = drive_service_backup.files().list(
                                q=query, spaces='drive', supportsAllDrives=True, includeItemsFromAllDrives=True
                            ).execute()
                            items = results.get('files', [])
                            
                            if not items:
                                st.warning(f"バックアップ先ファイル「{target_name}」が見つかりませんでした。Googleドライブ上に同名のファイルが存在するか確認してください。")
                            else:
                                backup_file_id = items[0]['id']
                                backup_sh = gc.open_by_key(backup_file_id)
                                old_worksheets = backup_sh.worksheets()
                                
                                
                                copied_info = []
                                for master_ws in sh.worksheets():
                                    res = master_ws.copy_to(backup_file_id)
                                    copied_info.append({'id': res['sheetId'], 'title': master_ws.title})
                                    time.sleep(2) 
                                    
                                for old_ws in old_worksheets:
                                    backup_sh.del_worksheet(old_ws)
                                    time.sleep(2)
                                    
                                for info in copied_info:
                                    ws_to_rename = backup_sh.get_worksheet_by_id(info['id'])
                                    ws_to_rename.update_title(info['title'])
                                    time.sleep(2)
                                    
                                backup_ws.update_acell('A1', current_week)
                                backup_ws.update_acell('B1', str(next_num + 1))
                                st.info(f"今週の初回登録のため、「{target_name}」に全シートを上書きバックアップしました。")
                    except Exception as backup_e:
                        st.warning(f"バックアップ処理でエラーが発生しました（データの登録は続行します）: {backup_e}")

                player_email_map = {}
                try:
                    settings_sheet = sh.worksheet("プレイヤー設定")
                    settings_data = settings_sheet.get_all_values()
                    for row in settings_data[1:]:
                        if len(row) >= 2 and row[1]: 
                            player_email_map[row[1]] = row[0]
                except gspread.exceptions.WorksheetNotFound:
                    st.warning("「プレイヤー設定」シートが見つかりません。メールアドレスは空白で登録されます。")
                except Exception as e:
                    st.warning(f"プレイヤー設定の読み込みに失敗しました: {e}")

                user_email = player_email_map.get(selected_player, "")
                
                try:
                    worksheet = sh.worksheet("マスター")
                except gspread.exceptions.WorksheetNotFound:
                    st.error("エラー: スプレッドシート内に「マスター」という名前のシートが見つかりません。")
                    st.stop()

                existing_data = worksheet.get_all_values()
                
                rows_to_append = []
                update_count = 0
                
                if not game_checkboxes:
                    st.warning("登録対象のデータがありません。")
                    st.stop()
                
                for item in game_checkboxes:
                    is_target = True if register_all else item["is_checked"]
                    if not is_target:
                        continue

                    row = item["export_row"]
                    new_date = row[0]
                    new_start = row[1]
                    new_end = row[2]
                    new_game = row[4] 
            
                    selected_lane, oil_len, oil_vol, ball_used = input_data.get((item["img_idx"], item["local_idx"]), ("", "", "", ""))

                    formatted_row = [
                        user_email,      
                        selected_player, 
                        row[0], row[1], row[2], 
                        selected_lane,   
                        row[4],          
                        oil_len, oil_vol, ball_used, 
                    ]

                    for f in range(9):
                        formatted_row.extend([
                            row[throw_cols[f*2]],
                            row[target_indices[f]],
                            row[throw_cols[f*2+1]],
                            ""
                        ])
            
                    formatted_row.extend([
                        row[throw_cols[18]], row[target_indices[9]],
                        row[throw_cols[19]], row[target_indices[10]],
                        row[throw_cols[20]], row[target_indices[11]]
                    ])
            
                    formatted_row.append(row[50]) 
                    
                    unique_id = f"{selected_player}_{new_date}_{new_start}_{new_game}"
                    formatted_row.append(unique_id)
                    
                    is_710_flag = row[51] if len(row) > 51 else False
                    formatted_row.append("TRUE" if is_710_flag else "FALSE")
                    
                    match_found = False
                    for i, ex_row in enumerate(existing_data):
                        if i == 0 or len(ex_row) < 7: 
                            continue
                        
                        ex_player = ex_row[1]
                        ex_date = ex_row[2]
                        ex_start = ex_row[3]
                        ex_end = ex_row[4]
                        ex_game = ex_row[6] 
                        
                        if ex_player == selected_player and ex_date == new_date and (ex_start == new_start or ex_end == new_end) and ex_game == new_game:
                            row_num = i + 1
                            worksheet.update(range_name=f"A{row_num}", values=[formatted_row])
                            existing_data[i] = formatted_row
                            update_count += 1
                            match_found = True
                            break
            
                    if not match_found:
                        rows_to_append.append(formatted_row)

                if rows_to_append:
                    worksheet.append_rows(rows_to_append)
                
                add_count = len(rows_to_append)

                all_master_data = worksheet.get_all_values()
                
                data_rows = []
                for r in all_master_data[1:]:
                    if len(r) >= 53 and str(r[0]).strip():
                        is_710_game = (len(r) > 54 and str(r[54]).strip().upper() == "TRUE")
                        if not is_710_game:
                            data_rows.append(r)
                            
                def sort_key(x):
                    d = str(x[2]).strip()
                    t = str(x[3]).strip()
                    g = str(x[6]).strip()
                    g_num = g.zfill(3) if g.isdigit() else g
                    return (d, t, g_num)
                data_rows.sort(key=sort_key)
                
                named_splits = {
                    "7-10": ("2P", "スネークアイ"),
                    "2-7": ("2P", "ベビースプリット"),
                    "3-10": ("2P", "ベビースプリット"),
                    "4-6": ("2P", "フォーシックス"),
                    "4-9": ("2P", "ビッグディボット"),
                    "6-8": ("2P", "ビッグディボット"),
                    "5-7": ("2P", "ダイムストア"),
                    "5-10": ("2P", "ダイムストア"),
                    "7-9": ("2P", "ムース"),
                    "8-10": ("2P", "ムース"),
                    "5-7-10": ("3P", "リリー"),
                    "2-7-10": ("3P", "クリスマスツリー"),
                    "3-7-10": ("3P", "クリスマスツリー"),
                    "4-7-9": ("3P", "マイティマイト"),
                    "6-7-10": ("3P", "マイティマイト"),
                    "4-6-7-10": ("4_5P", "ビッグフォー"),
                    "4-6-7-8-10": ("4_5P", "グリークチャーチ"),
                    "4-6-7-9-10": ("4_5P", "ワシントン条約")
                }

                def normalize_pin(pin_str):
                    return str(pin_str).replace(",", "-").replace(" ", "").replace("'", "").replace('"', "")

                def clean_res(val):
                    v = str(val).strip().upper()
                    if "X" in v: return "X"
                    if "/" in v: return "/"
                    return v
                    
                def get_left_pins(pin_str):
                    if not pin_str: return []
                    import re
                    return [str(p) for p in re.findall(r'\d+', str(pin_str)) if 1 <= int(p) <= 10]

                player_stats = {}

                for r in data_rows:
                    email = str(r[0]).strip()
                    p_name = str(r[1]).strip()
                    play_date = str(r[2]).strip()
                    if not email: continue
                    
                    lane = str(r[5]).strip()
                    oil_len = str(r[7]).strip()
                    oil_vol = str(r[8]).strip()
                    try:
                        total_score = int(r[52])
                    except ValueError:
                        continue
                        
                    if email not in player_stats:
                        player_stats[email] = {
                            "name": p_name,
                            "last_date": "",
                            "max_str": 0, "cur_str": 0,
                            "score_300": 0, "score_250_plus": 0, "score_220_plus": 0, "score_200_plus": 0,
                            "st_chances": 0, "st_success": 0,
                            "sp_chances": 0, "sp_success": 0,
                            "pin_7_c": 0, "pin_7_s": 0,
                            "pin_10_c": 0, "pin_10_s": 0,
                            "splits": {},
                            "seq": [],
                            "euro_g": 0, "euro_s": 0,
                            "am_g": 0, "am_s": 0,
                            "euro_lanes": {str(i): {"g": 0, "s": 0} for i in range(1, 19)},
                            "am_lanes": {f"{i}-{i+1}": {"g": 0, "s": 0} for i in range(1, 18, 2)},
                            "oil_lens": {k: {"g": 0, "s": 0} for k in ["L < 32ft", "32 ≦ L < 34ft", "34 ≦ L < 36ft", "36 ≦ L < 38ft", "38 ≦ L < 40ft", "40 ≦ L < 42ft", "42 ≦ L < 44ft", "44 ≦ L < 46ft", "46ft ≦ L"]},
                            "oil_vols": {k: {"g": 0, "s": 0} for k in ["V < 20ml", "20 ≦ V < 22ml", "22 ≦ V < 24ml", "24 ≦ V < 26ml", "26 ≦ V < 28ml", "28 ≦ V < 30ml", "30 ≦ V < 32ml", "32 ≦ V < 34ml", "34 ≦ V < 36ml", "36ml ≦ V"]},
                            "first_pitch_c": 0,
                            "pin_left": {str(i): 0 for i in range(1, 11)}
                        }
                        
                    stats = player_stats[email]
                    
                    if stats["last_date"] != play_date:
                        stats["cur_str"] = 0
                        stats["last_date"] = play_date
                        
                    if total_score == 300: stats["score_300"] += 1
                    if total_score >= 250: stats["score_250_plus"] += 1
                    if total_score >= 220: stats["score_220_plus"] += 1
                    if total_score >= 200: stats["score_200_plus"] += 1
                        
                    if "-" in lane:
                        stats["am_g"] += 1; stats["am_s"] += total_score
                    elif lane:
                        stats["euro_g"] += 1; stats["euro_s"] += total_score
                        
                    if lane:
                        import re
                        nums = [int(x) for x in re.findall(r'\d+', lane)]
                        if "-" in lane and len(nums) == 2:
                            n1, n2 = min(nums), max(nums)
                            if n1 % 2 == 1 and n2 == n1 + 1:
                                k = f"{n1}-{n2}"
                                if k in stats["am_lanes"]:
                                    stats["am_lanes"][k]["g"] += 1; stats["am_lanes"][k]["s"] += total_score
                        elif len(nums) == 1:
                            k = str(nums[0])
                            if k in stats["euro_lanes"]:
                                stats["euro_lanes"][k]["g"] += 1; stats["euro_lanes"][k]["s"] += total_score
                        
                    if oil_len:
                        try:
                            olen = float(oil_len)
                            if olen < 32: k = "L < 32ft"
                            elif olen >= 46: k = "46ft ≦ L"
                            else:
                                lower = int((olen - 32) // 2) * 2 + 32
                                k = f"{lower} ≦ L < {lower+2}ft"
                            if k in stats["oil_lens"]:
                                stats["oil_lens"][k]["g"] += 1; stats["oil_lens"][k]["s"] += total_score
                        except ValueError:
                            pass
                        
                    if oil_vol:
                        try:
                            ovol = float(oil_vol)
                            if ovol < 20: k = "V < 20ml"
                            elif ovol >= 36: k = "36ml ≦ V"
                            else:
                                lower = int((ovol - 20) // 2) * 2 + 20
                                k = f"{lower} ≦ V < {lower+2}ml"
                            if k in stats["oil_vols"]:
                                stats["oil_vols"][k]["g"] += 1; stats["oil_vols"][k]["s"] += total_score
                        except ValueError:
                            pass

                    game_seq = []
                    
                    for f in range(9):
                        res1 = clean_res(r[10 + f*4])
                        pin1 = str(r[11 + f*4]).strip()
                        res2 = clean_res(r[12 + f*4])
                        
                        game_seq.append(res1)
                        stats["st_chances"] += 1
                        stats["first_pitch_c"] += 1
                        
                        if res1 == "X":
                            stats["st_success"] += 1
                            stats["cur_str"] += 1
                            if stats["cur_str"] > stats["max_str"]: stats["max_str"] = stats["cur_str"]
                        else:
                            stats["cur_str"] = 0
                            for p in get_left_pins(pin1): stats["pin_left"][p] += 1
                            stats["sp_chances"] += 1
                            if res2 == "/": stats["sp_success"] += 1
                            
                            p_str = normalize_pin(pin1)
                            if p_str == "7":
                                stats["pin_7_c"] += 1
                                if res2 == "/": stats["pin_7_s"] += 1
                            elif p_str == "10":
                                stats["pin_10_c"] += 1
                                if res2 == "/": stats["pin_10_s"] += 1
                            elif p_str in named_splits: 
                                if p_str not in stats["splits"]: stats["splits"][p_str] = {"c": 0, "s": 0}
                                stats["splits"][p_str]["c"] += 1
                                if res2 == "/": stats["splits"][p_str]["s"] += 1
                                
                            game_seq.append(res2)

                    res10_1 = clean_res(r[46]) if len(r) > 46 else ""
                    pin10_1 = str(r[47]).strip() if len(r) > 47 else ""
                    res10_2 = clean_res(r[48]) if len(r) > 48 else ""
                    pin10_2 = str(r[49]).strip() if len(r) > 49 else ""
                    res10_3 = clean_res(r[50]) if len(r) > 50 else ""
                    pin10_3 = str(r[51]).strip() if len(r) > 51 else ""
                    
                    game_seq.append(res10_1)
                    stats["st_chances"] += 1
                    stats["first_pitch_c"] += 1
                    
                    if res10_1 == "X":
                        stats["st_success"] += 1
                        stats["cur_str"] += 1
                        if stats["cur_str"] > stats["max_str"]: stats["max_str"] = stats["cur_str"]
                        
                        game_seq.append(res10_2)
                        stats["st_chances"] += 1
                        stats["first_pitch_c"] += 1
                        
                        if res10_2 == "X":
                            stats["st_success"] += 1
                            stats["cur_str"] += 1
                            if stats["cur_str"] > stats["max_str"]: stats["max_str"] = stats["cur_str"]
                            
                            game_seq.append(res10_3)
                            stats["st_chances"] += 1
                            stats["first_pitch_c"] += 1
                            
                            if res10_3 == "X":
                                stats["st_success"] += 1
                                stats["cur_str"] += 1
                                if stats["cur_str"] > stats["max_str"]: stats["max_str"] = stats["cur_str"]
                            else:
                                stats["cur_str"] = 0
                                for p in get_left_pins(pin10_3): stats["pin_left"][p] += 1
                        else:
                            stats["cur_str"] = 0
                            for p in get_left_pins(pin10_2): stats["pin_left"][p] += 1
                            stats["sp_chances"] += 1
                            if res10_3 == "/": stats["sp_success"] += 1
                            game_seq.append(res10_3)
                            
                            p_str = normalize_pin(pin10_2)
                            if p_str == "7":
                                stats["pin_7_c"] += 1
                                if res10_3 == "/": stats["pin_7_s"] += 1
                            elif p_str == "10":
                                stats["pin_10_c"] += 1
                                if res10_3 == "/": stats["pin_10_s"] += 1
                            elif p_str in named_splits: 
                                if p_str not in stats["splits"]: stats["splits"][p_str] = {"c": 0, "s": 0}
                                stats["splits"][p_str]["c"] += 1
                                if res10_3 == "/": stats["splits"][p_str]["s"] += 1
                    else:
                        stats["cur_str"] = 0
                        for p in get_left_pins(pin10_1): stats["pin_left"][p] += 1
                        stats["sp_chances"] += 1
                        if res10_2 == "/":
                            stats["sp_success"] += 1
                            game_seq.append(res10_2)
                            game_seq.append(res10_3)
                            stats["st_chances"] += 1
                            stats["first_pitch_c"] += 1
                            if res10_3 == "X":
                                stats["st_success"] += 1
                                stats["cur_str"] += 1
                                if stats["cur_str"] > stats["max_str"]: stats["max_str"] = stats["cur_str"]
                            else:
                                for p in get_left_pins(pin10_3): stats["pin_left"][p] += 1
                        else:
                            game_seq.append(res10_2)
                            
                        p_str = normalize_pin(pin10_1)
                        if p_str == "7":
                            stats["pin_7_c"] += 1
                            if res10_2 == "/": stats["pin_7_s"] += 1
                        elif p_str == "10":
                            stats["pin_10_c"] += 1
                            if res10_2 == "/": stats["pin_10_s"] += 1
                        elif p_str in named_splits: 
                            if p_str not in stats["splits"]: stats["splits"][p_str] = {"c": 0, "s": 0}
                            stats["splits"][p_str]["c"] += 1
                            if res10_2 == "/": stats["splits"][p_str]["s"] += 1

                    stats["seq"].append(game_seq)

                award_rows = [["メールアドレス", "プレイヤー名", "カテゴリ", "項目", "母数", "成功数_合計スコア", "確率_アベレージ_回数"]]
                
                def calc_rate(s, c): return round((s / c) * 100, 1) if c > 0 else 0
                def calc_ave(s, g): return round(s / g, 1) if g > 0 else 0
                
                for email, stats in player_stats.items():
                    n = stats["name"]
                    award_rows.append([email, n, "1.記録", "①最大連続ストライク", "-", "-", stats["max_str"]])
                    award_rows.append([email, n, "1.記録", "①パーフェクト(300)", "-", "-", stats["score_300"]])
                    award_rows.append([email, n, "1.記録", "①250オーバー", "-", "-", stats["score_250_plus"]])
                    award_rows.append([email, n, "1.記録", "①220オーバー", "-", "-", stats["score_220_plus"]])
                    award_rows.append([email, n, "1.記録", "①200オーバー", "-", "-", stats["score_200_plus"]])
                    award_rows.append([email, n, "2.全体率", "②1投目ストライク率", stats["st_chances"], stats["st_success"], calc_rate(stats["st_success"], stats["st_chances"])])
                    award_rows.append([email, n, "2.全体率", "③2投目スペア率", stats["sp_chances"], stats["sp_success"], calc_rate(stats["sp_success"], stats["sp_chances"])])
                    award_rows.append([email, n, "3.特定ピン", "④7番ピン", stats["pin_7_c"], stats["pin_7_s"], calc_rate(stats["pin_7_s"], stats["pin_7_c"])])
                    award_rows.append([email, n, "3.特定ピン", "⑤10番ピン", stats["pin_10_c"], stats["pin_10_s"], calc_rate(stats["pin_10_s"], stats["pin_10_c"])])
                        
                    split_stats = {
                        "2P": {}, "3P": {}, "4_5P": {}
                    }
                    
                    for sp, (grp, name) in named_splits.items():
                        name_label = f"⑥{name} ({sp})"
                        split_stats[grp][name_label] = {"c": 0, "s": 0}
                            
                    for sp, d in stats["splits"].items():
                        if sp in named_splits:
                            grp, name = named_splits[sp]
                            name_label = f"⑥{name} ({sp})"
                            split_stats[grp][name_label]["c"] += d["c"]
                            split_stats[grp][name_label]["s"] += d["s"]
                            
                    for g_key, g_name in [("2P", "4.2Pスプリット"), ("3P", "4.3Pスプリット"), ("4_5P", "4.4・5Pスプリット")]:
                        for name_label, d in split_stats[g_key].items():
                            award_rows.append([email, n, g_name, name_label, d["c"], d["s"], calc_rate(d["s"], d["c"])])

                    st_after_st_c, st_after_st_s, st_after_db_c, st_after_db_s = 0, 0, 0, 0
                    for game_record in stats["seq"]:
                        for i in range(len(game_record) - 1):
                            if game_record[i] == "X":
                                st_after_st_c += 1
                                if game_record[i+1] == "X": st_after_st_s += 1
                        for i in range(len(game_record) - 2):
                            if game_record[i] == "X" and game_record[i+1] == "X":
                                st_after_db_c += 1
                                if game_record[i+2] == "X": st_after_db_s += 1
                                
                    award_rows.append([email, n, "5.連発率", "⑦ストライク後のストライク", st_after_st_c, st_after_st_s, calc_rate(st_after_st_s, st_after_st_c)])
                    award_rows.append([email, n, "5.連発率", "⑧ダブル後のストライク", st_after_db_c, st_after_db_s, calc_rate(st_after_db_s, st_after_db_c)])
                    award_rows.append([email, n, "6.投球方式", "⑨1レーン", stats["euro_g"], stats["euro_s"], calc_ave(stats["euro_s"], stats["euro_g"])])
                    award_rows.append([email, n, "6.投球方式", "⑨2レーン", stats["am_g"], stats["am_s"], calc_ave(stats["am_s"], stats["am_g"])])
                        
                    for i in range(1, 19):
                        k = str(i)
                        d = stats["euro_lanes"][k]
                        award_rows.append([email, n, "7.レーン別", f"⑩{k}レーン", d["g"], d["s"], calc_ave(d["s"], d["g"])])
                    for i in range(1, 18, 2):
                        k = f"{i}-{i+1}"
                        d = stats["am_lanes"][k]
                        award_rows.append([email, n, "7.レーン別", f"⑩{i}-{i+1}・{i+1}-{i} レーン", d["g"], d["s"], calc_ave(d["s"], d["g"])])
                        
                    for l_key, d in stats["oil_lens"].items():
                        award_rows.append([email, n, "8.オイル長別", f"⑪{l_key}", d["g"], d["s"], calc_ave(d["s"], d["g"])])
                    for v_key, d in stats["oil_vols"].items():
                        award_rows.append([email, n, "9.オイル量別", f"⑫{v_key}", d["g"], d["s"], calc_ave(d["s"], d["g"])])

                    for i in range(1, 11):
                        k = str(i)
                        c = stats["first_pitch_c"]
                        s = stats["pin_left"][k]
                        award_rows.append([email, n, "10.残存率", f"⑬{i}番ピン残存率", c, s, calc_rate(s, c)])

                try:
                    award_sheet = sh.worksheet("AWARD")
                except gspread.exceptions.WorksheetNotFound:
                    award_sheet = sh.add_worksheet(title="AWARD", rows="1000", cols="7")
                    
                award_sheet.clear()
                if award_rows:
                    award_sheet.update(range_name="A1", values=award_rows)

                st.success(f"登録完了！ 新規追加: {add_count}件 / 上書き更新: {update_count}件")
                st.session_state.sps_registered = True 
                
                # 登録成功時にテンキーの入力履歴をリセット
                for k in list(st.session_state.keys()):
                    if k.startswith("tk_"): del st.session_state[k]
                
                # キオスクモードの場合、登録完了後にSTATS画面へ遷移
                if st.session_state.get("kiosk_mode"):
                    st.session_state.kiosk_step = "stats"

                move_images_to_processed(is_discard=False)

            except Exception as e:
                st.error(f"SPSへの登録中にエラーが発生しました: {e}")








