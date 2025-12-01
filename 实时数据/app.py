from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import traceback
import threading
import time
import logging
import requests
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- 配置区域 ---
DEFAULT_SYMBOL = "XAUUSD"
HISTORY_HOURS = 0.5

# --- Gemini API 配置 ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAHDVU2cA_YD3JZHn7daDa-FoegibM5A1Q")

# [修改] 根据您的日志，切换到 gemini-2.0-flash
GEMINI_MODEL = "gemini-2.0-flash"

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# 禁用 Flask 日志刷屏
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# --- 全局状态管理 ---
class GlobalState:
    def __init__(self):
        self.period = '5s'
        self.ma_period = 10
        self.ema_period = 20
        self.boll_period = 20
        self.boll_std = 2.0
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_data_context = {}

    def update(self, data):
        with self.lock:
            self.period = data.get('period', self.period)
            self.ma_period = int(data.get('ma', self.ma_period))
            self.ema_period = int(data.get('ema', self.ema_period))
            self.boll_period = int(data.get('bollP', self.boll_period))
            self.boll_std = float(data.get('bollS', self.boll_std))

    def set_latest_context(self, context):
        with self.lock:
            self.latest_data_context = context

    def get_latest_context(self):
        with self.lock:
            return self.latest_data_context


state = GlobalState()


# --- MT5 逻辑 ---
def get_mt5_data(symbol, lookback_hours):
    if not mt5.initialize():
        return None

    target_symbol = symbol
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        alternatives = ["XAU", "GOLD", "XAUUSD.m", "Gold", "XAUUSD+", "XAUUSD.pro", "XAUUSD_i"]
        for alt in alternatives:
            tick_info = mt5.symbol_info_tick(alt)
            if tick_info is not None:
                target_symbol = alt
                break

    if tick_info is None: return None

    last_tick_time = datetime.fromtimestamp(tick_info.time)
    date_from = last_tick_time - timedelta(hours=lookback_hours)
    date_to = last_tick_time + timedelta(minutes=1)

    ticks = mt5.copy_ticks_range(target_symbol, date_from, date_to, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0: return None

    df = pd.DataFrame(ticks)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['price'] = (df['ask'] + df['bid']) / 2.0
    df.set_index('timestamp', inplace=True)
    return df


def calculate_data():
    with state.lock:
        p_period = state.period
        p_ma = state.ma_period
        p_ema = state.ema_period
        p_boll_p = state.boll_period
        p_boll_s = state.boll_std

    try:
        df = get_mt5_data(DEFAULT_SYMBOL, HISTORY_HOURS)
        if df is None or df.empty: return None

        if 'min' in p_period:
            safe_period = p_period.replace('min', 'T')
        else:
            safe_period = p_period.lower()

        ohlc = df['price'].resample(safe_period).ohlc()
        volume = df['price'].resample(safe_period).count()
        ohlc['volume'] = volume
        ohlc = ohlc.dropna()

        if ohlc.empty: return None

        ohlc['MA'] = ohlc['close'].rolling(window=p_ma).mean()
        ohlc['EMA'] = ohlc['close'].ewm(span=p_ema, adjust=False).mean()

        ohlc['BOLL_MID'] = ohlc['close'].rolling(window=p_boll_p).mean()
        ohlc['BOLL_STD'] = ohlc['close'].rolling(window=p_boll_p).std(ddof=0)
        ohlc['BOLL_UP'] = ohlc['BOLL_MID'] + (ohlc['BOLL_STD'] * p_boll_s)
        ohlc['BOLL_LOW'] = ohlc['BOLL_MID'] - (ohlc['BOLL_STD'] * p_boll_s)

        exp1 = ohlc['close'].ewm(span=12, adjust=False).mean()
        exp2 = ohlc['close'].ewm(span=26, adjust=False).mean()
        ohlc['MACD_DIF'] = exp1 - exp2
        ohlc['MACD_DEA'] = ohlc['MACD_DIF'].ewm(span=9, adjust=False).mean()
        ohlc['MACD_HIST'] = 2 * (ohlc['MACD_DIF'] - ohlc['MACD_DEA'])

        delta = ohlc['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        ohlc['RSI'] = 100 - (100 / (1 + rs))

        ohlc['date_str'] = ohlc.index.strftime('%Y-%m-%d %H:%M:%S')

        last_bar = ohlc.iloc[-1]
        context = {
            'time': ohlc.index[-1].strftime('%H:%M:%S'),
            'price': last_bar['close'],
            'ma': last_bar['MA'],
            'ema': last_bar['EMA'],
            'boll_up': last_bar['BOLL_UP'],
            'boll_low': last_bar['BOLL_LOW'],
            'rsi': last_bar['RSI'],
            'macd': last_bar['MACD_HIST'],
            'period': p_period
        }
        state.set_latest_context(context)

        return ohlc.tail(300)
    except:
        return None


def clean_nan(data):
    if isinstance(data, pd.Series):
        return data.astype(object).where(pd.notnull(data), None).tolist()
    return data


# --- 后台广播线程 ---
def background_thread():
    print("Background thread started (Threading Mode)")
    while True:
        socketio.sleep(0.1)
        try:
            data_df = calculate_data()

            if data_df is not None:
                response = {
                    'status': 'success',
                    'categoryData': data_df['date_str'].tolist(),
                    'values': data_df[['open', 'close', 'low', 'high', 'volume']].values.tolist(),
                    'maData': clean_nan(data_df['MA']),
                    'emaData': clean_nan(data_df['EMA']),
                    'boll': {
                        'up': clean_nan(data_df['BOLL_UP']),
                        'low': clean_nan(data_df['BOLL_LOW'])
                    },
                    'macd': {
                        'dif': clean_nan(data_df['MACD_DIF']),
                        'dea': clean_nan(data_df['MACD_DEA']),
                        'hist': clean_nan(data_df['MACD_HIST'])
                    },
                    'rsi': clean_nan(data_df['RSI'])
                }
                socketio.emit('market_update', response)
        except Exception as e:
            time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


# --- Gemini 调用函数 (更新后的模型列表逻辑) ---
def call_gemini_api(context):
    if not GEMINI_API_KEY or len(GEMINI_API_KEY) < 10 or "在此处粘贴" in GEMINI_API_KEY:
        return "⚠️ 错误: 请先在 app.py 文件中配置您的 Google API Key。"

    prompt = f"""
    你是一个专业的黄金(XAUUSD)日内交易员。现在是 {context.get('time')}。
    请根据以下 {context.get('period')} 周期的实时指标数据进行分析：

    - 现价: {context.get('price'):.2f}
    - 均线: MA={context.get('ma'):.2f}, EMA={context.get('ema'):.2f}
    - 布林带: 上轨={context.get('boll_up'):.2f}, 下轨={context.get('boll_low'):.2f}
    - 动能: RSI={context.get('rsi'):.2f}, MACD={context.get('macd'):.4f}

    请生成一份简短的中文分析（100字左右）：
    1. 现在的趋势是多头、空头还是震荡？
    2. 此时此刻应该做多、做空还是观望？为什么？
    3. 给出激进和稳健两个建议点位。
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    max_retries = 3
    # [核心修改] 默认使用 gemini-2.0-flash，这是您列表里明确支持的模型
    current_model_url = GEMINI_API_URL

    for attempt in range(max_retries):
        try:
            response = requests.post(current_model_url, json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "AI 思考中... (未返回有效内容)"

            elif response.status_code == 429:
                wait_time = (attempt + 1) * 2
                print(f"API 限流 (429)，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue

            elif response.status_code == 404:
                print(f"模型 {GEMINI_MODEL} 404。尝试切换模型...")
                # 自动降级/切换逻辑
                if "2.0" in current_model_url:
                    # 如果 2.0 失败，尝试 2.5
                    current_model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                elif "2.5" in current_model_url:
                    # 如果 2.5 也失败，尝试 2.0-flash-001
                    current_model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={GEMINI_API_KEY}"
                else:
                    return f"模型错误 (404): 请检查您的 API Key 权限。"
                continue

            else:
                return f"API 请求失败: {response.status_code} - {response.text}"

        except Exception as e:
            print(f"网络请求错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(1)

    return "⚠️ 暂时无法连接 AI 服务，请检查网络或 API Key。"


# --- WebSocket 事件处理 ---
@socketio.on('connect')
def handle_connect():
    if not state.is_running:
        state.is_running = True
        socketio.start_background_task(background_thread)
    print('Client connected')


@socketio.on('update_settings')
def handle_settings(data):
    state.update(data)


@socketio.on('request_ai_analysis')
def handle_ai_request():
    print("收到 AI 分析请求...")
    context = state.get_latest_context()
    if not context:
        emit('ai_analysis_result', {'text': '数据暂未准备好，请等待图表加载...'})
        return

    analysis_text = call_gemini_api(context)
    emit('ai_analysis_result', {'text': analysis_text})


if __name__ == '__main__':
    print("正在连接 MT5...")
    if mt5.initialize():
        print("MT5 连接成功! 启动 WebSocket 服务 (Threading Mode)...")
        socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
    else:
        print("严重错误: 无法连接到 MT5 客户端。")