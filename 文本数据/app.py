from flask import Flask, render_template, jsonify, request
import pandas as pd
import re
from datetime import datetime
import os
import traceback

app = Flask(__name__)

# --- 配置区域 ---
# 您的日志路径
CUSTOM_LOG_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\7828E22EBB043D0D00654BFB60323FE9\Tester\Logs\20251128.log"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_file_path():
    if CUSTOM_LOG_PATH and os.path.exists(CUSTOM_LOG_PATH):
        return CUSTOM_LOG_PATH
    local_data = os.path.join(BASE_DIR, 'data.txt')
    if os.path.exists(local_data):
        return local_data
    return CUSTOM_LOG_PATH if CUSTOM_LOG_PATH else local_data


# --- 数据解析函数 ---
def parse_data(file_path):
    data_list = []

    # --- 升级版正则 ---
    # 解释:
    # 1. Core\s+\d+ : 匹配 Core 1 或 Core 01
    # 2. .*? : 非贪婪匹配任意字符 (忽略中间的 Tab 或空格)
    # 3. (\d{4}...) : 抓取日期时间
    # 4. .*?ask: : 忽略日期和 ask 之间的任何东西 (比如序列号 5627)
    # 5. re.IGNORECASE : 忽略大小写 (Ask/ask 都能行)
    pattern = re.compile(r'Core\s+\d+.*?(\d{4}\.\d{2}\.\d{2}\s\d{2}:\d{2}:\d{2}).*?ask:\s*([\d.]+).*?bid:\s*([\d.]+)',
                         re.IGNORECASE)

    encodings_to_try = ['utf-16', 'utf-8', 'gbk', 'latin-1']
    lines = []

    print(f"正在尝试读取文件: {file_path}")

    if not os.path.exists(file_path):
        print("错误: 文件不存在！")
        return pd.DataFrame()

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            print(f"成功使用 [{enc}] 编码读取文件，共 {len(lines)} 行")
            break
        except UnicodeError:
            continue
        except Exception as e:
            print(f"读取错误: {e}")
            break

    if not lines:
        return pd.DataFrame()

    # 解析内容
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            quote_time_str = match.group(1)
            ask = float(match.group(2))
            bid = float(match.group(3))
            mid_price = (ask + bid) / 2

            data_list.append({
                'timestamp': datetime.strptime(quote_time_str, '%Y.%m.%d %H:%M:%S'),
                'price': mid_price
            })
        else:
            # --- 调试: 如果前3行都匹配失败，打印出来看看长什么样 ---
            if i < 3:
                print(f"Debug [第{i + 1}行匹配失败]: {repr(line)}")

    if not data_list:
        print("警告: 未匹配到数据。请查看上方 'Debug' 日志，确认日志格式与正则是否差异太大。")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    print(f"成功解析 {len(df)} 条 Tick 数据")
    return df


# --- 数据处理 ---
def process_market_data(period, ma_period, ema_period):
    file_path = get_data_file_path()
    df = parse_data(file_path)
    if df.empty: return None

    try:
        safe_period = period.replace('min', 'T')

        ohlc = df['price'].resample(safe_period).ohlc()
        volume = df['price'].resample(safe_period).count()
        ohlc['volume'] = volume
        ohlc.dropna(inplace=True)

        if ohlc.empty: return None

        # 指标计算
        ohlc['MA'] = ohlc['close'].rolling(window=int(ma_period)).mean()
        ohlc['EMA'] = ohlc['close'].ewm(span=int(ema_period), adjust=False).mean()

        # BOLL
        period_boll = 20
        std_dev = 2
        ohlc['BOLL_MID'] = ohlc['close'].rolling(window=period_boll).mean()
        ohlc['BOLL_STD'] = ohlc['close'].rolling(window=period_boll).std()
        ohlc['BOLL_UP'] = ohlc['BOLL_MID'] + (ohlc['BOLL_STD'] * std_dev)
        ohlc['BOLL_LOW'] = ohlc['BOLL_MID'] - (ohlc['BOLL_STD'] * std_dev)

        # MACD
        exp1 = ohlc['close'].ewm(span=12, adjust=False).mean()
        exp2 = ohlc['close'].ewm(span=26, adjust=False).mean()
        ohlc['MACD_DIF'] = exp1 - exp2
        ohlc['MACD_DEA'] = ohlc['MACD_DIF'].ewm(span=9, adjust=False).mean()
        ohlc['MACD_HIST'] = 2 * (ohlc['MACD_DIF'] - ohlc['MACD_DEA'])

        # RSI
        delta = ohlc['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        ohlc['RSI'] = 100 - (100 / (1 + rs))

        ohlc['date_str'] = ohlc.index.strftime('%Y-%m-%d %H:%M:%S')
        return ohlc
    except Exception as e:
        traceback.print_exc()
        return None


# --- 路由 ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/kline')
def get_kline_data():
    try:
        period = request.args.get('period', '1min')
        ma_val = request.args.get('ma', 10)
        ema_val = request.args.get('ema', 20)

        df = process_market_data(period, ma_val, ema_val)

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '未生成有效数据，请检查 PyCharm 控制台的 Debug 日志'})

        def clean_nan(series):
            return [val if not pd.isna(val) else None for val in series]

        return jsonify({
            'status': 'success',
            'categoryData': df['date_str'].tolist(),
            'values': df[['open', 'close', 'low', 'high', 'volume']].values.tolist(),
            'maData': clean_nan(df['MA']),
            'emaData': clean_nan(df['EMA']),
            'boll': {
                'up': clean_nan(df['BOLL_UP']),
                'mid': clean_nan(df['BOLL_MID']),
                'low': clean_nan(df['BOLL_LOW'])
            },
            'macd': {
                'dif': clean_nan(df['MACD_DIF']),
                'dea': clean_nan(df['MACD_DEA']),
                'hist': clean_nan(df['MACD_HIST'])
            },
            'rsi': clean_nan(df['RSI'])
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("应用启动中...")
    app.run(debug=True, port=5000)