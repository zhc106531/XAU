from flask import Flask, render_template, jsonify, request
import pandas as pd
import re
from datetime import datetime, timedelta
import os
import traceback
import numpy as np

app = Flask(__name__)

# --- 配置区域 ---
# 默认读取本地 data.txt，如果不存在则尝试读取指定路径
CUSTOM_LOG_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\7828E22EBB043D0D00654BFB60323FE9\Tester\Logs\20251203.log"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_file_path():
    # 优先检查项目目录下的 data.txt
    local_data = os.path.join(BASE_DIR, 'data.txt')
    if os.path.exists(local_data):
        return local_data
    # 其次检查配置的绝对路径
    if CUSTOM_LOG_PATH and os.path.exists(CUSTOM_LOG_PATH):
        return CUSTOM_LOG_PATH
    return None


def parse_ea_log(file_path):
    """
    专门解析 EA 生成的日志格式
    """
    kline_list = []

    # 临时存储交易相关
    deals_map = {}  # deal_id -> {time, price, type, ticket, direction}
    positions = {}  # ticket -> {open: deal_info, close: deal_info}

    # 正则表达式预编译
    # 1. K线: 2025.11.27 20:59:50   [20:59:30] O:4154.71 H:4154.96 L:4154.60 C:4154.77
    p_kline = re.compile(
        r'(\d{4}\.\d{2}\.\d{2})\s+.*\[(\d{2}:\d{2}:\d{2})\].*O:([\d.]+)\s+H:([\d.]+)\s+L:([\d.]+)\s+C:([\d.]+)')

    # 2. Deal (交易执行): 2025.11.27 21:00:31   deal #25 buy 0.01 XAU at 4151.45 done
    p_deal = re.compile(
        r'(\d{4}\.\d{2}\.\d{2})\s+(\d{2}:\d{2}:\d{2}).*?deal #(\d+)\s+(buy|sell)\s+.*?at\s+([\d.]+)\s+done')

    # 3. 平仓意图: market buy ... close #24
    p_close_intent = re.compile(r'market\s+(buy|sell).*?close\s+#(\d+)')

    # 4. 显式平仓确认: >> Position Closed: ... Ticket:24
    p_closed_confirm = re.compile(r'>>\s+Position\s+Closed:.*?Ticket:(\d+)')

    print(f"正在读取文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-16') as f:  # EA日志常用UTF-16
            lines = f.readlines()
    except:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取失败: {e}")
            return pd.DataFrame(), {}

    pending_close_ticket = None

    # 1. 第一遍扫描：解析所有 K 线
    for line in lines:
        m_kline = p_kline.search(line)
        if m_kline:
            date_str, time_str, o, h, l, c = m_kline.groups()
            full_time_str = f"{date_str.replace('.', '-')} {time_str}"
            kline_list.append({
                'timestamp': datetime.strptime(full_time_str, '%Y-%m-%d %H:%M:%S'),
                'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c),
                'date_str': full_time_str
            })

    if not kline_list:
        return pd.DataFrame(), {}

    df = pd.DataFrame(kline_list)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # 辅助函数：找到最近的 K 线时间（吸附交易点到 K 线）
    valid_times = df.index.sort_values()

    def snap_to_kline(trade_dt):
        idx = valid_times.searchsorted(trade_dt, side='right') - 1
        if idx >= 0:
            return valid_times[idx].strftime('%Y-%m-%d %H:%M:%S')
        return trade_dt.strftime('%Y-%m-%d %H:%M:%S')

    # 2. 第二遍扫描：解析交易
    for line in lines:
        m_ci = p_close_intent.search(line)
        if m_ci:
            pending_close_ticket = m_ci.group(2)
            continue

        m_deal = p_deal.search(line)
        if m_deal:
            date_str, time_str, deal_id, side, price = m_deal.groups()
            dt = datetime.strptime(f"{date_str.replace('.', '-')} {time_str}", '%Y-%m-%d %H:%M:%S')
            snap_time = snap_to_kline(dt)

            deal_info = {
                'time': snap_time,
                'real_time': dt,
                'price': float(price),
                'side': side,
                'deal_id': deal_id
            }

            if pending_close_ticket:
                ticket = pending_close_ticket
                if ticket not in positions: positions[ticket] = {}
                positions[ticket]['close'] = deal_info
                pending_close_ticket = None
            else:
                ticket = deal_id
                if ticket not in positions: positions[ticket] = {}
                positions[ticket]['open'] = deal_info

            continue

    # 构建前端所需的交易数据格式
    trade_data = {
        'points': [],
        'lines': []
    }

    for ticket, pos in positions.items():
        if 'open' in pos:
            p = pos['open']
            trade_data['points'].append({
                'name': 'Buy' if p['side'] == 'buy' else 'Sell',
                'coord': [p['time'], p['price']],
                'itemStyle': {'color': '#3b82f6' if p['side'] == 'buy' else '#f97316'},
                'symbol': 'arrow',
                'symbolSize': 12,
                'symbolRotate': 0 if p['side'] == 'buy' else 180,
                'symbolOffset': [0, 12 if p['side'] == 'buy' else -12]
            })

        if 'close' in pos:
            p = pos['close']
            trade_data['points'].append({
                'name': 'Close',
                'coord': [p['time'], p['price']],
                'itemStyle': {'color': '#a855f7'},
                'symbol': 'pin',
                'symbolSize': 14
            })

        if 'open' in pos and 'close' in pos:
            start = pos['open']
            end = pos['close']
            line_color = '#ef4444'
            if start['side'] == 'buy':
                if end['price'] > start['price']: line_color = '#22c55e'
            else:
                if end['price'] < start['price']: line_color = '#22c55e'

            trade_data['lines'].append([
                {'coord': [start['time'], start['price']],
                 'lineStyle': {'type': 'dashed', 'curveness': 0.2, 'color': line_color}},
                {'coord': [end['time'], end['price']]}
            ])

    return df, trade_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/kline')
def get_kline_data():
    try:
        file_path = get_data_file_path()
        if not file_path:
            return jsonify({'status': 'error', 'message': '未找到数据文件'})

        df, trade_data = parse_ea_log(file_path)

        if df.empty:
            return jsonify({'status': 'error', 'message': '数据解析为空'})

        # 指标计算
        df['MA'] = df['close'].rolling(window=10).mean()

        # 布林带
        period_boll = 20
        df['BOLL_MID'] = df['close'].rolling(window=period_boll).mean()
        df['BOLL_STD'] = df['close'].rolling(window=period_boll).std(ddof=0)
        df['BOLL_UP'] = df['BOLL_MID'] + (df['BOLL_STD'] * 2)
        df['BOLL_LOW'] = df['BOLL_MID'] - (df['BOLL_STD'] * 2)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # [新增] MACD 计算 (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_DIF'] = exp1 - exp2
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])

        def clean_nan(data):
            if isinstance(data, pd.Series):
                return data.astype(object).where(pd.notnull(data), None).tolist()
            return data

        kline_values = df[['open', 'close', 'low', 'high']].values.tolist()

        return jsonify({
            'status': 'success',
            'categoryData': df['date_str'].tolist(),
            'values': kline_values,
            'maData': clean_nan(df['MA']),
            'boll': {
                'up': clean_nan(df['BOLL_UP']),
                'low': clean_nan(df['BOLL_LOW'])
            },
            'rsi': clean_nan(df['RSI']),
            'macd': {  # [新增]
                'dif': clean_nan(df['MACD_DIF']),
                'dea': clean_nan(df['MACD_DEA']),
                'hist': clean_nan(df['MACD_HIST'])
            },
            'trades': trade_data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)