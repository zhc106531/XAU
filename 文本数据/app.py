from flask import Flask, render_template, jsonify, request
import pandas as pd
import re
from datetime import datetime
import os
import traceback
import numpy as np

app = Flask(__name__)

# --- 配置区域 ---
CUSTOM_LOG_PATH = r"C:\Users\Administrator\PyCharmMiscProject\自制K线\文本数据\exported_data.txt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. 交易报表 (前端显示用)
BACKTEST_REPORT_FILE = os.path.join(BASE_DIR, 'backtest_detail.csv')
# 2. 详细过程日志 (调试分析用，格式对齐 MT5)
PROCESS_LOG_FILE = os.path.join(BASE_DIR, 'backtest_process.log')


def get_data_file_path():
    if CUSTOM_LOG_PATH and os.path.exists(CUSTOM_LOG_PATH):
        return CUSTOM_LOG_PATH
    local_data = os.path.join(BASE_DIR, 'data.txt')
    if os.path.exists(local_data):
        return local_data
    return CUSTOM_LOG_PATH if CUSTOM_LOG_PATH else local_data


def parse_data(file_path):
    data_list = []
    pattern = re.compile(r'Core\s+\d+.*?(\d{4}\.\d{2}\.\d{2}\s\d{2}:\d{2}:\d{2}).*?ask:\s*([\d.]+).*?bid:\s*([\d.]+)',
                         re.IGNORECASE)

    print(f"正在尝试读取文件: {file_path}")
    if not os.path.exists(file_path):
        print("错误: 文件不存在！")
        return pd.DataFrame()

    encodings_to_try = ['utf-16', 'utf-8', 'gbk', 'latin-1']
    lines = []
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

    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            try:
                quote_time_str = match.group(1)
                ask = float(match.group(2))
                bid = float(match.group(3))
                mid_price = (ask + bid) / 2
                data_list.append({
                    'timestamp': datetime.strptime(quote_time_str, '%Y.%m.%d %H:%M:%S'),
                    'price': mid_price
                })
            except ValueError:
                continue

    if not data_list:
        print("警告: 未匹配到数据。")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    print(f"成功解析 {len(df)} 条 Tick 数据")
    return df


def process_market_data(period, ma_period, ema_period, boll_period, boll_std):
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
        ohlc['MA_SLOPE'] = ohlc['MA'].diff()

        ohlc['EMA'] = ohlc['close'].ewm(span=int(ema_period), adjust=False).mean()
        ohlc['EMA_SLOPE'] = ohlc['EMA'].diff()

        period_boll = int(boll_period)
        std_dev = float(boll_std)
        ohlc['BOLL_MID'] = ohlc['close'].rolling(window=period_boll).mean()

        # [核心修改] std(ddof=0) 使用总体标准差，与 MT5 的 MathSqrt(sum/N) 对齐
        ohlc['BOLL_STD'] = ohlc['close'].rolling(window=period_boll).std(ddof=0)

        ohlc['BOLL_UP'] = ohlc['BOLL_MID'] + (ohlc['BOLL_STD'] * std_dev)
        ohlc['BOLL_LOW'] = ohlc['BOLL_MID'] - (ohlc['BOLL_STD'] * std_dev)

        ohlc['BOLL_WIDTH'] = ohlc['BOLL_UP'] - ohlc['BOLL_LOW']
        ohlc['BOLL_WIDTH_DELTA'] = ohlc['BOLL_WIDTH'].diff()

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
        return ohlc
    except Exception as e:
        traceback.print_exc()
        return None


# --- 回测引擎 ---
class BacktestEngine:
    def __init__(self, df, ma_period, ema_period, boll_period, boll_std):
        self.df = df
        self.ma_period = ma_period
        self.ema_period = ema_period
        self.boll_period = boll_period
        self.boll_std = boll_std
        self.trades = []
        self.position = 0
        self.entry_price = 0.0
        self.entry_index = 0
        self.entry_time = None
        self.trade_type = ""
        self.half_spread = 0.1

    def log_init(self):
        """初始化日志文件"""
        try:
            with open(BACKTEST_REPORT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
                f.write("开仓时间,平仓时间,方向,策略类型,开仓价,平仓价,盈亏(PnL),持仓K线数,平仓原因\n")
        except Exception as e:
            print(f"CSV初始化失败: {e}")

        try:
            with open(PROCESS_LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"--- Web Backtest Process Log (Version 2.05 - Full Bars) Start at {datetime.now()} ---\n")
                f.write(f"InpMAPeriod={self.ma_period}\n")
                f.write(f"InpEmaPeriod={self.ema_period}\n")
                f.write(f"InpBollPeriod={self.boll_period}\n")
                f.write(f"InpBollStd={self.boll_std}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"Log初始化失败: {e}")

    def log_trade_csv(self, trade):
        try:
            with open(BACKTEST_REPORT_FILE, 'a', encoding='utf-8-sig', newline='') as f:
                direction_cn = "做多" if trade['type'] == 'Long' else "做空"
                line = (
                    f"{trade['entry_time']},"
                    f"{trade['exit_time']},"
                    f"{direction_cn},"
                    f"{trade['strategy']},"
                    f"{trade['entry_price']},"
                    f"{trade['exit_price']},"
                    f"{trade['pnl']},"
                    f"{trade['bars']},"
                    f"{trade['reason']}\n"
                )
                f.write(line)
        except Exception as e:
            print(f"写入交易记录失败: {e}")

    def log_process(self, msg):
        try:
            with open(PROCESS_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        except Exception as e:
            pass

    def run(self):
        self.log_init()

        # [核心修改] 循环从 0 开始，先打印日志，再判断是否足够数据
        for i in range(len(self.df)):
            curr = self.df.iloc[i]

            # 1. 无论 i 是多少，只要有 K 线就先打印 (对齐 MT5 行为)
            time_str = curr.name.strftime('%Y.%m.%d %H:%M:%S')
            close = curr['close']
            low = curr['low']
            high = curr['high']
            open_ = curr['open']

            log_line = f"BAR CLOSED [{time_str}]: O={open_:.2f} H={high:.2f} L={low:.2f} C={close:.2f}"
            self.log_process(log_line)

            # 2. 如果历史数据不足以计算 Prev 指标，跳过策略逻辑
            # 我们需要 i-1 和 i-2，所以至少 i >= 2
            if i < 2:
                continue

            # 3. 策略逻辑 (Safe to access prev/prev2)
            prev = self.df.iloc[i - 1]
            prev2 = self.df.iloc[i - 2]

            ma = prev['MA']
            upper = prev['BOLL_UP']
            lower = prev['BOLL_LOW']
            width = prev['BOLL_WIDTH']
            width_delta = prev['BOLL_WIDTH_DELTA']
            ma_slope = prev['MA_SLOPE']

            if pd.isna(ma) or pd.isna(upper) or pd.isna(width_delta) or pd.isna(ma_slope):
                continue

            is_trend = width_delta > 0
            is_oscillation = not is_trend
            is_volatile_enough = width > 2.0

            # --- 持仓处理 ---
            if self.position != 0:
                bars_held = i - self.entry_index
                close_signal = False
                close_reason = ""

                if bars_held > 10:
                    if (self.position == 1 and low <= ma) or (self.position == -1 and high >= ma):
                        close_signal = True
                        close_reason = "Time Stop (MA Return)"

                if not close_signal:
                    if self.trade_type == 'oscillation':
                        if (self.position == 1 and high >= ma) or (self.position == -1 and low <= ma):
                            close_signal = True
                            close_reason = "Oscillation Profit"
                    elif self.trade_type == 'trend':
                        if (self.position == 1 and high >= upper) or (self.position == -1 and low <= lower):
                            close_signal = True
                            close_reason = "Trend Target"

                if close_signal:
                    if self.position == 1:
                        exec_price_mid = close
                        exec_price_real = exec_price_mid - self.half_spread
                    else:
                        exec_price_mid = close
                        exec_price_real = exec_price_mid + self.half_spread

                    self.close_position(i, exec_price_real, close_reason, bars_held)
                    continue

                    # --- 开仓处理 ---
            if self.position == 0 and is_volatile_enough:
                if is_oscillation:
                    if low < lower:
                        self.log_process(
                            f">> SIGNAL [Oscillation BUY]: MidLow={low:.2f} < PrevBollLow={lower:.2f} | PrevWidth={width:.2f}")
                        self.open_position(i, lower + self.half_spread, 1, 'oscillation', "Oscillation Buy")
                    elif high > upper:
                        self.log_process(
                            f">> SIGNAL [Oscillation SELL]: MidHigh={high:.2f} > PrevBollUp={upper:.2f} | PrevWidth={width:.2f}")
                        self.open_position(i, upper - self.half_spread, -1, 'oscillation', "Oscillation Sell")
                elif is_trend:
                    if ma_slope > 0 and low <= ma:
                        self.log_process(
                            f">> SIGNAL [Trend BUY]: PrevSlope={ma_slope:.4f} | MidLow={low:.2f} <= PrevMA={ma:.2f}")
                        self.open_position(i, ma + self.half_spread, 1, 'trend', "Trend Buy")
                    elif ma_slope < 0 and high >= ma:
                        self.log_process(
                            f">> SIGNAL [Trend SELL]: PrevSlope={ma_slope:.4f} | MidHigh={high:.2f} >= PrevMA={ma:.2f}")
                        self.open_position(i, ma - self.half_spread, -1, 'trend', "Trend Sell")

        return self.generate_report()

    def open_position(self, index, price, direction, t_type, comment):
        self.position = direction
        self.entry_price = price
        self.entry_index = index
        self.entry_time = self.df.index[index]
        self.trade_type = t_type

        type_str = "BUY" if direction == 1 else "SELL"
        self.log_process(f"++ ORDER OPENED: {type_str} at {price:.2f} | Comment: {comment}")

    def close_position(self, index, price, reason, bars_held):
        pnl = (price - self.entry_price) * self.position
        self.log_process(f"<< CLOSE SIGNAL: Ticket={index} | Reason={reason} | BarsHeld={bars_held} | PnL={pnl:.2f}")

        trade_record = {
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': self.df.index[index].strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Long' if self.position == 1 else 'Short',
            'strategy': self.trade_type,
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(price, 2),
            'pnl': round(pnl, 2),
            'bars': bars_held,
            'reason': reason
        }

        self.trades.append(trade_record)
        self.log_trade_csv(trade_record)
        self.position = 0

    def generate_report(self):
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'trades': []}

        df_trades = pd.DataFrame(self.trades)
        total_pnl = df_trades['pnl'].sum()
        wins = df_trades[df_trades['pnl'] > 0]
        win_rate = (len(wins) / len(df_trades)) * 100

        return {
            'total_trades': len(df_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'trades': self.trades[::-1]
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/kline')
def get_kline_data():
    try:
        period = request.args.get('period', '1min')
        ma_val = request.args.get('ma', 10)
        ema_val = request.args.get('ema', 20)
        boll_period = request.args.get('bollP', 20)
        boll_std = request.args.get('bollS', 2)

        df = process_market_data(period, ma_val, ema_val, boll_period, boll_std)

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '无数据'})

        def clean_nan(data):
            if isinstance(data, pd.Series):
                return data.astype(object).where(pd.notnull(data), None).tolist()
            return data

        kline_df = df[['open', 'close', 'low', 'high', 'volume']].copy()
        kline_df = kline_df.astype(object).where(pd.notnull(kline_df), None)
        kline_values = kline_df.values.tolist()

        return jsonify({
            'status': 'success',
            'categoryData': df['date_str'].tolist(),
            'values': kline_values,
            'maData': clean_nan(df['MA']),
            'maSlope': clean_nan(df['MA_SLOPE']),
            'emaData': clean_nan(df['EMA']),
            'emaSlope': clean_nan(df['EMA_SLOPE']),
            'boll': {
                'up': clean_nan(df['BOLL_UP']),
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


@app.route('/api/run_backtest')
def run_backtest_api():
    try:
        period = request.args.get('period', '1min')
        ma_val = request.args.get('ma', 10)
        ema_val = request.args.get('ema', 20)
        boll_period = request.args.get('bollP', 20)
        boll_std = request.args.get('bollS', 2)

        df = process_market_data(period, ma_val, ema_val, boll_period, boll_std)
        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '无法加载数据进行回测'})

        # [Changed] Pass parameters to engine
        engine = BacktestEngine(df, ma_val, ema_val, boll_period, boll_std)
        report = engine.run()

        return jsonify({'status': 'success', 'report': report})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)