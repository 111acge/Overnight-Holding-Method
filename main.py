import akshare as ak
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from ta.volatility import BollingerBands


# ------------------------- 数据获取函数 -------------------------
def get_daily_data(symbol):
    """获取单只股票的日线数据"""
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq")
    df = df.rename(columns={
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '股票代码': 'symbol'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_float_share_and_mv(symbol):
    """获取流通股本和市值数据"""
    try:
        # 获取股票指标数据
        df = ak.stock_a_indicator_lg(symbol=symbol)

        # 检查数据是否为空
        if df.empty:
            print(f"股票{symbol}的指标数据为空")
            return None, None

        # 获取当前价格
        price_df = ak.stock_zh_a_spot_em()
        price_df = price_df[price_df['代码'] == symbol]
        current_price = price_df['最新价'].values[0] if not price_df.empty else None

        # 计算总市值
        total_mv = None
        if 'total_mv' in df.columns:
            total_mv = df['total_mv'].iloc[0] * 1e8  # 假设单位是亿元，转为元

        # 估算流通股数
        float_share = None
        if total_mv and current_price and current_price > 0:
            # 使用总市值估算总股数，然后假设70%是流通股
            est_float_share = (total_mv / current_price) * 0.7
            float_share = est_float_share

        return float_share, total_mv

    except Exception as e:
        print(f"获取股票{symbol}数据时出错：{str(e)}")
        return None, None


def get_intraday_data(symbol):
    """获取当日分时数据（1分钟级）"""
    df = ak.stock_zh_a_minute(symbol=symbol, period='1', adjust="")
    df = df.rename(columns={'时间': 'time', '成交': 'price'})
    df['time'] = pd.to_datetime(df['time'])
    df = df[df['time'].dt.time >= pd.to_datetime('14:30:00').time()]  # 仅保留尾盘30分钟
    return df


def get_index_data(index_symbol='000300'):  # 修改默认指数代码
    """获取沪深300指数分时涨幅（示例用日线近似）"""
    try:
        # 尝试不同的指数获取函数
        try:
            df = ak.index_zh_a_hist(symbol=index_symbol, period="daily")
        except:
            # 尝试使用另一个函数作为备选
            df = ak.stock_zh_index_daily(symbol=f"sh{index_symbol}")

        # 计算涨跌幅
        df['pct_change'] = df['收盘'].pct_change() * 100
        return df.iloc[-1]['pct_change']  # 返回最新一天的涨幅
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return 0  # 如果获取失败则返回0，表示与大盘持平


# ------------------------- 筛选逻辑 -------------------------
def filter_stocks(symbol_list):
    """主筛选函数"""
    selected = []
    rejected = {
        "数据不足": 0,
        "无流通市值数据": 0,
        "涨幅不符": 0,
        "量比不足": 0,
        "换手率不符": 0,
        "市值不符": 0,
        "成交量不稳定": 0,
        "布林带压力": 0,
        "尾盘表现不佳": 0,
        "其他错误": 0
    }

    index_pct = get_index_data()  # 获取大盘指数涨幅
    print(f"当前大盘涨幅: {index_pct:.2f}%")
    print(f"开始筛选股票，共{len(symbol_list)}只...")

    processed_count = 0

    for symbol in symbol_list:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count}/{len(symbol_list)} 只股票")

        try:
            # 获取日线数据
            daily_df = get_daily_data(symbol)
            if len(daily_df) < 20:  # 确保有足够数据计算布林带(至少20天)
                rejected["数据不足"] += 1
                continue

            # 获取流通股本和市值
            float_share, total_mv = get_float_share_and_mv(symbol)
            if not float_share or not total_mv:
                rejected["无流通市值数据"] += 1
                continue

            # 当日数据
            today = daily_df.iloc[-1]
            prev_day = daily_df.iloc[-2]

            # 1. 涨幅筛选：3% ≤ 当日涨幅 ≤5%
            pct_change = (today['close'] - prev_day['close']) / prev_day['close'] * 100
            if not (3 <= pct_change <= 5):
                rejected["涨幅不符"] += 1
                continue

            # 2. 量比筛选：当日成交量 / 过去5日平均 ≥1
            avg_volume_5d = daily_df['volume'].iloc[-6:-1].mean()
            volume_ratio = today['volume'] / avg_volume_5d if avg_volume_5d != 0 else 0
            if volume_ratio < 1:
                rejected["量比不足"] += 1
                continue

            # 3. 换手率筛选：直接使用成交额与总市值的比例来估算
            turnover_value = today['volume'] * today['close']  # 成交额
            est_turnover_rate = (turnover_value / total_mv) * 100
            if not (5 <= est_turnover_rate <= 10):
                rejected["换手率不符"] += 1
                continue

            # 4. 流通市值筛选：直接使用总市值的70%作为估计
            est_float_mv = total_mv * 0.7
            if not (50e8 <= est_float_mv <= 200e8):
                rejected["市值不符"] += 1
                continue

            # 5. 成交量稳定性：过去5日变异系数 ≤0.5
            volumes = daily_df['volume'].iloc[-5:].values
            cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) != 0 else 0
            if cv > 0.5:
                rejected["成交量不稳定"] += 1
                continue

            # 6. K线无压：距离布林带上轨 ≥3%
            # 创建布林带指标实例
            bb = BollingerBands(close=daily_df['close'], window=20, window_dev=2)

            # 获取布林带上轨
            upper = bb.bollinger_hband()

            resistance_distance = (upper.iloc[-1] - today['close']) / today['close'] * 100
            if resistance_distance < 3:
                rejected["布林带压力"] += 1
                continue

            # 7. 分时图条件：收盘价 ≥分时均线，且强于大盘
            intraday_df = get_intraday_data(symbol)
            if intraday_df.empty:
                rejected["尾盘表现不佳"] += 1
                continue

            vwap = intraday_df['price'].mean()  # 分时均线
            stock_pct = (intraday_df['price'].iloc[-1] - intraday_df['price'].iloc[0]) / intraday_df['price'].iloc[
                0] * 100

            if today['close'] < vwap or stock_pct < (index_pct + 1):
                rejected["尾盘表现不佳"] += 1
                continue

            # 股票通过所有筛选
            print(f"\n股票{symbol}通过所有筛选条件:")
            print(f"  - 涨幅: {pct_change:.2f}%")
            print(f"  - 量比: {volume_ratio:.2f}")
            print(f"  - 估算换手率: {est_turnover_rate:.2f}%")
            print(f"  - 估算流通市值: {est_float_mv / 1e8:.2f}亿元")
            print(f"  - 成交量变异系数: {cv:.2f}")
            print(f"  - 布林带上轨距离: {resistance_distance:.2f}%")
            print(f"  - 尾盘相对大盘: 大盘{index_pct:.2f}%, 个股尾盘{stock_pct:.2f}%")

            selected.append(symbol)

        except Exception as e:
            rejected["其他错误"] += 1
            print(f"处理{symbol}时出错：{str(e)}")
            continue

    # 打印筛选统计
    print("\n筛选统计:")
    print(f"总处理股票数: {len(symbol_list)}")
    print(f"通过筛选数量: {len(selected)}")
    print("未通过原因统计:")
    for reason, count in rejected.items():
        if count > 0:
            percent = (count / len(symbol_list)) * 100
            print(f"  - {reason}: {count}只 ({percent:.2f}%)")

    return selected


# ------------------------- 执行筛选 -------------------------
if __name__ == "__main__":
    print("开始获取A股代码列表...")
    # 获取当前所有A股代码
    stock_list = ak.stock_zh_a_spot_em()['代码'].tolist()
    print(f"共获取{len(stock_list)}只股票")

    # 可以选择限制测试数量，方便调试
    # stock_list = stock_list[:200]  # 取前200只测试

    print("开始筛选过程...")
    selected_stocks = filter_stocks(stock_list)
    print("\n最终符合条件的股票代码：", selected_stocks)
    print(f"符合条件的股票数量: {len(selected_stocks)}")