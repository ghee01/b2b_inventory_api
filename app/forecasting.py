'''
주간 수요 예측 로직

주문 데이터 표를 넣으면 예측 숫자가 나오는 "계산 담당" 파일
'''
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

LAGS = 4    # 최근 몇 주를 보고 다음 주를 예측할지
FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "lag_4", "avg_4", "month_start_days"]
ACTIVE_WINDOW_WEEKS = 52    # "상품별 주문 있던 주 비율"을 계산할 기간 (1년)

def _monday_of(ts: pd.Series) -> pd.Series:
    """각 시각이 속한 주의 월요일 00:00"""
    day = ts.dt.normalize()
    return day - pd.to_timedelta(day.dt.weekday, unit='D')

def weekly_sales(lines: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """
    주문 행(product_id, ordered_at, quantity)을 '주 x 상품' 수량 표로 바꾼다.

    - 주는 월~일 기준, 인덱스는 그 주의 월요일
    - 주문이 없던 주는 0으로 채운다.
    - 데이터가 주 중간에 시작하면 첫 주도 뺀다.
    - 아직 끝나지 않은 이번주는 뺀다.
    """
    if lines.empty:
        return pd.DataFrame()

    df = lines.copy()
    # DB마다 다른 타임존 포맷을 통일
    df['ordered_at'] = pd.to_datetime(df['ordered_at'], utc=True).dt.tz_localize(None)
    df['week_start'] = _monday_of(df['ordered_at'])

    weekly = df.pivot_table(
        index='week_start', columns='product_id', values='quantity', aggfunc='sum', fill_value=0
    )

    first_week = weekly.index.min()
    if df['ordered_at'].min().normalize() != first_week:
        first_week += pd.Timedelta(weeks=1)
    last_week = _monday_of(pd.Series([now])).iloc[0] - pd.Timedelta(weeks=1)
    full_weeks = pd.date_range(first_week, last_week, freq='7D')    

    weekly = weekly.reindex(full_weeks, fill_value=0).astype(float) # 주문 없었던 주 포함
    weekly.index.name = 'week_start'
    weekly.columns.name = None

    return weekly

def active_week_ratio(weekly: pd.DataFrame, window: int = ACTIVE_WINDOW_WEEKS) -> pd.Series:
    """상품별로 최근 window주 중 주문이 있었던 주의 비율 (0~1)"""
    return (weekly.tail(window) > 0).mean()

def month_start_days(week_start: pd.Timestamp) -> int:
    """그 주(월~일)에 '매월 1~7일'이 며칠 들어 있는지 (0~7). 월초 정기 발주 효과용"""
    return sum((week_start + pd.Timedelta(days=i)) <= 7 for i in range(7))

def build_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    학습/평가용 표
    
    lag_1 ~ lag_LAGS: 1~LAGS주 전 수량
    target: 그 주 수량
    """
    lag_cols = [f'lag_{i}' for i in range(1, LAGS+1)]
    tables = []

    for pid in weekly.columns:
        sales = weekly[pid]
        table = pd.DataFrame({
            'product_id':pid,
            'week_start':sales.index,
            'target':sales.values
        })

        for i, col in enumerate(lag_cols, start=1):
            table[col] = sales.shift(i).values

        table[f'avg_{LAGS}'] = table[lag_cols].mean(axis=1)
        table['month_start_days'] = [month_start_days(week) for week in table['week_start']]

        tables.append(table.iloc[LAGS:])    # 앞쪽 4주는 lag가 없어서 버림

        return pd.concat(tables, ignore_index=True)

def wape(actual, pred) -> float:
    """
    WAPE = |오차| 합 / 실제 판매량 합.
    ex) 0.2 → '전체 판매량의 20% 정도 틀림'.
    """
    actual = np.asarray(actual, dtype=float)
    total = actual.sum()
    return float(np.abs(actual - np.asarray(pred, dtype=float)).sum() / total) if total > 0 else float('nan')