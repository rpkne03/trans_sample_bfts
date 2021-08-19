# -*- coding: utf-8 -*-
"""OHLCデータ作成用プログラム
'http://api.bitcoincharts.com/v1/csv/'から取得した過去のレート(以下、元データ)をOHLCに変換する。
元データは、'時刻(unix_time_stamp)', '価格', '取引量'の3列。
"""

import MySQLdb
import csv
import datetime
import numpy as np

conn = MySQLdb.connect(  # MySQLの接続情報は都度編集
    user='<name>',
    passwd='<password>',
    host='<host>',
    db='<db_name>'
)

cur = conn.cursor()
time_start = datetime.datetime(2017, 7, 4, 17, 0)  # 取得開始日時を指定
time_stop = datetime.datetime(2017, 8, 3, 17, 1)  # 取得終了日時を指定

stamp_start = int(time_start.timestamp())
stamp_stop = int(time_stop.timestamp())
time_interval = 60


def make_ohlc(rate_data, start_timestamp):
    """OHLC作成

    '時刻(unix_time_stamp)', '価格', '取引量'の3列から構成されたレートを、OHLCに変換する

    Args:
        rate_data (numpy.ndarray): '時刻(unix_time_stamp)', '価格', '取引量'の3列から構成されたレート
        start_timestamp (int): 取引開始時刻

    Returns:
        OHLC_data (list): OHLCに変換後([time, open_price, high_price, low_price, close_price, volume] = ([str, (int or float), (int or float), (int or float), (int or float), (int or float)]))

    """
    time_data = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M')  # 取引開始時刻をOHLCの時刻として指定
    OHLC_data = []

    if rate_data.ndim == 2:  # rate_dataを複数存在する(期間中の取引が複数回)
        open_price = rate_data[0][1]
        high_price = rate_data[np.argmax(rate_data[:, 1])][1]
        low_price = rate_data[np.argmin(rate_data[:, 1])][1]
        close_price = rate_data[-1][1]
        volume = sum(rate_data[:, 2])

        OHLC_data.extend([time_data, open_price, high_price, low_price, close_price, volume])

    elif rate_data.ndim == 1:  # rate_dataが1つしか存在しない(期間中取引が1回だけ)

        OHLC_data.extend([time_data, rate_data[1], rate_data[1], rate_data[1], rate_data[1], rate_data[2]])

    return OHLC_data


def mysql_fetch_bytime(mysql_cursor, table_name, start, stop, interval=3600, time_name='timestamp', price_name='price', amount_name='amount'):
    """MySQLからレートデータを一定の時間間隔で抽出

    MySQLのDBから、指定された時刻内のレートを取り出す

    Args:
        mysql_cursor (MySQLdb.cursors.Cursor): 対象データベースに接続したCursorオブジェクト
        table_name (str): 対象テーブル名
        start (int): 取引開始時刻のunixtimestamp
        stop (int): 取引終了時刻のunixtimestamp
        interval (int): 取引の時間間隔(ex. hour=3600, minute=60, day=86400)
        time_name (str): データベース内の時刻の列名
        price_name (str): データベース内の価格の列名
        amount_name (str): データベース内の取引量の列名


    Returns:
        fetch_data (list): チャートデータ(interval毎のデータが要素)

    """
    fetch_data = []

    while stop > start:
        mysql_select = "select from_unixtime(" + time_name + "), " + price_name + ", " + amount_name + " from " + table_name + " where timestamp >= " + str(start) + " and timestamp <= " + str(start + (time_interval - 1))
        mysql_cursor.execute(mysql_select)
        part_fetch = np.array([list(i) for i in mysql_cursor.fetchall()])
        if part_fetch.any():
            part_fetch = make_ohlc(part_fetch, start)
            part_fetch[0] = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')
        else:
            time = datetime.datetime.fromtimestamp(start)
            part_fetch = [datetime.datetime.strftime(time, '%Y-%m-%d %H:%M'), None, None, None, None, None]
        fetch_data.append(part_fetch)

        start += interval

    return fetch_data


fetch_test = mysql_fetch_bytime(cur, '<table_name>', stamp_start, stamp_stop, interval=time_interval)

with open("<filepath>", 'a') as f:
    writer = csv.writer(f)
    for test_row in fetch_test:
        writer.writerow(test_row)

conn.close()
