# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from typing import List
import pandas as pd


class FinancialMarketIndicators:

    @staticmethod
    def calculate_rsi(ds: List[List[float]],
                      period=14):
        closes = ds[0]
        if len(closes) < period:
            raise ValueError("Input list of closes is too short for the given period.")

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [None] * (period - 1)

        for i in range(period - 1, len(closes)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

            if i < len(closes) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return rsi_values

    @staticmethod
    def calculate_macd(ds: List[List[float]], short_period=12, long_period=26, signal_period=9):
        ds = ds[0]

        short_ema = []
        long_ema = []
        macd_line = []
        signal_line = []

        for i in range(len(ds)):
            if i < long_period - 1:
                short_ema.append(None)
                long_ema.append(None)
                macd_line.append(None)
                signal_line.append(None)
            else:
                short_ema.append(sum(ds[i - short_period + 1:i + 1]) / short_period)
                long_ema.append(sum(ds[i - long_period + 1:i + 1]) / long_period)
                macd_line.append(short_ema[i] - long_ema[i])

                if i >= long_period + signal_period - 1:
                    signal_line.append(sum(macd_line[i - signal_period + 1:i + 1]) / signal_period)
                else:
                    signal_line.append(None)

        return macd_line, signal_line

    @staticmethod
    def calculate_bollinger_bands(ds: List[List[float]],
                                  window=20,
                                  num_std_dev=2):
        ds = ds[0]

        middle_band = []
        upper_band = []
        lower_band = []

        for i in range(len(ds)):
            if i < window - 1:
                middle_band.append(None)
                upper_band.append(None)
                lower_band.append(None)
            else:
                window_data = ds[i - window + 1:i + 1]
                middle = sum(window_data) / window
                std_dev = (sum((x - middle) ** 2 for x in window_data) / window) ** 0.5

                middle_band.append(middle)
                upper_band.append(middle + num_std_dev * std_dev)
                lower_band.append(middle - num_std_dev * std_dev)

        return middle_band, upper_band, lower_band

    @staticmethod
    def calculate_ema(ds: List[List[float]], length = 9):
        closes = ds[0]

        alpha = 2 / (length + 1)

        ema = []

        for a in range(0, length):
            ema.append(None)

        ema.append(sum(closes[:length]) / length)

        for i in range(length+1, len(closes)):
            ema_value = alpha * closes[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)

        return ema
