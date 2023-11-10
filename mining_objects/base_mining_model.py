# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC
import math
import numpy as np
import tensorflow
from numpy import ndarray

from mining_objects.financial_market_indicators import FinancialMarketIndicators
from vali_objects.scaling.scaling import Scaling


class BaseMiningModel:
    def __init__(self, features):
        self.neurons = [50]
        self.features = features
        self.loaded_model = None
        self.window_size = 100
        self.model_dir = None
        self.batch_size = 16
        self.learning_rate = 0.01

    def set_neurons(self, neurons):
        self.neurons = neurons
        return self

    def set_window_size(self, window_size):
        self.window_size = window_size
        return self

    def set_model_dir(self, model, stream_id=None):
        if model is None and stream_id is not None:
            self.model_dir = f'mining_models/{stream_id}.keras'
        elif model is not None:
            self.model_dir = model
        else:
            raise Exception("stream_id is not provided to define model")
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def load_model(self):
        self.loaded_model = tensorflow.keras.models.load_model(self.model_dir)
        return self

    def train(self, data: ndarray, epochs: int = 100):
        try:
            model = tensorflow.keras.models.load_model(self.model_dir)
        except OSError:
            model = None

        output_sequence_length = 100

        if model is None:
            model = tensorflow.keras.models.Sequential()
            if len(self.neurons) == 1:
                model.add(tensorflow.keras.layers.LSTM(self.neurons[0],
                                                       input_shape=(self.window_size, self.features-1)))
            if len(self.neurons) > 1:
                model.add(tensorflow.keras.layers.LSTM(self.neurons[0],
                                                       input_shape=(self.window_size, self.features),
                                                       return_sequences=True))
                model.add(tensorflow.keras.layers.Dropout(0.2))
                model.add(tensorflow.keras.layers.LSTM(self.neurons[1]))
            model.add(tensorflow.keras.layers.Dense(output_sequence_length))

            optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

        X_train, Y_train = [], []

        X_train_data = data.T[1:].T
        Y_train_data = data.T[0].T

        for i in range(len(Y_train_data) - output_sequence_length):
            # target_sequence = Y_train_data[i+self.window_size]
            target_sequence = Y_train_data[i:i+output_sequence_length]
            Y_train.append(target_sequence)

        for i in range(len(X_train_data) - output_sequence_length):
            input_sequence = X_train_data[i:i+self.window_size]
            X_train.append(input_sequence)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_train = tensorflow.convert_to_tensor(np.array(X_train, dtype=np.float32))
        Y_train = tensorflow.convert_to_tensor(np.array(Y_train, dtype=np.float32))

        # Y_train = tensorflow.reshape(Y_train, len(Y_train), 1)

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="loss", patience=10,

                                                                  restore_best_weights=True)

        model.fit(X_train, Y_train, epochs=epochs, batch_size=self.batch_size, callbacks=[early_stopping])
        model.save(self.model_dir)

    def predict(self, data: ndarray):
        predictions = []

        window_data = data[-self.window_size:]
        window_data = window_data.reshape(1, self.window_size, self.features)

        predicted_value = self.loaded_model.predict(window_data)
        predictions.append(predicted_value)

        return predictions

    def predict_close_only(self, data: ndarray):
        data = data.T[1:].T
        predictions = []

        window_data = data[-self.window_size:]
        window_data = window_data.reshape(1, self.window_size, self.features)

        predicted_value = self.loaded_model.predict(window_data)
        predictions.append(predicted_value)

        return predictions

    @staticmethod
    def base_model_dataset(samples):
        min_cutoff = 0

        cutoff_close = samples.tolist()[1][min_cutoff:]
        cutoff_high = samples.tolist()[2][min_cutoff:]
        cutoff_low = samples.tolist()[3][min_cutoff:]
        cutoff_volume = samples.tolist()[4][min_cutoff:]

        return np.array([cutoff_close,
                                 cutoff_high,
                                 cutoff_low,
                                 cutoff_volume]).T

    @staticmethod
    def base_model_additional_dataset(samples):
        samples = samples[1:]

        rsi = FinancialMarketIndicators.calculate_rsi(samples.tolist())

        macd, signal = FinancialMarketIndicators.calculate_macd(samples.tolist(), 12, 26, 30)
        macd2, signal2 = FinancialMarketIndicators.calculate_macd(samples.tolist(), 24, 52, 30)
        macd3, signal3 = FinancialMarketIndicators.calculate_macd(samples.tolist(), 48, 104, 30)
        macd4, signal4 = FinancialMarketIndicators.calculate_macd(samples.tolist(), 86, 208, 30)
        macd5, signal5 = FinancialMarketIndicators.calculate_macd(samples.tolist(), 172, 416, 30)
        macd6, signal6 = FinancialMarketIndicators.calculate_macd(samples.tolist(), 244, 824, 30)

        middle, upper, lower = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist())
        middle2, upper2, lower2 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=64)
        middle3, upper3, lower3 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=128)
        middle4, upper4, lower4 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=256)
        middle5, upper5, lower5 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=512)
        middle6, upper6, lower6 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=1024)

        min_cutoff = 0

        for i, val in enumerate(rsi):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        for i, val in enumerate(middle):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        for i, val in enumerate(middle6):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        for i, val in enumerate(signal6):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        min_cutoff += 1

        vmin_signal, vmax_signal, cutoff_signal = Scaling.scale_values(signal[min_cutoff:])
        vmin_signal, vmax_signal, cutoff_signal2 = Scaling.scale_values(signal2[min_cutoff:])
        vmin_signal, vmax_signal, cutoff_signal3 = Scaling.scale_values(signal3[min_cutoff:])
        vmin_signal, vmax_signal, cutoff_signal4 = Scaling.scale_values(signal4[min_cutoff:])
        vmin_signal, vmax_signal, cutoff_signal5 = Scaling.scale_values(signal5[min_cutoff:])
        vmin_signal, vmax_signal, cutoff_signal6 = Scaling.scale_values(signal6[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle = Scaling.scale_values(middle[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower = Scaling.scale_values(lower[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper = Scaling.scale_values(upper[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle2 = Scaling.scale_values(middle2[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower2 = Scaling.scale_values(lower2[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper2 = Scaling.scale_values(upper2[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle3 = Scaling.scale_values(middle3[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower3 = Scaling.scale_values(lower3[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper3 = Scaling.scale_values(upper3[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle4 = Scaling.scale_values(middle4[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower4 = Scaling.scale_values(lower4[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper4 = Scaling.scale_values(upper4[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle5 = Scaling.scale_values(middle5[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower5 = Scaling.scale_values(lower5[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper5 = Scaling.scale_values(upper5[min_cutoff:])

        vmin_bb, vmax_bb, cutoff_middle6 = Scaling.scale_values(middle6[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_lower6 = Scaling.scale_values(lower6[min_cutoff:])
        vmin_bb, vmax_bb, cutoff_upper6 = Scaling.scale_values(upper6[min_cutoff:])

        vmin_rsi, vmax_rsi, cutoff_rsi = Scaling.scale_values(rsi[min_cutoff:])

        cutoff_close = samples.tolist()[0][min_cutoff:]
        cutoff_high = samples.tolist()[1][min_cutoff:]
        cutoff_low = samples.tolist()[2][min_cutoff:]
        cutoff_volume = samples.tolist()[3][min_cutoff:]

        return np.array([cutoff_high,
                         cutoff_low,
                         cutoff_volume,
                         cutoff_signal,
                         cutoff_signal2,
                         cutoff_signal3,
                         cutoff_signal4,
                         cutoff_signal5,
                         cutoff_signal6,
                         cutoff_rsi,
                         cutoff_middle,
                         cutoff_lower,
                         cutoff_upper,
                         cutoff_middle2,
                         cutoff_lower2,
                         cutoff_upper2,
                         cutoff_middle3,
                         cutoff_lower3,
                         cutoff_upper3,
                         cutoff_middle4,
                         cutoff_lower4,
                         cutoff_upper4,
                         cutoff_middle5,
                         cutoff_lower5,
                         cutoff_upper5,
                         cutoff_middle6,
                         cutoff_lower6,
                         cutoff_upper6,
                         ]).T