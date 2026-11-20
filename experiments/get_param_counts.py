import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tensorflow as tf
from src.models import (
    MLPBaseline,
    Seq2SeqGRU,
    TimeSeriesTransformer,
    TemporalFusionModel
)
from train import calculate_model_size

models = {
    "MLP Baseline": MLPBaseline(),
    "Seq2Seq GRU": Seq2SeqGRU(),
    "TimeSeries Transformer": TimeSeriesTransformer(),
    "Temporal Fusion Model": TemporalFusionModel()
}

dummy_past = tf.zeros((1, 168, 9), dtype=tf.float32)
dummy_cal = tf.zeros((1, 24, 8), dtype=tf.float32)

for name, model in models.items():
    _ = model(dummy_past, dummy_cal, training=False)
    s = calculate_model_size(model)
    print(f"{name}: Total Params = {s['total_params']:,}, Size MB = {s['size_mb']} MB")
