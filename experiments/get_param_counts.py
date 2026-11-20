import os, sys
sys.path.insert(0, os.path.abspath("."))
import torch
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

for name, model in models.items():
    s = calculate_model_size(model)
    print(f"{name}: Total Params = {s['total_params']:,}, Size MB = {s['size_mb']} MB")
