"""
KG DataFrame / 向量載入
"""
import numpy as np
import pandas as pd

from ..core.paths import KG_EMB_PATH, KG_CSV_PATH

KG_VECS = np.load(KG_EMB_PATH)
KG_VECS_NORM = KG_VECS / np.linalg.norm(KG_VECS, axis=1, keepdims=True)
KG_DF = pd.read_csv(KG_CSV_PATH)
HP_COL = next((c for c in ['head_props'] if c in KG_DF.columns), None)
RP_COL = next((c for c in ['rel_props'] if c in KG_DF.columns), None)
TP_COL = next((c for c in ['tail_props'] if c in KG_DF.columns), None)
