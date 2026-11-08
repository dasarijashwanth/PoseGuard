"""Day 5: scikit-learn posture classifier + FastAPI service.

Train:  python -m experiments.model.train --csv <validation.csv> --save
Serve:  uvicorn experiments.model.api:app --port 8000
"""

from .features import FEATURES, build_windows, window_features

__all__ = ["FEATURES", "build_windows", "window_features"]
