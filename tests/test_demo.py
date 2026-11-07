from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_loads_and_forecasts():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert len(app.metric) == 3
    assert len(app.selectbox) == 2
    assert abs(sum(float(m.value.rstrip("%")) for m in app.metric) - 100) < 0.2
