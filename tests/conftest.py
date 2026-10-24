import os
import sys
from pathlib import Path

os.environ.setdefault("SLM_BACKEND", "none")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import respx

from app import config
from app.data import http as http_mod
from app.data.http import CACHE

http_mod._HOST_MIN_INTERVAL.clear()  # no politeness delays against the fake APIs
from tests import fake_world


@pytest.fixture
def world():
    from app.data import overpass
    CACHE.clear()
    overpass._down_until = 0.0  # reset the Overpass circuit breaker between tests
    with respx.mock(assert_all_called=False) as router:
        fake_world.install(router)
        yield router


@pytest.fixture
def now():
    return fake_world.FAKE_NOW.replace(tzinfo=config.TZ)
