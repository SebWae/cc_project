import os
from pathlib import Path


_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = _PROJECT_ROOT = Path(__file__).resolve().parent.parent  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data