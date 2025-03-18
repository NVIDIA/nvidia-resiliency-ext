import logging
import os


def pytest_configure():
    logging.basicConfig(
        level=os.getenv('FT_UNIT_TEST_LOGLEVEL', 'DEBUG'),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
