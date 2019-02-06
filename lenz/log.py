import logging
import sys

stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(name)-12s: %(levelname)-10s %(message)s')
stream_handler.setFormatter(formatter)
