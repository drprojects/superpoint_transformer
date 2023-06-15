import logging
from src.data import NAG
from src.transforms import Transform


log = logging.getLogger(__name__)


__all__ = ['HelloWorld']


class HelloWorld(Transform):
    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        log.info("\n**** Hello World ! ****\n")
        return nag
