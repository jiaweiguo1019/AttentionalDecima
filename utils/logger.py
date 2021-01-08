import threading

import tensorboardX


class Logger(threading.Thread):

    def __init__(self, status_deliver, **kwargs):
        self.status_deliver = status_deliver
        super().__init__(**kwargs)

    def run(self):
        pass




