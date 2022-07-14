"""
Abstract class, queue property created for child classes
For running methods self.set_queue("command") must be called outside this module after
self.run() method was called as a thread
eder.zermeno@acuitybrands.com
"""
import queue

class tde_runner:

    def __init__(self):
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()

    def set_response(self, response):
        self.response_queue.put(response)
        return

    def get_response(self):
        return self.response_queue.get()

    def clear_response(self):
        self.response_queue.empty()
        return

    def run(self):
        pass

    def set_message(self, item):
        self.message_queue.put(item)
        return

    def get_message(self):
        return self.message_queue.get()
