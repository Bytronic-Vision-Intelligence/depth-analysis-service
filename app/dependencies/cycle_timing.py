from datetime import datetime
from time import perf_counter


class CycleTiming():
    '''Times each step of one depth analysis cycle and reports them as a single log line, so a slow
    cycle shows which step lost the time and when, to match against the other services' logs'''
    def __init__(self, message_id):
        self.message_id = message_id
        self.received = datetime.now()
        self.start = self.last = perf_counter()
        self.steps = []

    def mark(self, step:str):
        '''records the time since the previous mark (or since the image arrived) against a step'''
        now = perf_counter()
        self.steps.append((step, now - self.last))
        self.last = now

    def line(self, outcome:str) -> str:
        '''the log line for the cycle
        Args:
            outcome: how the cycle ended, e.g. published or no plate
        '''
        received = self.received.strftime("%H:%M:%S.") + f"{self.received.microsecond // 1000:03d}"
        steps = ", ".join(f"{step} {seconds * 1000:.0f}" for step, seconds in self.steps)
        total = (self.last - self.start) * 1000
        return f"Timing : {self.message_id} received {received}; {steps}; total {total:.0f} ms ({outcome})"


def mark(state_machine, step:str):
    '''marks a step on the state machine's current cycle, if one is being timed'''
    timing = getattr(state_machine, "timing", None)
    if timing is not None:
        timing.mark(step)


def report(state_machine, outcome:str):
    '''prints the current cycle's timing line and ends its timing'''
    timing = getattr(state_machine, "timing", None)
    if timing is None:
        return
    print(timing.line(outcome), flush=True)
    state_machine.timing = None
