from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TimeSequence:
    start: datetime
    length: int = 1
    step: timedelta | None = None

    def __post_init__(self):
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if self.step is not None and self.step.total_seconds() <= 0:
            raise ValueError("step must be positive")

    def __iter__(self):
        for i in range(self.length):
            yield self.start + (self.step * i if self.step else timedelta(0))

    def __len__(self):
        return self.length


class TimeList:
    def __init__(self, times: list[datetime]):
        self.sequences: list[TimeSequence] = []
        sorted_times = sorted(times)
        index = 0

        while index < len(sorted_times):
            if index + 2 < len(sorted_times):
                step = sorted_times[index + 1] - sorted_times[index]
                next_step = sorted_times[index + 2] - sorted_times[index + 1]

                if step == next_step and step > timedelta(0):
                    length = 3
                    while (
                        index + length < len(sorted_times)
                        and sorted_times[index + length]
                        - sorted_times[index + length - 1]
                        == step
                    ):
                        length += 1

                    self.sequences.append(
                        TimeSequence(sorted_times[index], length, step)
                    )
                    index += length
                    continue

            self.sequences.append(TimeSequence(sorted_times[index]))
            index += 1

    def __iter__(self):
        for sequence in self.sequences:
            yield from sequence

    def __len__(self):
        return sum(len(sequence) for sequence in self.sequences)
