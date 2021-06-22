from bisect import bisect_right
from typing import Iterator, List, Sequence, Tuple

import torch

from lib.data.structures import WsodElement


__all__ = ['GroupedBatchSampler', 'compute_aspect_ratio_grouping']


class GroupedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        groups: Sequence[int],
        batch_size: int = 1,
        drop_last: bool = False,
        hard_grouping: bool = False,
    ):
        assert not (hard_grouping and not drop_last), "drop_last must be enabled for hard_grouping"
        self.groups = groups
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last
        self.hard_grouping = hard_grouping
        self.buckets: List[List[int]] = [] * len(set(groups))

    def __iter__(
        self,
    ) -> Iterator[List[int]]:
        for idx in self.sampler:
            group = self.groups[idx]
            self.buckets[group].append(idx)
            if len(self.buckets[group]) == self.batch_size:
                yield self.buckets[group]
                self.buckets[group] = []

        leftovers: List[int] = []
        if not self.hard_grouping:  # yield suboptimally grouped batches at the end
            for group in range(len(self.buckets)):
                leftovers = leftovers + self.buckets[group]
                self.buckets[group] = []
                if len(leftovers) >= self.batch_size:
                    yield leftovers[:self.batch_size]
                    leftovers = leftovers[self.batch_size:]

        if not self.drop_last:  # yield any under-size batches
            for group in range(len(self.buckets)):
                if len(self.buckets[group]) > 0:
                    yield self.buckets[group]
                    self.buckets[group] = []
            if len(leftovers) > 0:
                yield leftovers

    def __len__(
        self,
    ) -> int:
        if self.drop_last:
            if self.hard_grouping:  # might be leftovers in each group
                _, counts = torch.tensor(self.groups).unique(return_counts=True)
                return sum([c // self.batch_size for c in counts])
            else:
                return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def compute_aspect_ratio_grouping(
    dataset: torch.utils.data.IterableDataset,
    aspect_ratio_bounds: Tuple[float, ...] = (1.,)
) -> List[int]:
    return [bisect_right(aspect_ratio_bounds, element.image_size[0] / element.image_size[1])
            for element, _ in dataset]
