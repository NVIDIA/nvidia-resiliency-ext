# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import bisect
import collections
import dataclasses
import datetime
import enum
import heapq
import itertools
import warnings
from typing import Callable, Optional, Union

from . import exception
from .state import Mode, State
from .store import StoreMixin


class RankDiscarded(exception.RestartError):
    r'''
    Exception raised when a distributed rank is discarded by
    :py:class:`RankAssignment`.
    '''

    pass


@dataclasses.dataclass
class RankAssignmentCtx:
    r'''
    Represents inputs and outputs of :py:class:`RankAssignment`.

    Args:
        state: :py:class:`Wrapper` state
        store: distributed store
        terminated_ranks: a set containing indices of terminated ranks
    '''

    state: State
    store: StoreMixin
    terminated_ranks: set[int]


class RankAssignment(abc.ABC):
    r'''
    Abstract base class for ``rank_assignment`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`RankAssignment` is responsible for reassigning distributed
    ranks, computing the new world size and selecting which ranks are active in
    the next iteration of the wrapped function.

    Active ranks call the provided wrapped function. Inactive ranks are waiting
    idle, and could serve as a pool of static, preallocated and preinitialized
    reserve ranks. Reserve ranks would be activated in a subsequent restart
    iteration if previously active ranks were terminated or became unhealthy.

    Multiple instances of composable :py:class:`RankAssignment` could be
    composed with :py:class:`inprocess.Compose` to achieve the desired
    behavior.
    '''

    @abc.abstractmethod
    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        r'''
        Implementation of a :py:class:`RankAssignment`.

        Args:
            ctx: :py:class:`RankAssignmentCtx`

        Returns:
            Modified :py:class:`RankAssignmentCtx`
        '''
        raise NotImplementedError


class RankFilter(RankAssignment):
    r'''
    :py:class:`RankFilter` is a subclass of :py:class:`RankAssignment` which
    selects which ranks are active in the current restart iteration of
    :py:class:`inprocess.Wrapper`.

    Active ranks call the wrapped function. Inactive ranks are waiting idle,
    and could serve as a pool of static, preallocated and preinitialized
    reserve ranks. Reserve ranks would be activated in a subsequent restart
    iteration if one of the active ranks is terminated or becomes unhealthy.

    Multiple :py:class:`RankFilter` or :py:class:`RankAssignment` instances can
    be composed using :py:class:`inprocess.Compose` to achieve the desired
    behavior. Typically, all :py:class:`RankFilter` instances should follow any
    :py:class:`RankAssignment` steps that recalculate rank indices or adjust
    the world size.
    '''

    @abc.abstractmethod
    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        r'''
        Implementation of a :py:class:`RankFilter`.

        Args:
            ctx: :py:class:`RankAssignmentCtx`

        Returns:
            Modified :py:class:`RankAssignmentCtx`
        '''
        raise NotImplementedError


class ActivateAllRanks(RankFilter):
    r'''
    Activates all distributed ranks.

    All healthy distributed ranks will call the provided wrapped function in
    the next iteration of :py:class:`inprocess.Wrapper`.

    :py:class:`ActivateAllRanks` unconditionally activates all ranks, and
    cannot be combined with any other :py:class:`RankAssignment` performing
    rank activation.
    '''

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = dataclasses.replace(
            ctx.state,
            mode=Mode.ACTIVE,
            active_rank=ctx.state.rank,
            active_world_size=ctx.state.world_size,
        )
        ctx.state = state
        return ctx


class MaxActiveWorldSize(RankFilter):
    r'''
    :py:class:`MaxActiveWorldSize` ensures that the active world size is no
    greater than the specified ``max_active_world_size``. Ranks with indices
    less than the active world size are active and calling the wrapped
    function, while ranks outside this range are inactive.

    Args:
        max_active_world_size: maximum active world size, no limit if
            :py:obj:`None`
    '''

    def __init__(self, max_active_world_size: Optional[int] = None):
        self.max_active_world_size = max_active_world_size

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = ctx.state

        if state.active_world_size is None:
            active_world_size = state.world_size
        else:
            active_world_size = min(state.active_world_size, state.world_size)

        if self.max_active_world_size is not None:
            active_world_size = min(active_world_size, self.max_active_world_size)
        if state.rank < active_world_size:
            mode = Mode.ACTIVE
            active_rank = state.rank
        else:
            mode = Mode.INACTIVE
            active_rank = None

        state = dataclasses.replace(
            state,
            mode=mode,
            active_rank=active_rank,
            active_world_size=active_world_size,
        )
        ctx.state = state
        return ctx


class ActiveWorldSizeDivisibleBy(RankFilter):
    r'''
    :py:class:`ActiveWorldSizeDivisibleBy` ensures that the active world size
    is divisible by a given number. Ranks within the adjusted world size are
    marked as active and are calling the wrapped function, while ranks outside
    this range are marked as inactive.

    Args:
        divisor: the divisor to adjust the active world size by
    '''

    def __init__(self, divisor: int = 1) -> None:
        self.divisor = divisor

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = ctx.state
        divisor = self.divisor

        if state.active_world_size is None:
            active_world_size = state.world_size
        else:
            active_world_size = min(state.active_world_size, state.world_size)

        active_world_size = active_world_size // divisor * divisor
        if state.rank < active_world_size:
            mode = Mode.ACTIVE
            active_rank = state.rank
        else:
            mode = Mode.INACTIVE
            active_rank = None

        state = dataclasses.replace(
            state,
            mode=mode,
            active_rank=active_rank,
            active_world_size=active_world_size,
        )
        ctx.state = state
        return ctx


class LayerFlag(enum.Flag):
    r'''
    A flag to modify rank assignment or rank filtering policy in a given
    :py:class:`Layer` of a :py:class:`Tree` rank assignment.

    Attributes:
        RESERVE: indicates that branches at this layer of the topology tree may
            be traversed while searching for a replacement inactive rank
        BACKFILL: indicates that branches at this layer of the topology tree
            may be traversed while searching for a replacement active rank
    '''

    RESERVE = enum.auto()
    BACKFILL = enum.auto()


@dataclasses.dataclass
class Layer:
    r'''
    Represents a configuration for a layer of branches at a certain depth in
    a topology tree constructed by :py:class:`Tree`.

    Args:
        min_ranks: the minimum number of healthy ranks in a subtree
        max_ranks: the maximum number of ranks to activate in a subtree, no
            limit if :py:obj:`None`
        key_or_fn: a string key, or a ``Callable`` evaluated with
            :py:class:`inprocess.State` as input to produce a grouping string
            key
        flag: an optional flag that modifies rank assignment policy in a given
            branch
    '''

    min_ranks: int = 1
    max_ranks: Optional[int] = None
    key_or_fn: Union[str, Callable[[State], str]] = ''
    flag: Optional[LayerFlag] = None


class Node:
    def __init__(self, parent, name, layer, state):
        self.parent = parent
        self.name = name
        self.layer = layer
        self.state = state

        self.active_count = 0
        self.children = {}
        self.inactive_nodes = {}
        self.backfill_domain = None

    def add_child(self, name, layer, state):
        child = Node(self, name, layer, state)
        self.children[name] = child
        return child

    def is_leaf(self):
        return not self.children

    def iter_leaves(self):
        if self.is_leaf():
            yield self
        else:
            for child in self.children.values():
                yield from child.iter_leaves()

    def deactivate(self):
        self.state.mode = Mode.INACTIVE
        self.state.active_rank = None

        node = self
        while (parent := node.parent) is not None:
            parent.inactive_nodes[self.state.initial_rank] = self
            node = parent

    def terminate(self):
        self.state.mode = Mode.TERMINATED

        node = self
        while (parent := node.parent) is not None:
            if self.state.initial_rank in parent.inactive_nodes:
                parent.inactive_nodes.pop(self.state.initial_rank)
            node = parent

    def activate(self, active_rank):
        self.state.active_rank = active_rank
        self.state.mode = Mode.ACTIVE

        node = self
        while (parent := node.parent) is not None:
            if self.state.initial_rank in parent.inactive_nodes:
                parent.inactive_nodes.pop(self.state.initial_rank)
            node = parent

    def assign_backfill_domain(self):
        assert self.is_leaf()

        backfill_domain = None
        parent = self.parent

        while parent and parent.layer.flag and parent.layer.flag & LayerFlag.BACKFILL:
            backfill_domain = parent
            parent = parent.parent

        self.backfill_domain = backfill_domain

    def __repr__(self):
        return f'{type(self).__name__}({self.name=})'


def bounded_activate(node, counter, path=None):
    if path is None:
        path = []

    if node.is_leaf():
        if all(
            (
                ascendant.layer.max_ranks is None
                or ascendant.active_count < ascendant.layer.max_ranks
                for ascendant in path
            )
        ):
            node.activate(counter)
            counter += 1
            for ascendant in path:
                ascendant.active_count += 1
        else:
            node.deactivate()

    path.append(node)

    for child in node.children.values():
        counter = bounded_activate(child, counter, path)
    path.pop()
    return counter


def propagate_terminations(node, terminated_ranks):

    def count_not_terminated(node, terminated_ranks):
        return sum(
            1
            for leaf in node.iter_leaves()
            if leaf.state.mode != Mode.TERMINATED and leaf.state.rank not in terminated_ranks
        )

    for child in node.children.values():
        terminated_ranks = propagate_terminations(child, terminated_ranks)

    if not node.is_leaf() and count_not_terminated(node, terminated_ranks) < node.layer.min_ranks:
        terminated_ranks.update(
            set(
                leaf.state.rank for leaf in node.iter_leaves() if leaf.state.mode != Mode.TERMINATED
            )
        )

    return terminated_ranks


class Tree(RankAssignment):
    r'''
    Implements an integrated rank assignment and activation algorithm that
    builds a multi-layer topology tree for distributed ranks. Each layer in
    this tree specifies constraints and policies for assigning and activating
    ranks. Grouping keys in each layer can align with hardware properties
    (e.g., to confine ranks within a compute node) or application-driven
    requirements (e.g., ensuring a particular divisibility).

    :py:class:`Tree` constructs a rooted topology tree whose depth equals the
    number of layers. Each layer corresponds to a :py:class:`Layer`,
    determining the rank assignment policy within its subtree. The distributed
    ranks are represented as leaves.

    **Algorithm**

    **Initialization**

    The algorithm traverses all ranks in depth-first order. For each visited
    rank, if all ancestor layers permit more active ranks (i.e., if the
    already-active ranks do not exceed any ancestor layer’s
    :py:attr:`Layer.max_ranks`), that rank is activated.

    **Rank reassignment**

    When some ranks terminate or become unhealthy, the algorithm proceeds in
    several steps:

    1. **Propagate termination**

    Using a reverse depth-first search (children before parents), if the number
    of healthy ranks in a branch falls below :py:attr:`Layer.min_ranks`, that
    entire branch (and its subtree) is terminated.


    2. **Replace ranks from a reserve domain**

    The algorithm attempts to replace terminated or unhealthy active ranks with
    inactive ranks from the nearest ancestor subtree that has the
    :py:attr:`LayerFlag.RESERVE` flag. This search for an inactive rank
    continues recursively upward until a branch without the
    :py:attr:`LayerFlag.RESERVE` flag is reached.


    3. **Backfill ranks**

    Within any ancestor subtree flagged as :py:attr:`LayerFlag.BACKFILL`, an
    active rank with the largest rank index swaps places with a terminated
    rank, effectively filling local gaps (similar to :py:class:`FillGaps`).

    4. **Shift ranks**

    After local backfills, remaining gaps from unhandled terminations are
    closed by shifting healthy ranks left to fill any vacated indices.
    This global step reassigns all rank indices past the first unhealthy
    rank (similar to :py:class:`ShiftRanks`).

    5. **Optional filter**

    If a ``world_size_filter`` callable is provided, it can reduce the
    active ranks to a smaller ``world_size`` necessary for the workload.
    ``world_size_filter`` is invoked with the current number of active ranks as
    an argument, and returns the adjusted number of requested active ranks, no
    greater than the input. Healthy ranks with indices greater than the value
    returned value are deactivated and become part of the reserve pool.

    .. note::
        :py:class:`Tree` cannot be composed with any other instance of
        :py:class:`RankAssignment` or :py:class:`RankFilter`.

    **Example**

    .. code-block:: python

        inprocess.rank_assignment.Tree(
            [
                inprocess.rank_assignment.Layer(
                    min_ranks=128,
                    max_ranks=256,
                    key_or_fn='root',
                    flag=inprocess.rank_assignment.LayerFlag.RESERVE,
                ),
                inprocess.rank_assignment.Layer(
                    min_ranks=8,
                    max_ranks=8,
                    key_or_fn=lambda _: socket.gethostname(),
                    flag=inprocess.rank_assignment.LayerFlag.RESERVE,
                ),
            ]
        )

    In this two-level topology tree:

    - The first layer (hostname-based) allows up to 8 active ranks per host
      (:py:attr:`Layer.max_ranks=8`). If the number of healthy ranks in any
      host drops below 8 (:py:attr:`Layer.min_ranks=8`), that entire host’s
      subtree is terminated. The algorithm can look for inactive reserve ranks
      within the same hostname because of the :py:attr:`LayerFlag.RESERVE` flag.

    - All hosts are grouped under the ``'root'`` layer, which permits up to 256
      active ranks (:py:attr:`Layer.max_ranks=256`). If the global healthy
      rank count drops below 128 (:py:attr:`Layer.min_ranks=128`), all ranks
      are terminated. The :py:attr:`LayerFlag.RESERVE` flag at the ``'root'``
      level lets the algorithm traverse upward from one host to another host
      through the ``'root'`` to search for reserve ranks.

    Args:
        layers: a list of :py:class:`Layer` instances, each layer specifies
            properties corresponding to one grouping level in a topology tree
        world_size_filter: an optional ``Callable`` that takes the final
            application-visible world size, and returns the new world size, no
            greater than the input

    '''

    def __init__(
        self,
        layers: list[Layer],
        world_size_filter: Optional[Callable[int, int]] = None,
    ):
        self.layers = layers
        self.world_size_filter = world_size_filter

        self.tree = None
        self.rank_map = {}
        self.init_rank_map = {}

    def build_tree(self, state, store):
        key = [
            (layer.key_or_fn(state) if callable(layer.key_or_fn) else layer.key_or_fn)
            for layer in self.layers
        ]

        store.send_state(state, state.rank)
        states = store.get_states(range(state.world_size))

        store.send_key(key, state.rank)
        keys = store.get_keys(range(state.world_size))

        root_keys = set(key[0] for key in keys)
        if len(root_keys) != 1:
            msg = (
                f'all distributed ranks are required to share the same '
                f'grouping key at the root level of the topology tree, but '
                f'got {root_keys=}'
            )
            raise RuntimeError(msg)

        self.tree = Node(parent=None, name=keys[0], layer=self.layers[0], state=None)
        for key, state in zip(keys, states):
            node = self.tree
            for k, layer in zip(key[1:], self.layers[1:]):
                if k in node.children:
                    node = node.children[k]
                else:
                    node.add_child(k, layer, None)
                    node = node.children[k]

            child = node.add_child(state.initial_rank, None, state)
            self.init_rank_map[state.initial_rank] = child
            self.rank_map[state.rank] = child

        for idx, leaf in enumerate(self.tree.iter_leaves()):
            if idx != leaf.state.rank:
                topology_rank = idx
                environment_rank = leaf.state.rank
                msg = (
                    f'Initial environment rank assignment does not match the '
                    f'specified topology: {topology_rank=} {environment_rank=}'
                )
                warnings.warn(msg)

    def replace_with_inactive(self, terminated_active_ranks):
        replaced_terminate_active_ranks = set()

        for terminated_active_rank in sorted(terminated_active_ranks):
            terminated_active_node = self.rank_map[terminated_active_rank]

            node = terminated_active_node
            while (
                (parent := node.parent)
                and parent.layer.flag
                and parent.layer.flag & LayerFlag.RESERVE
            ):
                if parent.inactive_nodes:
                    _, inactive = parent.inactive_nodes.popitem()
                    inactive.activate(terminated_active_node.state.active_rank)
                    replaced_terminate_active_ranks.add(terminated_active_rank)
                    break
                node = parent

        return replaced_terminate_active_ranks

    def replace_with_backfill(self, unhandled_terminations):
        replaced_active = set()
        backfill_domains = collections.defaultdict(list)
        backfill_domains_id_map = {}

        for rank in sorted(unhandled_terminations):
            terminated_node = self.rank_map[rank]
            backfill_domain = terminated_node.backfill_domain

            if backfill_domain is not None:
                backfill_domains[id(backfill_domain)].append(terminated_node)
                backfill_domains_id_map[id(backfill_domain)] = backfill_domain
            else:
                replaced_active.add(terminated_node.state.active_rank)

        for domain_id, terminated_nodes in backfill_domains.items():
            backfill_domain = backfill_domains_id_map[domain_id]
            largest_active_nodes = heapq.nlargest(
                len(terminated_nodes),
                (leaf for leaf in backfill_domain.iter_leaves() if leaf.state.mode == Mode.ACTIVE),
                key=lambda node: node.state.active_rank,
            )

            for backfill_node, terminated_node in itertools.zip_longest(
                reversed(largest_active_nodes),
                terminated_nodes,
                fillvalue=None,
            ):
                if backfill_node is not None:
                    replaced_active.add(backfill_node.state.active_rank)
                    backfill_node.state.active_rank = terminated_node.state.active_rank
                else:
                    replaced_active.add(terminated_node.state.active_rank)

        return replaced_active

    def shift_ranks(self, replaced_active, unhandled_terminations):
        sorted_replaced_active = sorted(replaced_active)

        for n in self.rank_map.values():
            n.state.active_world_size -= len(unhandled_terminations)

            if n.state.active_rank is not None:
                count_less = bisect.bisect_left(sorted_replaced_active, n.state.active_rank)
                n.state.active_rank -= count_less

    def filter_active_world_size(self):
        active_world_size = next(iter(self.rank_map.values())).state.active_world_size

        new_active_world_size = self.world_size_filter(active_world_size)
        assert new_active_world_size <= active_world_size

        for leaf in self.tree.iter_leaves():
            leaf.state.active_world_size = new_active_world_size
            if leaf.state.mode == Mode.ACTIVE and leaf.state.active_rank >= new_active_world_size:
                leaf.deactivate()

    def recompute_rank(self):
        leaves = list(self.tree.iter_leaves())
        leaves.sort(key=lambda leaf: leaf.state.mode == Mode.TERMINATED)
        for idx, leaf in enumerate(leaves):
            leaf.state.rank = idx
            self.rank_map[idx] = leaf

    def update_tree(self, state, terminated_ranks):
        world_size = state.world_size - len(terminated_ranks)

        for node in self.tree.iter_leaves():
            node.state.world_size = world_size

        for terminated_rank in terminated_ranks:
            self.rank_map[terminated_rank].state.rank = None
            self.rank_map[terminated_rank].state.active_rank = None

    def get_state_from_tree(self, state, terminated_ranks):
        tree_state = self.init_rank_map[state.initial_rank].state

        if tree_state.mode == Mode.TERMINATED:
            raise RankDiscarded(f'{state.rank=} {terminated_ranks=}')

        state = State(**dataclasses.asdict(tree_state))
        return state

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = ctx.state
        store = ctx.store
        terminated_ranks = ctx.terminated_ranks

        if self.tree is None:
            self.build_tree(state, store)

            active_world_size = bounded_activate(self.tree, 0)
            for node in self.rank_map.values():
                node.state.active_world_size = active_world_size

            for leaf in self.tree.iter_leaves():
                leaf.assign_backfill_domain()

        for leaf in self.tree.iter_leaves():
            leaf.state.copy_from(state, fields=['fn_exception', 'iteration'])

        terminated_ranks = propagate_terminations(self.tree, terminated_ranks)

        terminated_active_ranks = set(
            rank for rank in terminated_ranks if self.rank_map[rank].state.mode == Mode.ACTIVE
        )
        for terminated_rank in terminated_ranks:
            self.rank_map[terminated_rank].terminate()

        replaced_terminate_active_ranks = self.replace_with_inactive(terminated_active_ranks)

        unhandled_terminations = terminated_active_ranks - replaced_terminate_active_ranks
        if unhandled_terminations:
            replaced_active = self.replace_with_backfill(unhandled_terminations)
            self.shift_ranks(replaced_active, unhandled_terminations)

        if self.world_size_filter is not None:
            self.filter_active_world_size()

        self.update_tree(state, terminated_ranks)
        self.recompute_rank()

        ctx.state = self.get_state_from_tree(state, terminated_ranks)
        ctx.terminated_ranks = set()

        return ctx


class FillGaps(RankAssignment):
    r'''
    A class for reassigning distributed ranks, filling in gaps caused by
    terminated or unhealthy ranks.

    The :py:class:`FillGaps` class is a specialized rank assignment strategy
    that reorders ranks to fill gaps created by terminated or unhealthy ranks.
    It preserves the previous rank assignment for the first ``world_size -
    len(terminated_ranks)`` healthy ranks; the remaining healthy ranks are
    reassigned to fill in gaps left by unhealthy ranks.

    Example:

    .. code-block:: python

        |<--- preserved --->|<- moved ->|     |<--new world size->|

        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | 0 | 6 | 2 | 3 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
              ^           ^        |  |
              |           |        |  |
              ---------------------   |
                          |           |
                          -------------
    '''

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = ctx.state
        rank = state.rank
        world_size = state.world_size
        terminated_ranks = ctx.terminated_ranks

        ordered_terminated_ranks = sorted(list(terminated_ranks))
        world_size = world_size - len(terminated_ranks)

        if rank in terminated_ranks:
            raise RankDiscarded(f'{rank=} {terminated_ranks=}')
        elif rank >= world_size:
            rank = ordered_terminated_ranks[rank - world_size]

        state = dataclasses.replace(
            state,
            rank=rank,
            world_size=world_size,
        )
        ctx.state = state
        ctx.terminated_ranks = set()
        return ctx


class ShiftRanks(RankAssignment):
    r'''
    A class for reassigning distributed ranks, filling in gaps caused by
    terminated or unhealthy ranks.

    The :py:class:`ShiftRanks` class is a specialized rank assignment strategy
    that shifts all healthy ranks to the left to fill gaps created by
    terminated or unhealthy ranks. :py:class:`ShiftRanks` preserves the
    relative order of all healthy ranks, but all ranks past the first unhealthy
    rank are reassigned (shifted).

    Example:

    .. code-block:: python

         <-   ->|<------- moved ------->|     |<--new world size->|

                  ----
                  v   |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | 0 | 2 | 3 | 6 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+
              ^   |   ^   ^       |   |
              |   |   |   |       |   |
              ----     ------------   |
                          |           |
                          ------------

    '''

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        state = ctx.state
        rank = state.rank
        world_size = state.world_size
        terminated_ranks = ctx.terminated_ranks

        world_size = world_size - len(terminated_ranks)
        if rank in terminated_ranks:
            raise RankDiscarded(f'{rank=} {terminated_ranks=}')
        else:
            rank = rank - sum(rank > terminated_rank for terminated_rank in terminated_ranks)

        state = dataclasses.replace(
            state,
            rank=rank,
            world_size=world_size,
        )
        ctx.state = state
        ctx.terminated_ranks = set()
        return ctx


class FilterCountGroupedByKey(RankAssignment):
    r'''
    A class for filtering distributed ranks by grouping by a key.

    :py:class:`FilterCountGroupedByKey` organizes ranks into groups based on a
    specified string key. For each group, it increments a group counter by 1
    for every healthy rank. A given boolean ``condition`` is then evaluated for
    each rank, with the corresponding group counter passed as input.

    - If ``condition(group_counter)`` evaluates to ``True``, the rank is
      preserved.
    - If it evaluates to ``False``, the rank is considered unhealthy and marked
      for termination.

    :py:class:`FilterCountGroupedByKey` needs to be followed by another
    :py:class:`RankAssignment` that performs the actual rank termination by
    raising :py:exc:`RankDiscarded` exception.

    .. code-block:: python

        condition = lambda count: count == 2

        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+---+---+---+
        | 0 | X | 2 | 3 | X | X | 6 | 7 | --> | X | X | 2 | 3 | X | X | 6 | 7 |
        +---+---+---+---+---+---+---+---+     +---+---+---+---+---+---+---+---+
        | key=0 | key=1 | key=2 | key=3 |     | key=0 | key=1 | key=2 | key=3 |
        |       |       |       |       |     |       |       |       |       |
        |count=1|count=2|count=0|count=2|     | False | True  | False | True  |

    Example:

    .. code-block:: python

        # hostname is the group key, and condition checks if exactly 8 ranks
        # corresponding to a given hostname are in a healthy state, if the
        # count is different than 8, all ranks from corresponding hostname are
        # considered unhealthy, and terminated; remaining healthy ranks are
        # shifted to the left to fill all gaps created by unhealthy ranks.

        rank_assignment = (
            inprocess.Compose(
                inprocess.rank_assignment.ShiftRanks(),
                inprocess.rank_assignment.FilterCountGroupedByKey(
                    key_or_fn=lambda _: socket.gethostname(),
                    condition=lambda count: count == 8,
                ),
            ),
        ),


    Args:
        key_or_fn: a string key, or a ``Callable`` evaluated with
            :py:class:`inprocess.state.State` as the input to produce a string
            key
        condition: condition to be evaluated with group counter as the input,
            if ``False`` the rank is terminated
        timeout: timeout for distributed barrier

    '''

    instance_count = 0

    def __init__(
        self,
        key_or_fn: Union[str, Callable[[State], str]],
        condition: Callable[int, bool],
        timeout: datetime.timedelta = datetime.timedelta(seconds=60),
    ):
        self.key_or_fn = key_or_fn
        self.condition = condition
        self.timeout = timeout

        self.name = f'{type(self).__name__}_{type(self).instance_count}'
        type(self).instance_count += 1

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        COUNT_ALIVE_BARRIER = f'count_alive_barrier_{self.name}'
        SUBMIT_MISMATCHING_BARRIER = f'submit_mismatching_barrier_{self.name}'
        RANKS_TO_TERMINATE = f'ranks_to_terminate_{self.name}'

        rank = ctx.state.rank
        world_size = ctx.state.world_size
        store = ctx.store
        terminated_ranks = ctx.terminated_ranks

        alive_world_size = world_size - len(terminated_ranks)

        if rank not in terminated_ranks:
            key = self.key_or_fn(ctx.state) if callable(self.key_or_fn) else self.key_or_fn
            prefixed_key = f'filter_grouped_by_key_{self.name}_{key}'
            store.add(prefixed_key, 1)
            store.barrier(
                ranks=[rank],
                group_name=COUNT_ALIVE_BARRIER,
                rendezvous_count=alive_world_size,
                timeout=self.timeout,
            )

            if not self.condition(int(store.get(prefixed_key))):
                store.append(RANKS_TO_TERMINATE, f'{rank},')

            store.barrier(
                ranks=[rank],
                group_name=SUBMIT_MISMATCHING_BARRIER,
                rendezvous_count=alive_world_size,
                timeout=self.timeout,
            )

            if store.check([RANKS_TO_TERMINATE]):
                ranks_to_terminate = set(
                    int(r) for r in store.get(RANKS_TO_TERMINATE).decode().rstrip(',').split(',')
                )
            else:
                ranks_to_terminate = set()

            ctx.terminated_ranks = terminated_ranks.union(ranks_to_terminate)

        return ctx
