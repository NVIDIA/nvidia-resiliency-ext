Rank Assignment
===============================================================================

Rank Assignment
---------------
Base class
^^^^^^^^^^
.. autoclass:: inprocess.rank_assignment.RankAssignment
    :special-members: __call__

.. autoclass:: inprocess.rank_assignment.RankAssignmentCtx

.. autoexception:: inprocess.rank_assignment.RankDiscarded

Tree
^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: Layer, LayerFlag, Tree

Composable Rank Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: FillGaps, ShiftRanks, FilterCountGroupedByKey

Rank Filtering
--------------
Base class
^^^^^^^^^^
.. autoclass:: inprocess.rank_assignment.RankFilter
    :special-members: __call__

Rank Filters
^^^^^^^^^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: ActivateAllRanks, MaxActiveWorldSize, ActiveWorldSizeDivisibleBy
