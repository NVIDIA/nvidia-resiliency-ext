Rank Assignment
===============================================================================

Rank Assignment
---------------
Base class
^^^^^^^^^^
.. autoclass:: nvidia_resiliency_ext.inprocess.rank_assignment.RankAssignment
    :special-members: __call__

.. autoclass:: nvidia_resiliency_ext.inprocess.rank_assignment.RankAssignmentCtx

.. autoexception:: nvidia_resiliency_ext.inprocess.rank_assignment.RankDiscarded

Tree
^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: Layer, LayerFlag, Tree

Composable Rank Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: FillGaps, ShiftRanks, FilterCountGroupedByKey
    :no-index:

Rank Filtering
--------------
Base class
^^^^^^^^^^
.. autoclass:: nvidia_resiliency_ext.inprocess.rank_assignment.RankFilter
    :special-members: __call__

Rank Filters
^^^^^^^^^^^^
.. automodule:: nvidia_resiliency_ext.inprocess.rank_assignment
    :members: ActivateAllRanks, MaxActiveWorldSize, ActiveWorldSizeDivisibleBy
    :no-index:
