Usage guide
############


Design Overview
****************

Package users can mark sections of their code as "straggler detection sections". 
Timing information is collected for such sections, which includes CPU (wall-clock) time spent in the section. 
Additionally, CUDA kernels launched in the detection sections can be benchmarked. 

Based on the timings collected, the following performance scores are computed:
   * For each detection section, there is a per-rank performance score based on the measured CPU wall-clock time spent in the section.
   * [Optionally] All CUDA kernel timing information from all monitored sections is aggregated into a single per-rank GPU performance score.

Performance scores are scalar values from 0.0 (worst) to 1.0 (best), reflecting each rank's performance.
A performance score can be interpreted as the ratio of current performance to reference performance.                                                                                                                                
                                                                                                                                                                                
Depending on the reference used, there are two types of performance scores:                              
   * Relative performance score: The best-performing rank in the workload is used as a reference.                                                                                                                               
   * Individual performance score: The best historical performance of the rank is used as a reference.                                                                                                                          
                                                                                                                                                                                                                                    
Examples:                                                                                                                                                                                                                           
   * If the relative performance score is 0.5, it means that a rank is twice as slow as the fastest rank.         
   * If the individual performance score is 0.5, it means that a rank is twice as slow as its best performance.
                                                                                                                                                                                                                                    
If there are 8 ranks and 2 sections "data_loading" and "fwd" defined, scores might look like this:

.. code-block:: text

   Relative section performance scores:
   {
   'data_loading': {0: 0.961, 1: 0.919, 2: 0.941, 3: 0.999, 4: 1.0, 5: 0.847, 6: 0.591, 7: 0.788},
   'fwd': {0: 0.994, 1: 0.949, 2: 0.994, 3: 0.998, 4: 1.0, 5: 0.947, 6: 0.991, 7: 0.988}
   }
   Relative GPU performance scores:
   {
   0: 0.703, 1: 0.978, 2: 0.689, 3: 0.663, 4: 0.673, 5: 0.925, 6: 0.720, 7: 0.737
   }

   # NOTE: GPU performance scores are not related to any particular section.

   # 
   # Individual scores have the same structure as relative scores.
   #

If the performance score drops below the specified threshold, the corresponding section or GPU is identified as a straggler.         
To detect the stragglers, users should call the reporting function periodically and check the returned report object for stragglers.

Usage Overview
***************

Sections of the user code that should be monitored for performance ("detection sections") can be defined using:
   * ``Detector.detection_section`` context manager that allows wrapping a custom code block.
   * ``Detector.wrap_callables`` method that wraps given callables within ``Detector.detection_section``.

Both methods yield the same results. Users can choose the API that is most convenient for their needs.

.. note::
   If a detection section is monitored for CPU performance, a given section is supposed to run a similar amount of work on all ranks 
   (e.g., if a significant amount of additional work was conducted on rank 0, rank 0 would be identified as a straggler).
   Captured CUDA kernels can differ between ranks, but there should be some subset common to all ranks.
   The simplest approach might be to wrap the model's ``forward`` call.

Using ``Detector.detection_section``:

.. code-block:: python

   import nvidia_resiliency_ext.attribution.straggler as straggler

   straggler.Detector.initialize(
      scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
      gather_on_rank0=True # all ranks' results will be available on rank 0
   )

   for training_iter in range(num_iters):
      
      ...

      with straggler.Detector.detection_section("data_loading", profile_cuda=False):
         data, labels = data_loader.get_next_batch()

      with straggler.Detector.detection_section("fwd", profile_cuda=True):
         out = model(data)
      ...

      if (training_iter % STRAGGLER_REPORT_INTERVAL) == 0:
         report = straggler.Detector.generate_report()
         if rank == 0:
               stragglers = report.identify_stragglers()
               maybe_notify_user(stragglers)

   ...

   straggler.Detector.shutdown()

Using ``Detector.wrap_callables``:

.. code-block:: python

   import nvidia_resiliency_ext.straggler as straggler

   straggler.Detector.initialize(
      scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
      gather_on_rank0=True # all ranks' results will be available on rank 0
   )

   straggler.Detector.wrap_callables(
      callable_ids=[
         straggler.CallableId(DataLoaderClass, "get_next_batch", ignored_args=("self",)),
         straggler.CallableId(ModelClass, "forward", ignored_args=("self",)),
      ]
   )

   for training_iter in range(num_iters):
      
      ...
      
      data, labels = data_loader.get_next_batch()
      out = model(data)

      ...

      if (training_iter % STRAGGLER_REPORT_INTERVAL) == 0:
         report = straggler.Detector.generate_report()
         if rank == 0:
               stragglers = report.identify_stragglers()
               maybe_notify_user(stragglers)

   ...

   straggler.Detector.shutdown()

Alternative Reporting
---------------------
Besides calling ``Detector.generate_report`` after a fixed number of iterations tweaked for a particular workload, 
users can choose to call ``.generate_report_if_interval_elapsed`` along with each training step. 
The report generation will occur **approximately** within each time period specified through ``Detector.initialize(report_time_interval=...)``.

.. note::
   Straggler detection might involve inter-rank synchronization and should be invoked with reasonable frequency (e.g., every few minutes).

.. code-block:: python

   import nvidia_resiliency_ext.straggler as straggler

   straggler.Detector.initialize(report_time_interval=300, ...) # Report every 5 minutes

   for training_iter in range(num_iters):

      ...

      # During each training iteration
      report = straggler.Detector.generate_report_if_interval_elapsed()
      if report is not None:
         handle_report(report)

      # Note: straggler.Detector.generate_report() works as usual, users can alternatively use:
      # if (iter_idx % REPORT_INTERVAL) == 0:
      #     straggler.Detector.generate_report()

   ...

   straggler.Detector.shutdown()

Reducing the Overhead
---------------------

CUDA kernel profiling imposes some small step time overhead that depends on the workload but generally is expected to be <1%. 
If needed, the amount of overhead can be reduced with ``Detector.initialize(profiling_interval=...)``. 
If ``profiling_interval`` is > 1, only a fraction of section runs are profiled. 

``Detector.generate_report()`` overhead depends on the ``Detector.initialize`` parameters:
   * Relative performance score calculation (``scores_to_compute=["relative_perf_scores", ...]``) involves sharing some data between ranks.
   * If ``gather_on_rank0=True``, results from all ranks are gathered on rank 0.

Hence, the following initialization parameters can be used to avoid any inter-rank synchronization:

.. code-block:: python

   straggler.Detector.initialize(
      scores_to_compute=["individual_perf_scores"],
      gather_on_rank0=False # each rank will report its own results
   )

In that case, all ranks compute and report their own, individual results only. 

Integration Guide
******************

This section describes integration with a PTL-based workload (e.g., NeMo) using ``StragglerDetectionCallback``.
All that is needed is to include :class:`StragglerDetectionCallback <nvidia_resiliency_ext.ptl_resiliency.straggler_det_callback.StragglerDetectionCallback>` in the PTL trainer callbacks. 

.. code-block:: python

   from nvidia_resiliency_ext.ptl_resiliency import StragglerDetectionCallback

   straggler_cb_args = dict(
      report_time_interval=300.0,
      calc_relative_gpu_perf=True,
      calc_individual_gpu_perf=True,
      num_gpu_perf_scores_to_log=3,
      gpu_relative_perf_threshold=0.7,
      gpu_individual_perf_threshold=0.7,
      stop_if_detected=False,
      logger_name="test_logger",
   )

   straggler_det_cb = StragglerDetectionCallback(**straggler_cb_args)

   trainer = pl.Trainer(
      ...
      callbacks=[..., straggler_det_cb],
   )

``StragglerDetectionCallback`` initialization parameters:
   
   .. automethod:: nvidia_resiliency_ext.ptl_resiliency.StragglerDetectionCallback.__init__
