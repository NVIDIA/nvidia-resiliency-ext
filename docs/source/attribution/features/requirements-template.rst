Attribution Feature Requirements Template
=========================================

Use this template for new attribution product features. Copy it to a
feature-specific file in this directory, for example
``<feature-slug>.rst``.

Summary
-------

Describe the feature in one or two paragraphs. Name the user-visible behavior,
the product surface it changes, and the expected outcome.

Problem
-------

Describe the customer or operator problem this feature solves. Include the
current failure mode, workflow gap, or missing attribution signal.

Goals
-----

* Define the intended behavior.
* Identify the primary user or caller.
* State the output contract or decision the feature should support.

Non-Goals
---------

* List adjacent behavior that is intentionally out of scope.
* Call out any lifecycle, service, UI, or policy changes this feature will not
  own.

User Flows
----------

Describe the expected flows, including input data, trigger points, and how the
result is consumed by users, services, or automation.

Functional Requirements
-----------------------

* Requirement 1.
* Requirement 2.
* Requirement 3.

API and Data Contract
---------------------

Document any public Python API, service API, result payload, dataflow record,
Slack notification, or compatibility contract this feature changes.

Configuration
-------------

List new config fields, defaults, environment variables, feature gates, and
deployment assumptions.

Observability
-------------

Describe logging, metrics, dataflow records, Slack output, status reporting,
and failure visibility.

Compatibility
-------------

Document backward compatibility requirements, migration behavior, and
interactions with existing attribution modes.

Validation
----------

List unit, integration, service, and end-to-end checks required before the
feature can ship.

Open Questions
--------------

* Question 1.
* Question 2.
