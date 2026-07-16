API reference
=============

The API reference is generated from the package's source docstrings on every
documentation build.

Estimators
----------

.. currentmodule:: overlapindex

.. autoclass:: OverlapIndex
   :members:
   :show-inheritance:

.. autoclass:: ContinuousOverlapIndex
   :members:
   :show-inheritance:

``predict`` returns global prototype ids for both estimators; it is not a
class-label or continuous-target prediction method. See :doc:`getting_started`
for method return values and :doc:`diagnostics` for fitted attributes.

Ball-cover backend
------------------

.. currentmodule:: overlapindex.BallCover

.. autoclass:: BallCoverManyToOne
   :members:
   :show-inheritance:

The ball-cover class is primarily a backend integration surface. Most users
should configure it through ``OverlapIndex(model_type="BallCover", ...)`` or
``ContinuousOverlapIndex(model_type="BallCover", ...)``; see
:doc:`backends/ballcover`.
