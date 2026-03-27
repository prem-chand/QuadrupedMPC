.. QuadrupedMPC documentation master file

Welcome to QuadrupedMPC's documentation!
=========================================

A modular Model Predictive Control (MPC) framework for quadruped locomotion,
implementing MIT Cheetah-style convex MPC for the Unitree Go2 robot.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Start
-----------

.. code-block:: bash

   pip install numpy mujoco cvxpy clarabel
   python main.py

Core Concepts
-------------

- **Centroidal Dynamics MPC**: Convex QP formulation
- **Whole-Body Control**: Jacobian-transpose WBC
- **Simulator-Agnostic**: Zero simulator deps in controllers
- **GPU Batched**: Optional PyTorch-based parallel solving

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
