iterreg
=======

|image0| |image1|

Fast iterative regularization approach for non strongly convex regularizers, e.g. sparsity (L1) or low rank (nuclear norm).

Documentation
=============

Please visit https://LCSL.github.io/iterreg/ for the latest version
of the documentation.


Install and work with the development version
=============================================

From a console or terminal clone the repository and install iterreg:

::

    git clone https://github.com/LCSL/iterreg.git
    cd iterreg/
    pip install -e .

To build the documentation you will need to run:


::

    cd doc/
    pip install doc_requirements.txt
    make html


Demos & Examples
================

In the `example section <https://LCSL.github.io/iterreg/auto_examples/index.html>`__ of the documentation,
you will find scripts to reproduce the figures of the paper.

Dependencies
============

All dependencies are in the ``./requirements.txt`` file.
They are installed automatically when ``pip install -e .`` is run.

Cite
====

If you use this code, please cite:

::

  @inproceedings{iterreg,
    title = {Iterative regularization with convex regularizers},
    author = {Molinari, Cesare and Massias, Mathurin and Rosasco, Lorenzo and Villa, Silvia},
    booktitle = AISTATS,
    year = 2021,
  }


ArXiv links:

- https://arxiv.org/abs/2006.09859


.. |image0| image:: https://github.com/LCSL/iterreg/workflows/build/badge.svg
   :target: https://github.com/LCSL/iterreg/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/LCSL/iterreg/branch/main/graphs/badge.svg?branch=main
   :target: https://codecov.io/gh/LCSL/iterreg
