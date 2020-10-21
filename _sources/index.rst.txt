.. iterreg documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Iterreg
=======

This is a library to run iterative regularization for convex bias. [1].

Installation
------------
First clone the repository available at https://github.com/LCSL/iterreg::

    $ git clone https://github.com/LCSL/iterreg.git
    $ cd iterreg/


We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.

From a working environment, you can install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import iterreg'

and it should not give any error message.

From a Python shell you can just do::

    >>> import iterreg

If you don't want to use Anaconda, you should still be able to install using `pip`.

Cite
----

If you use this code, please cite:

.. code-block:: bibtex

  @article{arxiv,
    title = {Iterative regularization with convex regularizers},
    author = {Molinari, Cesare and Massias, Mathurin and Rosasco, Lorenzo and Villa, Silvia},
    journal = Arxiv preprint,
    year = 2020,
  }



ArXiv links:

- TODO




API
---

.. toctree::
    :maxdepth: 1

    api.rst
