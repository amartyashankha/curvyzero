#!/usr/bin/env python3
"""Build the opt-in vendored fixed-A=3 CTree profiling extension."""

from __future__ import annotations

from distutils.sysconfig import get_python_inc
from pathlib import Path
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


ROOT = Path(__file__).resolve().parents[1]
PYX = ROOT / "src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/mz_tree_a3.pyx"


def main() -> None:
    include_dirs = [np.get_include(), get_python_inc()]
    extension = Extension(
        "curvyzero.vendor.lightzero_ctree_a3.ctree_muzero.mz_tree_a3",
        [str(PYX)],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    )
    script_args = sys.argv[1:] or ["build_ext", "--inplace"]
    setup(
        name="curvyzero-lightzero-ctree-a3",
        package_dir={"": "src"},
        ext_modules=cythonize(
            [extension],
            compiler_directives={"language_level": "3"},
            force=True,
        ),
        script_args=script_args,
    )


if __name__ == "__main__":
    main()
