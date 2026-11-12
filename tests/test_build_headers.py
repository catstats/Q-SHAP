import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
from setuptools.errors import CompileError


@pytest.fixture
def build_setup():
    with patch('setuptools.setup'):
        return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))


def test_missing_headers_explain_interpreter_mismatch(build_setup, tmp_path):
    with patch('sysconfig.get_paths', return_value={'include': str(tmp_path)}):
        with pytest.raises(CompileError, match='python3-dev may target a different'):
            build_setup['check_python_headers']([])


def test_headers_in_custom_include_directory_are_accepted(build_setup, tmp_path):
    (tmp_path / 'Python.h').touch()
    with patch('sysconfig.get_paths', return_value={}):
        build_setup['check_python_headers']([str(tmp_path)])
