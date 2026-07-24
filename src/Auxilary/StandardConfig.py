# -*- coding: utf-8 -*-
"""Archive run settings and quality-control files for an SH product."""

import os
import tempfile
from pathlib import Path


class StandardConfig:
    """Preserve directly reusable JSON settings at the start of a run."""

    def __init__(self, setting_directory):
        self._setting_directory = Path(setting_directory)
        if not self._setting_directory.is_dir():
            raise NotADirectoryError(
                'setting directory does not exist: %s'
                % self._setting_directory
            )

    def save(self, spherical_harmonic_path):
        """Copy every JSON setting directly to ``<GFC stem>_setting``."""
        product_path = Path(spherical_harmonic_path)
        output_directory = (
            product_path.parent
            / (product_path.stem + '_setting')
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        source_files = tuple(sorted(self._setting_directory.glob('*.json')))
        if not source_files:
            raise FileNotFoundError(
                'no JSON settings found in %s' % self._setting_directory
            )

        source_names = {path.name for path in source_files}
        for existing in output_directory.glob('*.json'):
            if existing.name not in source_names:
                existing.unlink()
        for source in source_files:
            self._atomic_copy(source, output_directory / source.name)
        return output_directory

    def archive_quality_control(self, spherical_harmonic_path,
                                arc_filter_path=None,
                                arc_definition_path=None):
        """Copy arc selection files beside the related GFC product."""
        product_path = Path(spherical_harmonic_path)
        outputs = {}
        if arc_filter_path is not None:
            output = product_path.with_name(
                product_path.stem + '_arcft.txt'
            )
            self._atomic_copy(Path(arc_filter_path), output)
            outputs['arc_filter'] = output
        if arc_definition_path is not None:
            output = product_path.with_name(
                product_path.stem + '_arc.txt'
            )
            self._atomic_copy(Path(arc_definition_path), output)
            outputs['arc_definition'] = output
        return outputs

    @staticmethod
    def _atomic_copy(source, output):
        if not source.is_file():
            raise FileNotFoundError(
                'source file does not exist: %s' % source
            )
        raw = source.read_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb', delete=False, dir=str(output.parent),
                    suffix='.tmp') as file:
                temporary_path = Path(file.name)
                file.write(raw)
            os.replace(str(temporary_path), str(output))
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise
