# -*- coding: utf-8 -*-
"""Write PyHawk spherical-harmonic solutions in ICGEM GFC format."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.Auxilary.GeoMathKit import GeoMathKit


class StandardSH:
    """Standardize a complete triangular set of spherical harmonics."""

    DEFAULT_RADIUS = 6.3781363e6
    DEFAULT_GM = 3.986004415e14
    README_LINK = 'https://github.com/NCSGgroup/PyHawk/blob/main/README.md'
    LICENSE_LINK = 'https://github.com/NCSGgroup/PyHawk/blob/main/LICENSE'
    GENERATING_INSTITUTE = 'https://github.com/NCSGgroup'
    GRACE_FO_SUMMARY = (
        "Spherical harmonic coefficients representing an estimate of Earth's "
        'mean gravity field during the specified timespan derived from '
        'GRACE-FO mission measurements. These coefficients represent the full '
        'magnitude of land hydrology, ice, and solid Earth processes. Further, '
        'they represent atmospheric and oceanic processes not captured in the '
        'accompanying GAC product.'
    )
    GRACE_FO_ACKNOWLEDGEMENT = (
        'GRACE-FO is a joint mission of the US National Aeronautics and Space '
        'Administration and the German Research Center for Geosciences. Use '
        '"Dahle et al. (2019): GRACE-FO Geopotential GSM Coefficients GFZ '
        'RL06. V. 6.0. GFZ Data Services. '
        'http://doi.org/10.5880/GFZ.GRACEFO_06_GSM" when citing this data.'
    )

    def __init__(self):
        self.order_by_degree = False
        self.product_type = 'gravity_field'
        self.model_center = 'Hawk'
        self.release = '06'
        self.product_version = 'BA01'
        self.solution_id = '0100'
        self.modelname = None
        self.radius = self.DEFAULT_RADIUS
        self.earth_gravity_constant = self.DEFAULT_GM
        self.max_degree = None
        self.norm = 'fully_normalized'
        self.tide_system = 'zero_tide'
        self.errors = 'formal'

        self._file_directory = None
        self._file_path = None
        self._date_span = None
        self._mission = None
        self._unused_days = ()
        self._c = None
        self._s = None
        self._sigma_c = None
        self._sigma_s = None

    @property
    def file_path(self):
        return self._file_path

    @property
    def orderFirst(self):
        """Compatibility alias for the legacy degree-first switch."""
        return self.order_by_degree

    @orderFirst.setter
    def orderFirst(self, value):
        self.order_by_degree = bool(value)

    def configure(self, filedir, data, degree, c, s, sigmaC, sigmaS,
                  mission=None, reference_model=None, unused_days=None):
        """Configure one unfiltered GSM solution.

        Coefficient arrays use PyHawk's triangular one-dimensional sequence:
        00, 10, 11, 20, 21, 22, ... . The written GFC records remain ordered
        by order to match the existing CSR/GFZ/JPL reference products.
        """
        self._date_span = self._validate_date_span(data)
        self._mission = mission
        self._unused_days = self._validate_unused_days(
            unused_days, self._date_span
        )
        self.max_degree = self._validate_degree(degree)
        expected_size = (self.max_degree + 1) * (self.max_degree + 2) // 2

        self._c = self._validate_coefficients('c', c, expected_size)
        self._s = self._validate_coefficients('s', s, expected_size)
        self._sigma_c = self._validate_coefficients(
            'sigmaC', sigmaC, expected_size, nonnegative=True
        )
        self._sigma_s = self._validate_coefficients(
            'sigmaS', sigmaS, expected_size, nonnegative=True
        )

        self._apply_reference_model(reference_model)
        mission_code = self._mission_code(mission)
        start, end = self._date_span
        product_id = 'GSM-2_%s-%s_%s_%s_%s_%s' % (
            start, end, mission_code, self.model_center,
            self.product_version, self.solution_id
        )
        self.modelname = product_id
        filename = self.product_filename(
            data=self._date_span,
            degree=self.max_degree,
            mission=mission,
            solution_id=self.solution_id
        )

        self._file_directory = Path(filedir) / Path(filename).stem
        self._file_directory.mkdir(parents=True, exist_ok=True)
        self._file_path = self._file_directory / filename
        return self

    @classmethod
    def product_filename(cls, data, degree, mission=None,
                         solution_id='0100'):
        """Return the canonical filename without creating the product."""
        start, end = cls._validate_date_span(data)
        degree = cls._validate_degree(degree)
        mission_code = cls._mission_code(mission)
        return 'Hawk_%dx%d_unfiltered_GSM-2_%s-%s_%s_%s.gfc' % (
            degree, degree, start, end, mission_code, solution_id
        )

    def unfiltered(self):
        """Write the configured solution atomically and return its path."""
        if self._file_path is None:
            raise RuntimeError('StandardSH must be configured before writing')

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='ascii', newline='\n', delete=False,
                    dir=str(self._file_directory), suffix='.tmp') as file:
                temporary_path = Path(file.name)
                self._write_header(file)
                self._write_coefficients(file)
            os.replace(str(temporary_path), str(self._file_path))
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise

        return self._file_path

    def _write_header(self, file):
        created = datetime.now(timezone.utc)
        file.write('*' * 80 + '\n')
        file.write(
            'model converted into ICGEM-format at: %s\n'
            % created.strftime('%Y-%m-%dT%H:%M:%SZ')
        )
        file.write('*' * 80 + '\n\n')
        self._write_yaml_metadata(file, created)
        file.write('begin_of_head ' + '=' * 65 + '\n')
        self._write_header_value(file, 'product_type', self.product_type)
        self._write_header_value(file, 'modelname', self.modelname)
        self._write_header_value(file, 'radius', '%.16E' % self.radius)
        self._write_header_value(
            file, 'earth_gravity_constant',
            '%.16E' % self.earth_gravity_constant
        )
        self._write_header_value(file, 'max_degree', self.max_degree)
        self._write_header_value(file, 'norm', self.norm)
        self._write_header_value(file, 'tide_system', self.tide_system)
        self._write_header_value(file, 'errors', self.errors)
        file.write('\n')
        file.write(
            '%-10s%-7s%-10s%-25s%-25s%-25s%-25s\n'
            % ('key', 'L', 'M', 'C', 'S', 'sigma C', 'sigma S')
        )
        file.write('end_of_head ' + '=' * 67 + '\n')

    def _write_yaml_metadata(self, file, created):
        start = datetime.strptime(self._date_span[0], '%Y-%m-%d')
        last_day = datetime.strptime(self._date_span[1], '%Y-%m-%d')
        coverage_end = last_day + timedelta(days=1)
        midpoint = start + (coverage_end - start) / 2
        unused_days = '[%s]' % ', '.join(self._unused_days)

        file.write('**** some information from original YAML header ****\n')
        self._write_yaml_value(file, 'dictionary', self.README_LINK)
        self._write_yaml_value(file, 'summary', self._summary())
        self._write_yaml_value(
            file, 'history',
            'Original solution produced on %s'
            % created.strftime('%Y-%m-%dT%H:%M:%S')
        )
        self._write_yaml_value(
            file, 'acknowledgement', self._acknowledgement()
        )
        self._write_yaml_value(file, 'license', self.LICENSE_LINK)
        file.write('\n')
        self._write_yaml_value(
            file, 'time_coverage_start',
            start.strftime('%Y-%m-%dT%H:%M:%S.00')
        )
        self._write_yaml_value(
            file, 'time_coverage_end',
            coverage_end.strftime('%Y-%m-%dT%H:%M:%S.00')
        )
        self._write_yaml_value(file, 'unused_days', unused_days)
        file.write('**********  end of original YAML header  ***********\n')
        file.write(
            'time_period_of_data    %s - %s   (mid: %s)\n'
            % (start.strftime('%Y%m%d'), last_day.strftime('%Y%m%d'),
               midpoint.strftime('%Y%m%d'))
        )
        file.write(
            'generating_institute    %s\n\n' % self.GENERATING_INSTITUTE
        )

    @staticmethod
    def _write_yaml_value(file, key, value):
        file.write('%-22s: %s\n' % (key, value))

    def _summary(self):
        if self._is_grace_fo():
            return self.GRACE_FO_SUMMARY
        return self.GRACE_FO_SUMMARY.replace('GRACE-FO', 'GRACE')

    def _acknowledgement(self):
        if self._is_grace_fo():
            return self.GRACE_FO_ACKNOWLEDGEMENT
        return (
            'GRACE is a joint mission of the US National Aeronautics and Space '
            'Administration and the German Aerospace Center.'
        )

    @staticmethod
    def _write_header_value(file, key, value):
        file.write('%-31s%s\n' % (key, value))

    def _write_coefficients(self, file):
        c = GeoMathKit.CS_1dTo2d(self._c)
        s = GeoMathKit.CS_1dTo2d(self._s)
        sigma_c = GeoMathKit.CS_1dTo2d(self._sigma_c)
        sigma_s = GeoMathKit.CS_1dTo2d(self._sigma_s)

        if self.order_by_degree:
            indices = (
                (degree, order)
                for degree in range(self.max_degree + 1)
                for order in range(degree + 1)
            )
        else:
            indices = (
                (degree, order)
                for order in range(self.max_degree + 1)
                for degree in range(order, self.max_degree + 1)
            )

        for degree, order in indices:
            file.write(
                'gfc %7d %6d  %+24.16E  %+24.16E  %+24.16E  %+24.16E\n'
                % (degree, order, c[degree, order], s[degree, order],
                   sigma_c[degree, order], sigma_s[degree, order])
            )

    def _apply_reference_model(self, reference_model):
        if reference_model is None:
            return

        radius = getattr(reference_model, 'Radius', None)
        gm = getattr(reference_model, 'GM', None)
        norm = getattr(reference_model, 'norm', None)
        zero_tide = getattr(reference_model, 'zero_tide', None)

        if radius is not None:
            self.radius = self._validate_positive_number('radius', radius)
        if gm is not None:
            self.earth_gravity_constant = self._validate_positive_number(
                'earth_gravity_constant', gm
            )
        if norm:
            self.norm = str(norm)
        if zero_tide is not None:
            self.tide_system = 'zero_tide' if bool(zero_tide) else 'tide_free'

    @staticmethod
    def _validate_date_span(data):
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError('data must contain start and end dates')

        parsed = []
        normalized = []
        for value in data:
            if not isinstance(value, str):
                raise TypeError('date values must be YYYY-MM-DD strings')
            try:
                epoch = datetime.strptime(value, '%Y-%m-%d')
            except ValueError as exc:
                raise ValueError('invalid date: %s' % value) from exc
            parsed.append(epoch)
            normalized.append(epoch.strftime('%Y-%m-%d'))

        if parsed[0] > parsed[1]:
            raise ValueError('start date must not be later than end date')
        return tuple(normalized)

    @staticmethod
    def _validate_unused_days(unused_days, date_span):
        if unused_days is None:
            return ()
        if not isinstance(unused_days, (list, tuple)):
            raise TypeError('unused_days must be a list or tuple of dates')

        start = datetime.strptime(date_span[0], '%Y-%m-%d')
        end = datetime.strptime(date_span[1], '%Y-%m-%d')
        normalized = []
        for value in unused_days:
            if not isinstance(value, str):
                raise TypeError('unused day values must be YYYY-MM-DD strings')
            try:
                day = datetime.strptime(value, '%Y-%m-%d')
            except ValueError as exc:
                raise ValueError('invalid unused day: %s' % value) from exc
            if start <= day <= end:
                normalized.append(day.strftime('%Y-%m-%d'))
        return tuple(sorted(set(normalized)))

    @staticmethod
    def _validate_degree(degree):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError('degree must be a non-negative integer')
        if degree < 0:
            raise ValueError('degree must be a non-negative integer')
        return int(degree)

    @staticmethod
    def _validate_coefficients(name, values, expected_size, nonnegative=False):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError('%s must be a one-dimensional array' % name)
        if array.size != expected_size:
            raise ValueError(
                '%s has %d values; degree requires %d'
                % (name, array.size, expected_size)
            )
        if not np.all(np.isfinite(array)):
            raise ValueError('%s contains non-finite values' % name)
        if nonnegative and np.any(array < 0):
            raise ValueError('%s must not contain negative values' % name)
        return array.copy()

    @staticmethod
    def _validate_positive_number(name, value):
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError('%s must be a finite positive number' % name)
        return number

    @staticmethod
    def _mission_code(mission):
        value = '' if mission is None else str(mission).upper()
        if 'GRACE_FO' in value or value.startswith('GFO'):
            return 'GRFO'
        if not value or value.startswith('GRACE_'):
            return 'GRAC'
        raise ValueError('unsupported SH mission: %s' % mission)

    def _is_grace_fo(self):
        value = '' if self._mission is None else str(self._mission).upper()
        return 'GRACE_FO' in value or value.startswith('GFO')


# Backward-compatible names used by existing PyHawk scripts.
FormatWrite = StandardSH


class FormatWrite_FO(StandardSH):
    def configure(self, filedir, data, degree, c, s, sigmaC, sigmaS,
                  mission='GRACE_FO_RL04', reference_model=None,
                  unused_days=None):
        return super().configure(
            filedir=filedir, data=data, degree=degree, c=c, s=s,
            sigmaC=sigmaC, sigmaS=sigmaS, mission=mission,
            reference_model=reference_model, unused_days=unused_days
        )
