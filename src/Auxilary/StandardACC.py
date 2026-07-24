"""
@Description:
    Save calibrated PyHawk accelerometer data in a GRACE-FO ACT1B-like
    ASCII format. Arc results are cached independently and merged into one
    file per UTC day and satellite after orbit calibration is complete.
"""

import datetime
import glob
import os

import numpy as np


class StandardACC:
    _EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0)

    def __init__(self):
        self.__fileDir = None
        self.__cacheDir = None
        self.__dateSpan = None
        self.__mission = None
        self.__productName = 'Hawk_ACC1C'
        self.__productVersion = '01'

    def configure(self, filedir, date_span=None, mission=None,
                  product_name='Hawk_ACC1C', product_version='01'):
        self.__fileDir = os.path.abspath(str(filedir))
        self.__cacheDir = os.path.join(self.__fileDir, '.standard_acc_cache')
        self.__dateSpan = date_span
        self.__mission = mission
        self.__productName = product_name
        self.__productVersion = product_version
        return self

    @staticmethod
    def output_directory(non_conservative_directory):
        directory = os.path.abspath(str(non_conservative_directory))
        return os.path.join(os.path.dirname(directory), 'StandardACC')

    def prepare(self, satellites=None):
        """Remove stale arc snapshots for the configured time span."""
        self.__check_configured()
        os.makedirs(self.__cacheDir, exist_ok=True)
        satellite_ids = None
        if satellites is not None:
            satellite_ids = {self.__standard_satellite(item) for item in satellites}

        for filename in self.__cache_files():
            if satellite_ids is None or self.__cache_satellite(filename) in satellite_ids:
                os.remove(filename)
        if self.__dateSpan is not None:
            for day in self.__configured_days():
                for satellite_id in satellite_ids or ('C', 'D'):
                    filename = '%s_%s_%s_%s.txt' % (
                        self.__productName,
                        day,
                        satellite_id,
                        self.__productVersion
                    )
                    paths = (
                        os.path.join(self.__daily_directory(day), filename),
                        os.path.join(self.__fileDir, filename)
                    )
                    for path in paths:
                        if os.path.exists(path):
                            os.remove(path)
        return self

    def save_arc(self, gps_time, acceleration, bias, satellite, arc_no):
        """Atomically cache one calibrated arc; repeated iterations overwrite it."""
        self.__check_configured()
        time = np.asarray(gps_time, dtype=np.float64).reshape(-1)
        acc = np.asarray(acceleration, dtype=np.float64)
        if acc.ndim != 2 or acc.shape != (time.size, 3):
            raise ValueError('acceleration should have shape (epoch_number, 3)')
        bias = self.__format_bias(bias=bias, epoch_number=time.size)

        valid = np.isfinite(time)
        time = time[valid]
        acc = acc[valid]
        bias = bias[valid]
        satellite_id = self.__standard_satellite(satellite)

        os.makedirs(self.__cacheDir, exist_ok=True)
        filename = '%s_%s_arc%06d.npz' % (
            self.__date_tag(), satellite_id, int(arc_no)
        )
        target = os.path.join(self.__cacheDir, filename)
        temporary = target + '.tmp.npz'
        np.savez_compressed(
            temporary,
            gps_time=time,
            acceleration=acc,
            bias=bias,
            satellite=np.asarray(satellite_id)
        )
        os.replace(temporary, target)
        return target

    def finalize(self, remove_cache=True):
        """Merge all arc snapshots and write one standardized file per day."""
        self.__check_configured()
        grouped = {}
        cache_files = self.__cache_files()
        for filename in cache_files:
            with np.load(filename, allow_pickle=False) as data:
                time = np.asarray(data['gps_time'], dtype=np.float64)
                acc = np.asarray(data['acceleration'], dtype=np.float64)
                bias = np.asarray(data['bias'], dtype=np.float64)
                satellite = str(data['satellite'].item())

            day_numbers = self.__day_numbers(time)
            for day_number in np.unique(day_numbers):
                day = self.__day_label(day_number)
                if not self.__date_is_required(day):
                    continue
                index = day_numbers == day_number
                grouped.setdefault((day, satellite), []).append(
                    (time[index], acc[index], bias[index])
                )

        output_files = []
        for key in sorted(grouped):
            day, satellite = key
            time, acc, bias = self.__merge_chunks(grouped[key])
            filename = '%s_%s_%s_%s.txt' % (
                self.__productName, day, satellite, self.__productVersion
            )
            directory = self.__daily_directory(day)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, filename)
            self.__write_file(path, time, acc, bias, satellite)
            output_files.append(path)

        if remove_cache:
            for filename in cache_files:
                os.remove(filename)
        return output_files

    def __write_file(self, path, time, acc, bias, satellite):
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='ascii', newline='\n') as file:
            self.__write_header(file, time, bias.shape[1], satellite)
            for i in range(time.size):
                values = [self.__format_time(time[i]), satellite]
                values.extend('%.15e' % value for value in acc[i])
                values.extend('%.15e' % value for value in bias[i])
                file.write(' '.join(values) + '\n')
        os.replace(temporary, path)

    def __write_header(self, file, time, bias_dimension, satellite):
        created = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        start = self._EPOCH + datetime.timedelta(seconds=float(time[0]))
        stop = self._EPOCH + datetime.timedelta(seconds=float(time[-1]))
        platform = 'GRACE %s' % satellite
        mission_name = 'GRACE Follow-On' if self.__is_grace_fo() else 'GRACE'

        file.write('header:\n')
        file.write('  dimensions:\n')
        file.write('    num_records: %d\n' % time.size)
        file.write('  global_attributes:\n')
        file.write('    creator: Yi Wu(wu_yi@hust.edu.cn); Fan Yang(fany@plan.aau.dk)\n')
        file.write('    creator_institution: NCSG(https://github.com/NCSGgroup)\n')
        file.write('    date_created: %s\n' % created)
        file.write('    instrument: ACC\n')
        file.write('    source: PyHawk calibrated accelerometer data for %s\n' % platform)
        file.write('    summary: Calibrated linear acceleration and evaluated bias in GCRS\n')
        file.write('    time_coverage_start: %s\n' % start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    time_coverage_stop: %s\n' % stop.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    title: %s Hawk Level-1C Calibrated Accelerometer Data\n' % mission_name)
        file.write('  variables:\n')
        self.__write_primary_variables(file)
        for i in range(bias_dimension):
            column = i + 6
            if bias_dimension == 3:
                axis = ('x', 'y', 'z')[i]
                variable_name = 'bias_%s' % axis
                long_name = 'Evaluated calibrated accelerometer bias along %s-axis in GCRS' % axis.upper()
            else:
                variable_name = 'bias_component_%d' % (i + 1)
                long_name = 'Evaluated calibrated accelerometer bias component %d in GCRS' % (i + 1)
            file.write('    - %s:\n' % variable_name)
            file.write('        comment: %s column\n' % self.__ordinal(column))
            file.write('        coverage_content_type: modelResult\n')
            file.write('        long_name: %s\n' % long_name)
            file.write('        units: m/s2\n')
        file.write('# End of YAML header\n')

    def __write_primary_variables(self, file):
        file.write('    - gps_time:\n')
        file.write('        comment: 1st column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write('        long_name: Continuous seconds past 01-Jan-2000 11:59:47 UTC\n')
        file.write('        units: second\n')
        if self.__is_grace_fo():
            variable_name = 'GRACEFO_id'
            long_name = 'GRACE-FO satellite identifier'
            valid_range = 'C,D'
            meanings = (
                '          - C = GRACE-FO 1 (GRACE C)\n'
                '          - D = GRACE-FO 2 (GRACE D)\n'
            )
        else:
            variable_name = 'GRACE_id'
            long_name = 'GRACE satellite identifier'
            valid_range = 'A,B'
            meanings = (
                '          - A = GRACE 1 (GRACE A)\n'
                '          - B = GRACE 2 (GRACE B)\n'
            )
        file.write('    - %s:\n' % variable_name)
        file.write('        comment: 2nd column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write('        long_name: %s\n' % long_name)
        file.write('        units: char\n')
        file.write('        valid_range: %s\n' % valid_range)
        file.write('        value_meanings:\n')
        file.write(meanings)
        for column, axis in enumerate(('X', 'Y', 'Z'), start=3):
            file.write('    - lin_accl_%s:\n' % axis.lower())
            file.write('        comment: %s column\n' % self.__ordinal(column))
            file.write('        coverage_content_type: physicalMeasurement\n')
            file.write('        long_name: Linear acceleration along %s-axis in GCRS\n' % axis)
            file.write('        units: m/s2\n')

    def __merge_chunks(self, chunks):
        bias_dimension = max(item[2].shape[1] for item in chunks)
        times = []
        accelerations = []
        biases = []
        for time, acc, bias in chunks:
            if bias.shape[1] < bias_dimension:
                padding = np.full(
                    (bias.shape[0], bias_dimension - bias.shape[1]),
                    np.nan,
                    dtype=np.float64
                )
                bias = np.hstack((bias, padding))
            times.append(time)
            accelerations.append(acc)
            biases.append(bias)

        time = np.concatenate(times)
        acc = np.vstack(accelerations)
        bias = np.vstack(biases)
        order = np.argsort(time, kind='stable')
        time = time[order]
        acc = acc[order]
        bias = bias[order]

        reverse_unique = np.unique(time[::-1], return_index=True)[1]
        keep = np.sort(time.size - 1 - reverse_unique)
        return time[keep], acc[keep], bias[keep]

    @staticmethod
    def __format_bias(bias, epoch_number):
        if bias is None:
            return np.empty((epoch_number, 0), dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        if bias.ndim == 0:
            return np.full((epoch_number, 1), float(bias), dtype=np.float64)
        if bias.ndim == 1:
            if bias.size == epoch_number:
                return bias.reshape((-1, 1))
            return np.repeat(bias.reshape((1, -1)), epoch_number, axis=0)
        if bias.ndim == 2:
            if bias.shape[0] == epoch_number:
                return bias
            if bias.shape[1] == epoch_number:
                return bias.transpose()
        return np.repeat(bias.reshape((1, -1)), epoch_number, axis=0)

    def __standard_satellite(self, satellite):
        value = satellite.name if hasattr(satellite, 'name') else str(satellite)
        value = value.upper()
        if self.__is_grace_fo():
            mapping = {'A': 'C', 'B': 'D', 'C': 'C', 'D': 'D'}
        else:
            mapping = {'A': 'A', 'B': 'B'}
        if value not in mapping:
            raise ValueError('unsupported satellite for %s: %s' % (self.__mission, value))
        return mapping[value]

    def __is_grace_fo(self):
        return self.__mission is not None and 'GRACE_FO' in str(self.__mission).upper()

    @staticmethod
    def __day_numbers(time):
        return np.floor((time + 43200.0) / 86400.0).astype(np.int64)

    @classmethod
    def __day_label(cls, day_number):
        day_start = int(day_number) * 86400 - 43200
        return (cls._EPOCH + datetime.timedelta(seconds=day_start)).strftime('%Y-%m-%d')

    def __date_is_required(self, day):
        if self.__dateSpan is None:
            return True
        return self.__dateSpan[0] <= day <= self.__dateSpan[1]

    def __date_tag(self):
        if self.__dateSpan is None:
            return 'all'
        return '%s_%s' % (self.__dateSpan[0], self.__dateSpan[1])

    def __daily_directory(self, day):
        return os.path.join(self.__fileDir, day[0:4], day[0:7])

    def __configured_days(self):
        start = datetime.datetime.strptime(self.__dateSpan[0], '%Y-%m-%d')
        stop = datetime.datetime.strptime(self.__dateSpan[1], '%Y-%m-%d')
        while start <= stop:
            yield start.strftime('%Y-%m-%d')
            start += datetime.timedelta(days=1)

    def __cache_files(self):
        pattern = os.path.join(self.__cacheDir, self.__date_tag() + '_*_arc*.npz')
        return sorted(glob.glob(pattern))

    @staticmethod
    def __cache_satellite(filename):
        return os.path.basename(filename).rsplit('_arc', 1)[0].rsplit('_', 1)[1]

    @staticmethod
    def __sampling_interval(time):
        difference = np.diff(np.unique(time))
        difference = difference[difference > 0]
        if difference.size == 0:
            return 0.0
        return float(np.median(difference))

    @staticmethod
    def __format_time(value):
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) < 1.0e-9:
            return str(rounded)
        return '%.9f' % float(value)

    @staticmethod
    def __ordinal(number):
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        return '%d%s' % (number, suffix)

    def __check_configured(self):
        if self.__fileDir is None:
            raise ValueError('StandardACC should be configured before use')
