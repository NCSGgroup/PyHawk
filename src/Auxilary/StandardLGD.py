"""
@Description:
    Save PyHawk LGD results in a KBRR-like Level-1C ASCII format. Arc
    snapshots are merged into one file per UTC day.
"""

import datetime
import glob
import os

import numpy as np


class StandardLGD:
    _EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0)
    _EARTH_RADIUS = 6378136.3

    def __init__(self):
        self.__fileDir = None
        self.__cacheDir = None
        self.__dateSpan = None
        self.__mission = None
        self.__productVersion = '01'

    def configure(self, filedir, date_span, mission=None, product_version='01'):
        self.__fileDir = os.path.abspath(str(filedir))
        self.__cacheDir = os.path.join(self.__fileDir, '.standard_lgd_cache')
        self.__dateSpan = date_span
        self.__mission = mission
        self.__productVersion = product_version
        return self

    def prepare(self):
        """Remove stale snapshots and daily files for this time span."""
        self.__check_configured()
        os.makedirs(self.__cacheDir, exist_ok=True)
        for filename in self.__cache_files():
            os.remove(filename)
        for day in self.__configured_days():
            output_filename = self.__output_filename(day)
            paths = (
                os.path.join(self.__daily_directory(day), output_filename),
                os.path.join(self.__fileDir, output_filename)
            )
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)
        return self

    def save_arc(self, gps_time, longitude, latitude, altitude,
                 lgd_full, lgd_band, arc_no):
        """Atomically cache one LGD arc for later daily aggregation."""
        self.__check_configured()
        arrays = [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in (
                gps_time, longitude, latitude, altitude, lgd_full, lgd_band
            )
        ]
        names = ('gps_time', 'longitude', 'latitude', 'altitude', 'LGD_full', 'LGD_band')
        sizes = [array.size for array in arrays]
        if len(set(sizes)) != 1:
            raise ValueError(
                'LGD column size mismatch: ' +
                ', '.join('%s=%d' % item for item in zip(names, sizes))
            )

        valid = np.isfinite(arrays[0])
        arrays = [array[valid] for array in arrays]
        os.makedirs(self.__cacheDir, exist_ok=True)
        target = os.path.join(
            self.__cacheDir,
            '%s_arc%06d.npz' % (self.__date_tag(), int(arc_no))
        )
        temporary = target + '.tmp.npz'
        np.savez_compressed(
            temporary,
            gps_time=arrays[0],
            longitude=arrays[1],
            latitude=arrays[2],
            altitude=arrays[3],
            lgd_full=arrays[4],
            lgd_band=arrays[5]
        )
        os.replace(temporary, target)
        return target

    def finalize(self, remove_cache=True):
        """Write one standardized LGD file for every covered day."""
        self.__check_configured()
        grouped = {}
        cache_files = self.__cache_files()
        for filename in cache_files:
            with np.load(filename, allow_pickle=False) as data:
                columns = tuple(
                    np.asarray(data[name], dtype=np.float64)
                    for name in (
                        'gps_time', 'longitude', 'latitude', 'altitude',
                        'lgd_full', 'lgd_band'
                    )
                )
            day_numbers = self.__day_numbers(columns[0])
            for day_number in np.unique(day_numbers):
                day = self.__day_label(day_number)
                if not self.__date_is_required(day):
                    continue
                index = day_numbers == day_number
                grouped.setdefault(day, []).append(
                    tuple(column[index] for column in columns)
                )

        output_files = []
        for day in sorted(grouped):
            columns = self.__merge_chunks(grouped[day])
            directory = self.__daily_directory(day)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, self.__output_filename(day))
            self.__write_file(path, columns)
            output_files.append(path)

        if remove_cache:
            for filename in cache_files:
                os.remove(filename)
        return output_files

    @classmethod
    def midpoint_altitude(cls, pos_a_itrs, pos_b_itrs):
        pos_a = np.asarray(pos_a_itrs, dtype=np.float64)
        pos_b = np.asarray(pos_b_itrs, dtype=np.float64)
        if pos_a.shape != pos_b.shape or pos_a.ndim != 2 or pos_a.shape[1] != 3:
            raise ValueError('ITRS satellite positions should both have shape (epoch_number, 3)')
        midpoint = 0.5 * (pos_a + pos_b)
        return np.linalg.norm(midpoint, axis=1) - cls._EARTH_RADIUS

    def __write_file(self, path, columns):
        time, longitude, latitude, altitude, lgd_full, lgd_band = columns
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='ascii', newline='\n') as file:
            self.__write_header(file, time)
            for values in zip(time, longitude, latitude, altitude, lgd_full, lgd_band):
                file.write(
                    '%s %.12f %.12f %.6f %.15e %.15e\n' % (
                        self.__format_time(values[0]),
                        values[1], values[2], values[3], values[4], values[5]
                    )
                )
        os.replace(temporary, path)

    def __write_header(self, file, time):
        created = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        start = self._EPOCH + datetime.timedelta(seconds=float(time[0]))
        stop = self._EPOCH + datetime.timedelta(seconds=float(time[-1]))
        mission_name = 'GRACE Follow-On' if self.__is_grace_fo() else 'GRACE'

        file.write('header:\n')
        file.write('  dimensions:\n')
        file.write('    num_records: %d\n' % time.size)
        file.write('  global_attributes:\n')
        file.write('    creator: Yi Wu(wu_yi@hust.edu.cn); Fan Yang(fany@plan.aau.dk)\n')
        file.write('    creator_institution: NCSG(https://github.com/NCSGgroup)\n')
        file.write('    date_created: %s\n' % created)
        file.write('    instrument: KBR\n')
        file.write('    source: PyHawk line-of-sight gravity difference processing\n')
        file.write('    summary: Full-spectrum and bandpass LGD at the twin-satellite midpoint\n')
        file.write('    time_coverage_start: %s\n' % start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    time_coverage_stop: %s\n' % stop.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    title: %s Hawk LGD Level-1C Data\n' % mission_name)
        file.write('  variables:\n')
        variables = (
            ('gps_time', '1st', 'referenceInformation',
             'Continuous seconds past 01-Jan-2000 11:59:47 UTC', 'second'),
            ('longitude', '2nd', 'referenceInformation',
             "Longitude of the twin satellites' midpoint", 'degree_east'),
            ('latitude', '3rd', 'referenceInformation',
             "Geocentric latitude of the twin satellites' midpoint", 'degree_north'),
            ('altitude', '4th', 'referenceInformation',
             "Geocentric altitude of the twin satellites' midpoint", 'm'),
            ('LGD_full', '5th', 'modelResult',
             'Unfiltered full-spectrum line-of-sight gravity difference', 'm/s2'),
            ('LGD_band', '6th', 'modelResult',
             'Bandpass-retained line-of-sight gravity difference', 'm/s2'),
        )
        for name, column, content_type, long_name, units in variables:
            file.write('    - %s:\n' % name)
            file.write('        comment: %s column\n' % column)
            file.write('        coverage_content_type: %s\n' % content_type)
            file.write('        long_name: %s\n' % long_name)
            file.write('        units: %s\n' % units)
        file.write('# End of YAML header\n')

    @staticmethod
    def __merge_chunks(chunks):
        columns = tuple(
            np.concatenate([chunk[index] for chunk in chunks])
            for index in range(6)
        )
        order = np.argsort(columns[0], kind='stable')
        columns = tuple(column[order] for column in columns)
        reverse_unique = np.unique(columns[0][::-1], return_index=True)[1]
        keep = np.sort(columns[0].size - 1 - reverse_unique)
        return tuple(column[keep] for column in columns)

    @staticmethod
    def __day_numbers(time):
        return np.floor((time + 43200.0) / 86400.0).astype(np.int64)

    @classmethod
    def __day_label(cls, day_number):
        day_start = int(day_number) * 86400 - 43200
        return (cls._EPOCH + datetime.timedelta(seconds=day_start)).strftime('%Y-%m-%d')

    def __date_is_required(self, day):
        return self.__dateSpan[0] <= day <= self.__dateSpan[1]

    def __configured_days(self):
        start = datetime.datetime.strptime(self.__dateSpan[0], '%Y-%m-%d')
        stop = datetime.datetime.strptime(self.__dateSpan[1], '%Y-%m-%d')
        while start <= stop:
            yield start.strftime('%Y-%m-%d')
            start += datetime.timedelta(days=1)

    def __date_tag(self):
        return '%s_%s' % (self.__dateSpan[0], self.__dateSpan[1])

    def __daily_directory(self, day):
        return os.path.join(self.__fileDir, day[0:4], day[0:7])

    def __cache_files(self):
        return sorted(glob.glob(os.path.join(
            self.__cacheDir, self.__date_tag() + '_arc*.npz'
        )))

    def __output_filename(self, day):
        return 'Hawk_LGD1C_%s_%s.txt' % (day, self.__productVersion)

    def __is_grace_fo(self):
        return self.__mission is not None and 'GRACE_FO' in str(self.__mission).upper()

    @staticmethod
    def __sampling_interval(time):
        difference = np.diff(np.unique(time))
        difference = difference[difference > 0]
        return 0.0 if difference.size == 0 else float(np.median(difference))

    @staticmethod
    def __format_time(value):
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) < 1.0e-9:
            return str(rounded)
        return '%.9f' % float(value)

    def __check_configured(self):
        if self.__fileDir is None:
            raise ValueError('StandardLGD should be configured before use')
