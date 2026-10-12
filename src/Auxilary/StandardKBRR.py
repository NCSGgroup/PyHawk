"""
@Description:
    Save aligned PyHawk raw O-C, pre-solution, post-solution, and monthly
    gravity-field KBRR series in one GRACE KBR1B-like ASCII file per UTC day.
"""

import datetime
import glob
import os

import numpy as np

from src.Auxilary.GeoMathKit import GeoMathKit


class StandardKBRR:
    _EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0)
    _EARTH_RADIUS = 6378136.3
    _STAGES = ('Raw', 'Pre', 'Post', 'Geofit')

    def __init__(self):
        self.__fileDir = None
        self.__cacheDir = None
        self.__dateSpan = None
        self.__mission = None
        self.__stage = None
        self.__productVersion = '01'

    def configure(self, filedir, date_span, stage, mission=None,
                  product_version='01'):
        stage = str(stage).capitalize()
        if stage not in self._STAGES:
            raise ValueError(
                "KBRR stage should be 'Raw', 'Pre', 'Post', or 'Geofit'"
            )
        self.__fileDir = os.path.abspath(str(filedir))
        self.__cacheDir = os.path.join(self.__fileDir, '.standard_kbrr_cache')
        self.__dateSpan = date_span
        self.__mission = mission
        self.__stage = stage
        self.__productVersion = product_version
        return self

    @staticmethod
    def output_directory(range_rate_directory):
        directory = os.path.abspath(str(range_rate_directory))
        return os.path.join(os.path.dirname(directory), 'StandardKBRR')

    def prepare(self):
        """Remove stale snapshots and combined daily files for this time span."""
        self.__check_configured()
        os.makedirs(self.__cacheDir, exist_ok=True)
        for filename in self.__cache_files(stage=None):
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
            for stage in self._STAGES:
                legacy_filename = 'Hawk_%s_KBRR1C_%s_%s.txt' % (
                    stage, day, self.__productVersion
                )
                legacy_paths = (
                    os.path.join(
                        self.__daily_directory(day), legacy_filename
                    ),
                    os.path.join(self.__fileDir, legacy_filename)
                )
                for legacy in legacy_paths:
                    if os.path.exists(legacy):
                        os.remove(legacy)
        return self

    def save_arc(self, gps_time, longitude, latitude, altitude,
                 range_rate, arc_no):
        """Atomically cache one arc for later daily aggregation."""
        self.__check_configured()
        arrays = [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in (
                gps_time, longitude, latitude, altitude, range_rate
            )
        ]
        names = ('gps_time', 'longitude', 'latitude', 'altitude', 'range_rate')
        sizes = [array.size for array in arrays]
        if len(set(sizes)) != 1:
            raise ValueError(
                'KBRR column size mismatch: ' +
                ', '.join('%s=%d' % item for item in zip(names, sizes))
            )
        valid = np.isfinite(arrays[0])
        arrays = [array[valid] for array in arrays]

        os.makedirs(self.__cacheDir, exist_ok=True)
        filename = '%s_%s_arc%06d.npz' % (
            self.__date_tag(), self.__stage, int(arc_no)
        )
        target = os.path.join(self.__cacheDir, filename)
        temporary = target + '.tmp.npz'
        np.savez_compressed(
            temporary,
            gps_time=arrays[0],
            longitude=arrays[1],
            latitude=arrays[2],
            altitude=arrays[3],
            range_rate=arrays[4]
        )
        os.replace(temporary, target)
        return target

    def finalize(self, remove_cache=True):
        """Align all KBRR components and write one KBRR file per day."""
        self.__check_configured()
        grouped = {
            stage: self.__group_cache(stage)
            for stage in self._STAGES
        }
        days = sorted(set().union(*(set(grouped[stage])
                                    for stage in self._STAGES)))

        output_files = []
        for day in days:
            missing = [stage for stage in self._STAGES
                       if day not in grouped[stage]]
            if missing:
                raise ValueError(
                    'KBRR day %s is missing %s data'
                    % (day, '/'.join(missing))
                )
            raw = self.__merge_chunks(grouped['Raw'][day])
            pre = self.__merge_chunks(grouped['Pre'][day])
            post = self.__merge_chunks(grouped['Post'][day])
            geofit = self.__merge_chunks(grouped['Geofit'][day])
            columns = self.__combine_stages(raw, pre, post, geofit)
            directory = self.__daily_directory(day)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, self.__output_filename(day))
            self.__write_file(path, columns)
            output_files.append(path)

        if remove_cache:
            self.cleanup()
        return output_files

    def cleanup(self):
        """Remove temporary KBRR arc snapshots without touching products."""
        self.__check_configured()
        for filename in self.__cache_files(stage=None):
            os.remove(filename)
        if os.path.isdir(self.__cacheDir) and not os.listdir(self.__cacheDir):
            os.rmdir(self.__cacheDir)
        return self

    def discard_arcs(self, arc_numbers, stages=None):
        """Remove rejected arc snapshots before daily aggregation."""
        self.__check_configured()
        if stages is None:
            stages = self._STAGES
        stages = tuple(str(stage).capitalize() for stage in stages)
        invalid = [stage for stage in stages if stage not in self._STAGES]
        if invalid:
            raise ValueError(
                'Unsupported KBRR stages: %s' % ', '.join(invalid)
            )

        removed = []
        for arc_number in arc_numbers:
            for stage in stages:
                filename = os.path.join(
                    self.__cacheDir,
                    '%s_%s_arc%06d.npz' % (
                        self.__date_tag(), stage, int(arc_number)
                    )
                )
                if os.path.exists(filename):
                    os.remove(filename)
                    removed.append(filename)
        return removed

    def __group_cache(self, stage):
        grouped = {}
        for filename in self.__cache_files(stage=stage):
            with np.load(filename, allow_pickle=False) as data:
                columns = tuple(
                    np.asarray(data[name], dtype=np.float64)
                    for name in (
                        'gps_time', 'longitude', 'latitude', 'altitude',
                        'range_rate'
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
        return grouped

    def __write_file(self, path, columns):
        (time, longitude, latitude, altitude, pre_rate, post_rate,
         raw_oc_rate, geofit_rate) = columns
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='ascii', newline='\n') as file:
            self.__write_header(file, time)
            for values in zip(
                    time, longitude, latitude, altitude, pre_rate, post_rate,
                    raw_oc_rate, geofit_rate):
                file.write(
                    '%s %.12f %.12f %.6f %.15e %.15e %.15e %.15e\n' % (
                        self.__format_time(values[0]),
                        values[1], values[2], values[3], values[4], values[5],
                        values[6], values[7]
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
        file.write('    source: PyHawk raw O-C, pre-, post-solution, and monthly gravity-field KBRR series\n')
        file.write('    summary: Raw O-C, pre- and post-solution residuals, and the monthly global gravity contribution\n')
        file.write('    time_coverage_start: %s\n' % start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    time_coverage_stop: %s\n' % stop.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4])
        file.write('    title: %s Hawk KBRR Level-1C Data\n' % mission_name)
        file.write('  variables:\n')
        file.write('    - gps_time:\n')
        file.write('        comment: 1st column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write('        long_name: Continuous seconds past 01-Jan-2000 11:59:47 UTC\n')
        file.write('        units: second\n')
        file.write('    - longitude:\n')
        file.write('        comment: 2nd column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write("        long_name: Longitude of the twin satellites' midpoint\n")
        file.write('        units: degree_east\n')
        file.write('    - latitude:\n')
        file.write('        comment: 3rd column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write("        long_name: Geocentric latitude of the twin satellites' midpoint\n")
        file.write('        units: degree_north\n')
        file.write('    - altitude:\n')
        file.write('        comment: 4th column\n')
        file.write('        coverage_content_type: referenceInformation\n')
        file.write("        long_name: Geocentric altitude of the twin satellites' midpoint\n")
        file.write('        units: m\n')
        file.write('    - pre_range_rate:\n')
        file.write('        comment: 5th column\n')
        file.write('        coverage_content_type: modelResult\n')
        file.write(
            '        long_name: Pre-solution inter-satellite range-rate '
            'O-C after OneCPR-assisted abnormal-epoch screening; no trend '
            'or CPR term is subtracted\n'
        )
        file.write('        units: m/s\n')
        file.write('    - post_range_rate:\n')
        file.write('        comment: 6th column\n')
        file.write('        coverage_content_type: modelResult\n')
        file.write('        long_name: Post-solution inter-satellite range-rate residual\n')
        file.write('        units: m/s\n')
        file.write('    - raw_oc_range_rate:\n')
        file.write('        comment: 7th column\n')
        file.write('        coverage_content_type: modelResult\n')
        file.write(
            '        long_name: Retained inter-satellite range-rate '
            'observation-minus-computation residual; identical to the '
            'pre-solution input before gravity-field adjustment\n'
        )
        file.write('        units: m/s\n')
        file.write('    - geofit_range_rate:\n')
        file.write('        comment: 8th column\n')
        file.write('        coverage_content_type: modelResult\n')
        file.write(
            '        long_name: Monthly global gravity contribution reconstructed '
            'from the final spherical-harmonic solution in the KBRR observation domain\n'
        )
        file.write('        units: m/s\n')
        file.write('# End of YAML header\n')

    @staticmethod
    def __merge_chunks(chunks):
        columns = [
            np.concatenate([item[index] for item in chunks])
            for index in range(5)
        ]
        time = columns[0]
        order = np.argsort(time, kind='stable')
        columns = [column[order] for column in columns]
        time = columns[0]
        reverse_unique = np.unique(time[::-1], return_index=True)[1]
        keep = np.sort(time.size - 1 - reverse_unique)
        return tuple(column[keep] for column in columns)

    @staticmethod
    def __combine_stages(raw, pre, post, geofit):
        raw_key = np.round(raw[0], decimals=9)
        pre_key = np.round(pre[0], decimals=9)
        post_key = np.round(post[0], decimals=9)
        geofit_key = np.round(geofit[0], decimals=9)
        if not (np.array_equal(raw_key, pre_key)
                and np.array_equal(raw_key, post_key)
                and np.array_equal(raw_key, geofit_key)):
            common = np.intersect1d(
                np.intersect1d(
                    np.intersect1d(raw_key, pre_key, assume_unique=True),
                    post_key,
                    assume_unique=True
                ),
                geofit_key,
                assume_unique=True
            )
            raise ValueError(
                'KBRR epoch mismatch: raw=%d, pre=%d, post=%d, '
                'geofit=%d, common=%d'
                % (raw_key.size, pre_key.size, post_key.size,
                   geofit_key.size, common.size)
            )
        return (
            raw[0],
            raw[1],
            raw[2],
            raw[3],
            pre[4],
            post[4],
            raw[4],
            geofit[4]
        )

    @classmethod
    def midpoint_coordinates(cls, gps_time, orbit_a, orbit_b, frame):
        """Compute LGD-compatible midpoint longitude, latitude and altitude."""
        time = np.asarray(gps_time, dtype=np.float64).reshape(-1)
        orbit_a = np.asarray(orbit_a, dtype=np.float64)
        orbit_b = np.asarray(orbit_b, dtype=np.float64)
        if orbit_a.ndim != 2 or orbit_a.shape[1] < 4:
            raise ValueError('orbit_a should contain gps_time and three position columns')
        if orbit_b.ndim != 2 or orbit_b.shape[1] < 4:
            raise ValueError('orbit_b should contain gps_time and three position columns')

        pos_a = cls.__interpolate_position(orbit_a[:, 0], orbit_a[:, 1:4], time)
        pos_b = cls.__interpolate_position(orbit_b[:, 0], orbit_b[:, 1:4], time)
        longitude = np.zeros(time.size, dtype=np.float64)
        latitude = np.zeros(time.size, dtype=np.float64)
        altitude = np.zeros(time.size, dtype=np.float64)

        current_frame = frame
        for index, epoch in enumerate(time):
            current_frame = current_frame.setTime(float(epoch))
            rotation = current_frame.getRotationMatrix
            pos_a_itrs = current_frame.PosGCRS2ITRS(pos_a[index], rotation)
            pos_b_itrs = current_frame.PosGCRS2ITRS(pos_b[index], rotation)
            midpoint = 0.5 * (pos_a_itrs + pos_b_itrs)
            lon, lat, radius = GeoMathKit.CalcPolarAngles(midpoint)
            longitude[index] = np.degrees(lon)
            latitude[index] = np.degrees(lat)
            altitude[index] = radius - cls._EARTH_RADIUS
            if longitude[index] >= 180.0:
                longitude[index] -= 360.0

        return longitude, latitude, altitude

    @staticmethod
    def __interpolate_position(reference_time, reference_position, target_time):
        position = np.zeros((target_time.size, 3), dtype=np.float64)
        for axis in range(3):
            position[:, axis] = np.interp(
                target_time,
                reference_time,
                reference_position[:, axis]
            )
        return position

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

    def __cache_files(self, stage=None):
        stage_tag = '*' if stage is None else stage
        pattern = os.path.join(
            self.__cacheDir,
            '%s_%s_arc*.npz' % (self.__date_tag(), stage_tag)
        )
        return sorted(glob.glob(pattern))

    def __output_filename(self, day):
        return 'Hawk_KBRR1C_%s_%s.txt' % (
            day, self.__productVersion
        )

    def __is_grace_fo(self):
        return self.__mission is not None and 'GRACE_FO' in str(self.__mission).upper()

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

    def __check_configured(self):
        if self.__fileDir is None:
            raise ValueError('StandardKBRR should be configured before use')
