import numpy as np
from src.Preference.EnumType import FieldType
from src.Preference.Pre_Constants import GeoConstants


class ConvertSHC:
    def __init__(self, cqlm, sqlm, field_type: FieldType, ln):
        self.cqlm = cqlm
        self.sqlm = sqlm
        self.nmax = np.shape(cqlm)[1] - 1

        self.field_type = field_type

        self.ln = ln

    def convert_to(self, fieldtype: FieldType):
        self._convertToGeoid()
        self._convertFromGeoidTo(fieldtype)
        return self

    def _convertToGeoid(self):
        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth

        if self.field_type is FieldType.geoid:
            return self

        elif self.field_type is FieldType.EWH:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 * density_water / (
                    radius_e * density_earth)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.field_type = FieldType.geoid
            return self

        elif self.field_type is FieldType.density:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 / (radius_e * density_earth)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.cqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.field_type = FieldType.geoid
            return self

        else:
            # TODO how to raise an error standardly?
            pass

    def _convertFromGeoidTo(self, fieldtype: FieldType):
        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth

        if fieldtype is FieldType.geoid:
            return self

        elif fieldtype is FieldType.EWH:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / (
                    3 * density_water)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.type = FieldType.EWH
            return self

        elif fieldtype is FieldType.density:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / 3

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.type = FieldType.density
            return self

        else:
            # TODO how to raise an error standardly?
            pass
