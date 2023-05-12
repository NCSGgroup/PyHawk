from src.Auxilary.GeoMathKit import GeoMathKit
import src.Preference.EnumType as EnumType
from src.Interface.ArcOperation import ArcData
import h5py
import pathlib
import src.Preference.backups.Pre_AdjustOrbit as Pre_AdjustOrbit
import pygmt
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import numpy as np

# fig = pygmt.Figure()
# fig.coast(region="g", frame=True, shorelines=1)
# fig.show()
#
# fig = pygmt.Figure()
# fig.coast(projection="H10c", region="g", frame=True, land="gray")
# fig.show()


def getKBRRGMT():
    fr = Frame(EOP().load())
    date_span = ['2008-01-01', '2008-01-31']
    rd = ArcData(date_span=date_span)
    Lon = []
    Lat = []
    Prefit = []
    Postfit = []
    for i in range(0, 243):
        GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp).joinpath(date_span[0] + '_' + date_span[1])
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        t1 = h5['pre_t'][()]
        t2 = h5['post_t'][()]
        Prefit.extend(h5['pre_residual'][()])
        # Postfit.extend(h5['post_residual'][()])
        # postfit = h5['post_residual'][()]

        new_index = [list(GNVTime).index(x) for x in t1]
        new_Time = GNVTime[new_index]
        new_GNVR = GNVR[new_index]
        # new_KBRR = h5['pre_residual'][()][new_index]
        for i in range(len(new_Time)):
            time = new_Time[i]
            pos = new_GNVR[i]
            fr = fr.setTime(float(time))
            itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
            lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
            Lon.append(lon)
            Lat.append(lat)

    np.savez('PreKBRR.npz', x=Lon, y=Lat, z=Prefit)


def readDat():
    path = 'percent_6000premq.dat'
    xyz = []
    f = open(path, 'r')

    r = f.readlines()
    for line in r:
        line = line.split()
        xyz.append(list(map(float, line)))
    xyz = np.array(xyz)
    fig = pygmt.Figure()
    grid = pygmt.xyz2grd(x=xyz[0], y=xyz[1], z=xyz[2], spacing="1", projection="x15c/10c", region=[0, 2600, 100, 1500])
    fig.grdimage(grid=grid, region=[0, 2600, 100, 1500], projection="x15c/10c", frame=["af", "WSne"])
    fig.show()


def draw():
    res = np.load("PostKBRR.npz", allow_pickle=True)
    x = np.degrees(np.array(res['x']))
    y = np.degrees(np.array(res['y']))
    z = res['z'] * 1e7
    # data = pygmt.datasets.load_sample_data(name="japan_quakes")
    # data = data[["longitude", "latitude", "depth_km"]]
    # region = [130, 152.5, 32.5, 52.5]
    # spacing = "150m"
    # df = pygmt.blockmean(data=data, region=region, spacing=spacing)
    # grd = pygmt.xyz2grd(data=df, region=region, spacing=spacing)

    grid = pygmt.xyz2grd(x=x, y=y, z=z, spacing=(1, 1), region="d", duplicate='r')
    fig = pygmt.Figure()
    pygmt.makecpt(series=[0, 7, 1])
    # fig.basemap(frame="afg")
    fig.grdimage(grid=grid, region="d", projection="Q12c", frame="afg")
    fig.coast(shorelines="1/0.5p", projection="Q12c")
    fig.colorbar()
    fig.show()


if __name__ == '__main__':
    # getKBRRGMT()
    draw()
    # readDat()
