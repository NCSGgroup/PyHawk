from src.Auxilary.GeoMathKit import GeoMathKit
import src.Preference.EnumType as EnumType
from src.Interface.ArcOperation import ArcData
import h5py
import pathlib
import pygmt
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import numpy as np
import json
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig

# fig = pygmt.Figure()
# fig.coast(region="g", frame=True, shorelines=1)
# fig.show()
#
# fig = pygmt.Figure()
# fig.coast(projection="H10c", region="g", frame=True, land="gray")
# fig.show()


def getKBRRGMT():
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict

    frame_config = FrameConfig()
    frame = json.load(open('../setting/Calibrate/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    rd = ArcData(interfaceConfig=interface_config)
    date_span = ['2009-05-01', '2009-05-31']

    fr = Frame(EOP().configure(frameConfig=frame_config).load()).configure(frameConfig=frame_config)

    Lon = []
    Lat = []
    Prefit = []
    Postfit = []
    for i in range(0, 120):
        GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        res_dir = pathlib.Path('../temp/RangeRate').joinpath(date_span[0] + '_' + date_span[1])
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        t1 = h5['pre_t'][()]
        t2 = h5['post_t'][()]
        # Prefit.extend(h5['pre_residual'][()])
        Postfit.extend(h5['post_residual'][()])
        # postfit = h5['post_residual'][()]

        new_index = [list(GNVTime).index(x) for x in t2]
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

    np.savez('PostKBRR.npz', x=Lon, y=Lat, z=Postfit)


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
    # res1 = np.load("BeforeRes.npz", allow_pickle=True)
    res1 = h5py.File("../result/20250728/GRFO/acc-1.5h/pre-2019-01-01_2019-01-31.hdf5", "r")
    x1 = np.array(res1['lon'][:])
    y1 = np.array(res1['lat'][:])
    z1 = res1['prefit'][:] * 1e6

    # res = np.load("AfterRes.npz", allow_pickle=True)
    res = h5py.File("../result/IGG-2019-01-01_2019-01-31.hdf5", "r")
    x = np.array(res['lon'][:])
    y = np.array(res['lat'][:])
    z = res['prefit'][:] * 1e6

    grid1 = pygmt.xyz2grd(x=x1, y=y1, z=z1, spacing=(1, 1), region="d", duplicate='r')
    grid1 = pygmt.grdfill(grid=grid1, mode="n")

    grid = pygmt.xyz2grd(x=x, y=y, z=z, spacing=(1, 1), region="d", duplicate='r')
    grid = pygmt.grdfill(grid=grid, mode="n")

    grid2 = pygmt.xyz2grd(x=x, y=y, z=z, spacing=(1, 1), region="d", duplicate='r')
    grid2 = pygmt.grdfill(grid=grid, mode="n")
    grid2.data = grid.data - grid1.data

    fig = pygmt.Figure()
    # pygmt.makecpt(cmap=cpt, series=[0, 10, 2]) # color bar range

    with fig.subplot(
            nrows=2, ncols=1, subsize=("12c", "6c"),
            frame="afg",
            # sharey=True,
            # sharex=True,
            autolabel="+JTL+o0.5/0.5",
            # margins="0.5c",
            margins=["0.5c", "0.8c"],
    ):
        with fig.set_panel(panel=0):
            pygmt.makecpt(series=[0, 1, 0.1], background="red")  # color bar range
            fig.grdimage(grid=grid1, frame="f")
            fig.coast(region=[-180, 180, -90, 90], shorelines="1/0.5p", projection="Q12c")
            fig.basemap(projection="Q12c", frame=["f"])
            fig.colorbar(frame=["y"])

        with fig.set_panel(panel=1):
            pygmt.makecpt(series=[0, 1, 0.1], background="red")  # color bar range
            fig.basemap(region="d", frame=["f"])
            fig.grdimage(grid=grid)
            fig.coast(shorelines="1/0.5p", projection="Q12c")
            fig.colorbar(frame=["y"])

        # with fig.set_panel(panel=2):
        #     pygmt.makecpt(series=[0, 10, 1], background="red")  # color bar range
        #     fig.basemap(region="d", frame=["f", "+tAOD-CRA"])
        #     fig.grdimage(grid=grid2)
        #     fig.coast(shorelines="1/0.5p", projection="Q12c")
        #     fig.colorbar(frame=["y+lum/s"])
    # fig.savefig("output.png")
    fig.show()


def draw2():
    # res1 = np.load("BeforeRes.npz", allow_pickle=True)
    res1 = h5py.File("../result/2009-05-01_2009-05-31-pre.hdf5", "r")
    x1 = np.array(res1['lon'][:])
    y1 = np.array(res1['lat'][:])
    z1 = res1['postfit'][:] * 1e7

    # res = np.load("AfterRes.npz", allow_pickle=True)
    res = h5py.File("../result/2009-05-01_2009-05-31-post.hdf5", "r")
    x = np.array(res['lon'])
    y = np.array(res['lat'])
    z = res['postfit'][:] * 1e7 - z1

    grid = pygmt.xyz2grd(x=x, y=y, z=z, spacing=(1, 1), region="d", duplicate='r')
    grid = pygmt.grdfill(grid=grid, mode="n")

    fig = pygmt.Figure()
    cpt = pygmt.makecpt(series=[0, 10, 2], background="red") # color bar range

    # pygmt.makecpt(cmap=cpt, series=[0, 10, 2]) # color bar range
    with fig.subplot(
            nrows=1, ncols=1, subsize=("12c", "6c"),
            frame="afg",
            sharey=True,
            sharex=True,
            # autolabel="+JTL+o0.3/0.5",
            # margins="0.5c",
            margins=["0.5c", "0.5c"],
    ):
        with fig.set_panel(panel=0):
            fig.basemap(region='d', projection="Q12c")
            fig.grdimage(grid=grid, cmap=cpt, frame=True)
            fig.coast(shorelines="1/0.5p", projection="Q12c")
            fig.colorbar(cmap=cpt, frame=["y+lm/s"])
    fig.show()


if __name__ == '__main__':
    # getKBRRGMT()
    draw()
    # draw2()
    # readDat()
