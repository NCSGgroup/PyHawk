#include<iostream>
#include <iomanip>
#include<math.h>
#include"GravHar.h"

using namespace std;

int main() {
    bool reload;
    double Radius = 6378136.3e0;
    //double xyz[3] = { 100000 + Radius, 100000 + Radius, 200000 + Radius };
    double xyz[3] = { -4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06 };
    double acc[3] = { 0e0 };
    double DU2XYZ[9] = { 0e0 };
    const int Nmax = 100;
    int Nli[2] = { 2,100 };
    double CPOT[(Nmax + 2) * (Nmax + 1) / 2], SPOT[(Nmax + 2) * (Nmax + 1) / 2];
    double dxdc[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 }, dydc[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 }, dzdc[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 };
    double dxds[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 }, dyds[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 }, dzds[(Nmax + 2) * (Nmax + 1) / 2] = { 0.e0 };

    double ts[3] = { -4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06 };

    double r_sqr = pow(ts[0], 2) + pow(ts[1], 2) + pow(ts[2], 2);

    cout << setprecision(16) << r_sqr << " " << Radius << endl;
    // reload = GravHar(CPOT, SPOT, 6);
    //cout << setprecision(16) << GM << " " << Radius << endl;
    // reload = GravHar(CPOT, SPOT, 6);

    for (int i = 0; i < (Nmax + 2) * (Nmax + 1) / 2; i++) {
        CPOT[i] = (double)i;
        SPOT[i] = (double)i;
    }
    setPar(398600.44150e9, 6378136.3e0);
    cout << setprecision(16) << r_sqr << " " << Radius << endl;

    preprocess(Nmax);
    calVW(xyz, Nmax);
    cout << setprecision(16) << r_sqr << " " << Radius << endl;
    //DWORD start_time = GetTickCount();
    for (int i = 0; i < 1; i++) {
        getAcc(Nmax, CPOT, SPOT, acc);
	cout << setprecision(16) << r_sqr << " 1 " << Radius << endl;
        getDu2xyz(100, CPOT, SPOT, DU2XYZ);
	cout << setprecision(16) << r_sqr << " 2 " << Radius << endl;
        getDu2CS(2,100, dxdc, dydc, dzdc, dxds, dyds, dzds);
	cout << setprecision(16) << r_sqr << " 3 " << Radius << endl;
        double Upot = getPotential(Nmax, CPOT, SPOT);
    }
    //DWORD end_time = GetTickCount();

    //cout << "The run time is:" << (end_time - start_time) << "ms!" << endl;
    
    //cout << f3meq1[162] << endl;
    //cout << grdtcn11[2] << endl;
    //cout << grdtcnm6[124] << endl;
    //cout << V[0][0] << endl;
    cout << setprecision(16) << r_sqr << " " << Radius << endl;
    cout << setprecision(20)<<acc[0] << endl;
    cout << DU2XYZ[4] << endl;
    cout << dxdc[1231] << endl;
    cout << dydc[2341] << endl;
    cout << dzdc[3223] << endl;
    cout << dxds[423] << endl;
    cout << dyds[1823] << endl;
    cout << dzds[4923] << endl;
    //cout << Upot << endl;




    return 0;
};
