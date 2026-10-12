#include "GravHar.h"
#include<iostream>
#include <iomanip>
#include<math.h>

using namespace std;

static double GM = 398600.44150e9, Radius = 6378136.3e0;
static const int Ns = 360;

static double f1meq0[Ns + 1] = { 0e0 }, f2meq0[Ns + 1] = { 0e0 }, f1meq1[Ns + 1] = { 0e0 }, f2meq1[Ns + 1] = { 0e0 }, f3meq1[Ns + 1] = { 0e0 };
static double grdtcn1[Ns + 1] = { 0e0 }, grdtcn2[Ns + 1] = { 0e0 }, grdtcn3[Ns + 1] = { 0e0 }
, grdtcn4[Ns + 1] = { 0e0 }, grdtcn5[Ns + 1] = { 0e0 };
static double grdtcn6[Ns + 1] = { 0e0 }, grdtcn7[Ns + 1] = { 0e0 }
, grdtcn8[Ns + 1] = { 0e0 }, grdtcn9[Ns + 1] = { 0e0 }, grdtcn10[Ns + 1] = { 0e0 };
static double grdtcn11[Ns + 1] = { 0e0 }, grdtcn12[Ns + 1] = { 0e0 };
static double vwmeq1[Ns + 1] = { 0e0 }, vwn1[Ns + 1] = { 0e0 }, vwn2[Ns + 1] = { 0e0 };

static double f1mgt1[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, f2mgt1[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, f3mgt1[(Ns + 1) * (Ns + 2) / 2] = { 0e0 };
static double grdtcnm1[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, grdtcnm2[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, grdtcnm3[(Ns + 1) * (Ns + 2) / 2] = { 0e0 };
static double grdtcnm4[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, grdtcnm5[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, grdtcnm6[(Ns + 1) * (Ns + 2) / 2] = { 0e0 };
static double vwnm1[(Ns + 1) * (Ns + 2) / 2] = { 0e0 }, vwnm2[(Ns + 1) * (Ns + 2) / 2] = { 0e0 };

static double V[Ns + 1][Ns + 1] = { 0e0 }, W[Ns + 1][Ns + 1] = { 0e0 };

int getNlimit() {
    return Ns;
};

void setPar(double GM1, double R1 ) {
    GM = GM1;
    Radius = R1;
}

void preprocess(int const Nmax) {


    for (int in = 0; in <= Nmax + 2; in++) {

        double n = (double)in;

        f1meq0[in] = sqrt((2 * n + 1) * (n + 2) * (n + 1) / (4 * n + 6));
        f2meq0[in] = (n + 1) * sqrt((2 * n + 1) / (2 * n + 3));
        f1meq1[in] = sqrt((2 * n + 1) * (n + 2) * (n + 3) / (2 * n + 3));
        f2meq1[in] = sqrt((2 * n + 1) * (n + 1) * n * 2 / (2 * n + 3));
        f3meq1[in] = sqrt((2 * n + 1) * (n + 2) * n / (2 * n + 3));

        // # for tensor coefficient
        grdtcn1[in] = sqrt((2 * n + 1) * (n + 1) * (n + 2) * (n + 3) * (n + 4) / (4 * n + 10));
        grdtcn2[in] = (n + 1) * sqrt((2 * n + 1) * (n + 2) * (n + 3) / (4 * n + 10));
        grdtcn3[in] = (n + 1) * (n + 2) * sqrt((2 * n + 1) / (2 * n + 5));
        grdtcn4[in] = sqrt((2 * n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) / (2 * n + 5));
        grdtcn5[in] = sqrt((2 * n + 1) * n * (n + 1) * (n + 2) * (n + 3) / (2 * n + 5));
        grdtcn6[in] = sqrt(n * (n + 2) * (n + 3) * (n + 4) * (2 * n + 1) / (2 * n + 5));
        grdtcn7[in] = (n + 2) * sqrt(2 * (2 * n + 1) * (n + 1) * n / (2 * n + 5));
        grdtcn8[in] = sqrt((n + 3) * (n + 4) * (n + 5) * (n + 6) * (2 * n + 1) / (2 * n + 5));
        grdtcn9[in] = 2 * sqrt((n - 1) * n * (n + 3) * (n + 4) * (2 * n + 1) / (2 * n + 5));
        grdtcn10[in] = sqrt((n - 1) * n * (n + 1) * (n + 2) * (4 * n + 2) / (2 * n + 5));
        grdtcn11[in] = sqrt((n - 1) * (n + 3) * (n + 4) * (n + 5) * (2 * n + 1) / (2 * n + 5));
        grdtcn12[in] = sqrt((n + 3) * (n - 1) * n * (n + 1) * (2 * n + 1) / (2 * n + 5));

        // # coefficient for V, W
        vwmeq1[in] = sqrt((2 * n + 1) / (2 * n));
        vwn1[in] = sqrt((2 * n + 1) / (pow(n, 2))) * sqrt((2 * n - 1));
        vwn2[in] = sqrt((2 * n + 1) / (pow(n, 2))) * (n - 1) / sqrt((2 * n - 3));
    }


    for (int im = 0; im <= Nmax + 2; im++) {
        for (int in = im; in <= Nmax + 2; in++) {
            int index = in * (in + 1) / 2 + im;
            double n = (double)in;
            double m = (double)im;

            f1mgt1[index] = sqrt((2 * n + 1) * (n + m + 2) * (n + m + 1) / (2 * n + 3));
            f2mgt1[index] = sqrt((2 * n + 1) * (n - m + 2) * (n - m + 1) / (2 * n + 3));
            f3mgt1[index] = sqrt((2 * n + 1) * (n + m + 1) * (n - m + 1) / (2 * n + 3));
            // ! tensor coefficient
            grdtcnm1[index] = sqrt((n + m + 1) * (n + m + 2) * (n + m + 3) * (n + m + 4) * (2 * n + 1) / (2 * n + 5));
            grdtcnm2[index] = sqrt((n - m + 1) * (n + m + 1) * (n + m + 2) * (n + m + 3) * (2 * n + 1) / (2 * n + 5));
            grdtcnm3[index] = 2.0 * sqrt((n - m + 1) * (n - m + 2) * (n + m + 1) * (n + m + 2) * (2 * n + 1) / (2 * n + 5));
            grdtcnm4[index] = sqrt((n - m + 4) * (n - m + 1) * (n - m + 2) * (n - m + 3) * (2 * n + 1) / (2 * n + 5));
            grdtcnm5[index] = sqrt((n + m + 1) * (n - m + 1) * (n - m + 2) * (n - m + 3) * (2 * n + 1) / (2 * n + 5));
            grdtcnm6[index] = sqrt((2 * n + 1) * (n + m + 1) * (n + m + 2) * (n - m + 1) * (n - m + 2) / (2 * n + 5));
            // ! coefficient for V,M
            vwnm1[index] = sqrt((2 * n + 1) / (pow(n, 2) - pow(m, 2))) * sqrt((2 * n - 1));
            vwnm2[index] = sqrt((2 * n + 1) / (pow(n, 2) - pow(m, 2))) * sqrt(((n - m - 1) * (n + m - 1)) / (2 * n - 3));

        }
    }

};

void calVW(double const xyz[], int const Nmax) {

    // V = {0e0}, W={0e0};

    double r_sqr = pow(xyz[0], 2) + pow(xyz[1], 2) + pow(xyz[2], 2);

    double rho = Radius * Radius / r_sqr;

    // ! Normalized coordinates

    double x0 = Radius * xyz[0] / r_sqr;
    double y0 = Radius * xyz[1] / r_sqr;
    double z0 = Radius * xyz[2] / r_sqr;

    V[0][0] = Radius / sqrt(r_sqr);
    W[0][0] = 0.e0;

    V[1][0] = sqrt(3.e0) * z0 * V[0][0];
    W[1][0] = 0.e0;


    for (int n = 2; n <= Nmax+2; n++) {

        V[n][0] = vwn1[n] * z0 * V[n - 1][0] - vwn2[n] * rho * V[n - 2][0];
        W[n][0] = 0.e0;
    }


    for (int m = 1; m <= Nmax + 2; m++) {
        if (m == 1) {
            V[1][1] = sqrt(3.) * x0 * V[0][0];
            W[1][1] = sqrt(3.) * y0 * V[0][0];
        }
        else
        {
            V[m][m] = vwmeq1[m] * (x0 * V[m - 1][m - 1] - y0 * W[m - 1][m - 1]);
            W[m][m] = vwmeq1[m] * (x0 * W[m - 1][m - 1] + y0 * V[m - 1][m - 1]);
        }

        if (m <= Nmax + 1) {
            V[m + 1][m] = sqrt((2 * m + 3)) * z0 * V[m][m];
            W[m + 1][m] = sqrt((2 * m + 3)) * z0 * W[m][m];
        }

        for (int n = m + 2; n <= Nmax + 2; n++) {
            int index = n * (n + 1) / 2 + m;
            V[n][m] = vwnm1[index] * z0 * V[n - 1][m] - vwnm2[index] * rho * V[n - 2][m];
            W[n][m] = vwnm1[index] * z0 * W[n - 1][m] - vwnm2[index] * rho * W[n - 2][m];
        }
    }

}

void getAcc(const int Nmax, const double Cpot[], const double Spot[], double acc[]) {
    // cout <<"sss1"<< V[20][3] << endl;
    for (int i=0; i < 3; i++) {
        acc[i] = 0.e0;
    }

    for (int m = 0; m <= Nmax; m++) {
        for (int n = m; n <= Nmax; n++) {
            int index = n * (n + 1) / 2 + m;
            double C = Cpot[index], S = Spot[index];
            if (m == 0) {
                acc[0] += -C * V[n + 1][1] * f1meq0[n];
                acc[1] += -C * W[n + 1][1] * f1meq0[n];
                acc[2] += -C * V[n + 1][0] * f2meq0[n];
            }else if(m==1) {
                acc[0] += 0.5e0 * (f1meq1[n] * (-C * V[n + 1][2] - S * W[n + 1][2]) + f2meq1[n] * C * V[n + 1][0]);
                acc[1] += 0.5e0 * (f1meq1[n] * (-C * W[n + 1][2] + S * V[n + 1][2]) + f2meq1[n] * S * V[n + 1][0]);
                acc[2] += f3meq1[n] * (-C * V[n + 1][1] - S * W[n + 1][1]);

            }
            else
            {
                acc[0] += 0.5e0 * (f1mgt1[index] * (-C * V[n + 1][m + 1] - S * W[n + 1][m + 1]) +
                    f2mgt1[index] * (+C * V[n + 1][m - 1] + S * W[n + 1][m - 1]));
                acc[1] += 0.5e0 * (f1mgt1[index] * (-C * W[n + 1][m + 1] + S * V[n + 1][m + 1]) +
                    f2mgt1[index] * (-C * W[n + 1][m - 1] + S * V[n + 1][m - 1]));
                acc[2] += f3mgt1[index] * (-C * V[n + 1][m] - S * W[n + 1][m]);
            }

        }

    }

    for (int i = 0; i <= 2; i++) {
        acc[i] = (GM / (Radius * Radius)) * acc[i];
    }
    // std::cout << "GM is:" << acc[1] << "ms!" << std::endl;
}

void getDu2xyz(const int Nmax, const double Cpot[], const double Spot[], double DU2XYZ[]) {

    for (int i = 0; i < 9; i++) {
        DU2XYZ[i] = 0.e0;
    }

    for (int m=0; m <= Nmax; m++) {
        for (int n = m; n <= Nmax; n++) {
            int index = n * (n + 1) / 2 + m;
            double C = Cpot[index], S = Spot[index];

            if (m == 0) {
                DU2XYZ[0] += 0.5 * (grdtcn1[n] * C * V[n + 2][2] - grdtcn3[n] * C * V[n + 2][0]);
                DU2XYZ[1] += 0.5 * grdtcn1[n] * C * W[n + 2][2];
                DU2XYZ[2] += grdtcn2[n] * C * V[n + 2][1];
                DU2XYZ[5] += grdtcn2[n] * C * W[n + 2][1];
                DU2XYZ[8] += grdtcnm6[index] * C * V[n + 2][0];
            }
            else if(m==1)
            {
                DU2XYZ[0] += 0.25 * (grdtcn4[n] * (C * V[n + 2][3] + S * W[n + 2][3]) -
                    grdtcn5[n] * (3. * C * V[n + 2][1] + S * W[n + 2][1]));
                DU2XYZ[1] += 0.25 * (grdtcn4[n] * (C * W[n + 2][3] - S * V[n + 2][3]) +
                    grdtcn5[n] * (-C * W[n + 2][1] - S * V[n + 2][1]));
                DU2XYZ[2] += 0.5 * (grdtcn6[n] * (C * V[n + 2][2] + S * W[n + 2][2]) -
                    grdtcn7[n] * C * V[n + 2][0]);
                DU2XYZ[5] += 0.5 * (grdtcn6[n] * (C * W[n + 2][2] - S * V[n + 2][2]) -
                    grdtcn7[n] * S * V[n + 2][0]);
                DU2XYZ[8] += grdtcnm6[index] * (C * V[n + 2][1] + S * W[n + 2][1]);
            }
            else if(m==2)
            {
                DU2XYZ[0] += 0.25 * (grdtcn8[n] * (+C * V[n + 2][4] + S * W[n + 2][4]) +
                    grdtcn9[n] * (-C * V[n + 2][2] - S * W[n + 2][2]) +
                    grdtcn10[n] * C * V[n + 2][0]);
                DU2XYZ[1] += 0.25 * (grdtcn8[n] * (+C * W[n + 2][m + 2] - S * V[n + 2][m + 2]) +
                    grdtcn10[n] * (-C * W[n + 2][m - 2] + S * V[n + 2][m - 2]));
                DU2XYZ[2] += 0.5 * (grdtcn11[n] * (+C * V[n + 2][m + 1] + S * W[n + 2][m + 1]) +
                    grdtcn12[n] * (-C * V[n + 2][m - 1] - S * W[n + 2][m - 1]));
                DU2XYZ[5] += 0.5 * (grdtcn11[n] * (+C * W[n + 2][m + 1] - S * V[n + 2][m + 1]) +
                    grdtcn12[n] * (+C * W[n + 2][m - 1] - S * V[n + 2][m - 1]));
                DU2XYZ[8] += grdtcnm6[index] * (+C * V[n + 2][m] + S * W[n + 2][m]);

            }
            else
            {
                DU2XYZ[0] += 0.25 * (grdtcnm1[index] * (+C * V[n + 2][m + 2] + S * W[n + 2][m + 2]) +
                    grdtcnm3[index] * (-C * V[n + 2][m] - S * W[n + 2][m]) +
                    grdtcnm4[index] * (+C * V[n + 2][m - 2] + S * W[n + 2][m - 2]));
                DU2XYZ[1] += 0.25 * (grdtcnm1[index] * (+C * W[n + 2][m + 2] - S * V[n + 2][m + 2]) +
                    grdtcnm4[index] * (-C * W[n + 2][m - 2] + S * V[n + 2][m - 2]));
                DU2XYZ[2] += 0.5 * (grdtcnm2[index] * (+C * V[n + 2][m + 1] + S * W[n + 2][m + 1]) +
                    grdtcnm5[index] * (-C * V[n + 2][m - 1] - S * W[n + 2][m - 1]));
                DU2XYZ[5] += 0.5 * (grdtcnm2[index] * (+C * W[n + 2][m + 1] - S * V[n + 2][m + 1]) +
                    grdtcnm5[index] * (+C * W[n + 2][m - 1] - S * V[n + 2][m - 1]));
                DU2XYZ[8] += grdtcnm6[index] * (+C * V[n + 2][m] + S * W[n + 2][m]);
            }

        }
    }

    double factor = GM / pow(Radius, 3);

    DU2XYZ[0] = factor * DU2XYZ[0];
    DU2XYZ[1] = factor * DU2XYZ[1];
    DU2XYZ[2] = factor * DU2XYZ[2];
    DU2XYZ[5] = factor * DU2XYZ[5];
    DU2XYZ[8] = factor * DU2XYZ[8];
    DU2XYZ[3] = DU2XYZ[1];
    DU2XYZ[7] = DU2XYZ[5];
    DU2XYZ[6] = DU2XYZ[2];
    DU2XYZ[4] = -DU2XYZ[0] - DU2XYZ[8];

}

void getDu2CS(const int Nmin, const int Nmax,  double dxdc[], double dydc[], double dzdc[],
    double dxds[], double dyds[], double dzds[]) {

    double factor = GM/pow(Radius,2);

    int max_n = Nmax;
    int min_n = Nmin;

    // cout <<"sss1"<<max_n << endl;
    // cout <<"sss1"<<min_n << endl;
    // cout <<"sss1"<<Nmin << endl;
    // cout <<"sss1"<<Nmax << endl;

    for (int m = 0; m <= max_n; m++) {
        // cout << dxdc[1231] << endl;
        for (int n = m; n <= max_n; n++) {
            int index = n * (n + 1) / 2 + m;

            if (n >= min_n) {

                if (m == 0) {
                    dxdc[index] = -factor * V[n + 1][1] * f1meq0[n];
                    dydc[index] = -factor * W[n + 1][1] * f1meq0[n];
                    dzdc[index] = -factor * V[n + 1][0] * f2meq0[n];
                }
                else if (m == 1)
                {
                    dxdc[index] = 0.5 * factor * (-V[n + 1][2] * f1meq1[n] + V[n + 1][0] * f2meq1[n]);
                    dxds[index] = 0.5 * factor * (-W[n + 1][2] * f1meq1[n]);
                    dydc[index] = 0.5 * factor * (-W[n + 1][2] * f1meq1[n]);
                    dyds[index] = 0.5 * factor * (+V[n + 1][2] * f1meq1[n] + V[n + 1][0] * f2meq1[n]);
                    dzdc[index] = -factor * V[n + 1][1] * f3meq1[n];
                    dzds[index] = -factor * W[n + 1][1] * f3meq1[n];
                }
                else
                {
                    dxdc[index] = 0.5 * factor * (-V[n + 1][m + 1] * f1mgt1[index]
                        + V[n + 1][m - 1] * f2mgt1[index]);
                    dxds[index] = 0.5 * factor * (-W[n + 1][m + 1] * f1mgt1[index]
                        + W[n + 1][m - 1] * f2mgt1[index]);
                    dydc[index] = 0.5 * factor * (-W[n + 1][m + 1] * f1mgt1[index]
                        - W[n + 1][m - 1] * f2mgt1[index]);
                    dyds[index] = 0.5 * factor * (+V[n + 1][m + 1] * f1mgt1[index]
                        + V[n + 1][m - 1] * f2mgt1[index]);
                    dzdc[index] = -factor * V[n + 1][m] * f3mgt1[index];
                    dzds[index] = -factor * W[n + 1][m] * f3mgt1[index];

                }

            }


        }
    }

}

double getPotential(const int Nmax, const double Cpot[], const double Spot[]) {

    double U_pot = 0.;
    for (int m = 0; m <= Nmax; m++) {
        for (int n = m; n <= Nmax; n++) {
            int index = n * (n + 1) / 2 + m;
            double C = Cpot[index], S = Spot[index];
            U_pot += C * V[n][m] + S * W[n][m];
        }
    }

    U_pot = GM / Radius * U_pot;
    return U_pot;
}



