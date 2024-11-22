#pragma once
#ifndef GRAVHDEF
#define GRAVHDEF

#include <iomanip>
#include<math.h>


#ifdef __cplusplus
extern "C" {
#endif

int getNlimit();

void setPar(double GM1, double R1);

void preprocess(int const Nmax);

void calVW(double const xyz[], int const Nmax);

void getAcc(const int Nmax, const double Cpot[], const double Spot[], double acc[]);

void getDu2xyz(const int Nmax, const double Cpot[], const double Spot[], double DU2XYZ[]);

void getDu2CS(const int Nmin, const int Nmax, double dxdc[], double dydc[], double dzdc[], double dxds[], double dyds[], double dzds[]);

double getPotential(const int Nmax, const double Cpot[], const double Spot[]);

#ifdef __cplusplus
}
#endif

#endif
