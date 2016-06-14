/**
*	export LHAPATH=$HOME/Documentos/git/TFM/programas/PDFs
*	g++ crossections.cpp -o crossections -I $(lhapdf-config --incdir) -L $(lhapdf-config --libdir) -lLHAPDF
*/

#include <LHAPDF/LHAPDF.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace LHAPDF;
using namespace std;

#define NMASS 500
#define NxPoints 50


int main(){
	const double s = 13000*13000; //CM-energy at LHC
	const double mt = 173; //top quark mass
	const double GF = 4.54164e3; //Fermi constant (pb)
	initPDFSet("cteq66.LHgrid");
	double mass[NMASS];
	double theta[NMASS];
	double factor[NMASS];
	FILE *fin, *fout;
	fin = fopen("angles.txt", "rt");	

	for(int i=0; i<NMASS; i++){ // Reading masses and mixing angles from file (computed with decays.py)
		fscanf(fin, "%lf", &mass[i]);
		fscanf(fin, "%lf", &theta[i]);
		fscanf(fin, "%lf", &factor[i]);
	}
	fclose(fin);

	fout = fopen("crosssec.txt", "wt");
	for(int i=0; i<NMASS; i++){
		double Q = mass[i];
		double xmin = mass[i]*mass[i]/s;
		double delta = (1.0-xmin)/NxPoints;
		double integ=0;
		double sigma0 = GF * pow(alphasPDF(Q)* sin(theta[i]), 2)/(128*sqrt(2)*M_PI)*factor[i];
		for (int j=0; j<NxPoints; j++){
			initPDF(0);
			double x = xmin + j*delta;
			integ += xfx(x,Q,0)*xfx(mass[i]*mass[i]/(x*s), Q,0)*delta ; 
		}
		double sigma = sigma0*integ;
		fprintf(fout, "%e\t%e\n", mass[i], sigma);
	}
	fclose(fout);
	
}
