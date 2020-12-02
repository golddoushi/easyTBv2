#include<stdio.h>
#include<complex.h>
#include<Python.h>
#include<lapacke.h>

#define TwoPI 6.28318530717958647692
// style==0 conventional Monkhorst-Pack algorithm, otherwise GAMMA
long long getKMesh(int style, int Lx, int Ly, int Lz){
    double * kmesh=(double*)malloc((Lx*Ly*Lz*3)*sizeof(double));
    for(int ix=0;ix<Lx;ix++){
        for(int iy=0;iy<Ly;iy++){
            for(int iz=0;iz<Lz;iz++){ // original Monhorst-Pack algorithm, avoid the gamma sampling
                double kx=(2.*ix-Lx+1)/(Lx*2);
                double ky=(2.*iy-Ly+1)/(Ly*2);
                double kz=(2.*iz-Lz+1)/(Lz*2);
                //printf("%.3f %.3f %.3f\n",kx,ky,kz);
                if(style!=0){ // biased sampling to include gamma
                    if(Lx%2==0) kx=(2.*ix-Lx)/(Lx*2);
                    if(Ly%2==0) ky=(2.*iy-Ly)/(Ly*2);
                    if(Lz%2==0) kz=(2.*iz-Lz)/(Lz*2);
                }
                int ikpt=(ix*Ly*Lz+iy*Lz+iz)*3;
                *(kmesh+ikpt)=kx;
                *(kmesh+ikpt+1)=ky;
                *(kmesh+ikpt+2)=kz;
                //printf("%.3f %.3f %.3f\n",kx,ky,kz);
            }
        }
    }

    printf("in ham.dll the addr. of kmesh is %p\n",kmesh);
    long long kmesh_address=(long long)kmesh;
    return kmesh_address;
}

long long getHk(int nkpt, long long kmesh_address,
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings]){//,
                       //int ikpt_print){
    double * kmesh=(double*)kmesh_address;
    printf("in ham.dll the addr. of kmesh is %p\n",kmesh);
    lapack_complex_double * Hk=(lapack_complex_double *)malloc((nkpt*norbs*norbs)*sizeof(lapack_complex_double));
    // initialize
    for(int i=0;i<(nkpt*norbs*norbs);i++) *(Hk+i)=0;

    // calculate matrix elements
    for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
        int kBaseIndex=ikpt*norbs*norbs;
        for(int ihopping=0;ihopping<nhoppings;ihopping++){
            double phase=0;
            for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*(*(kmesh+ikpt*3+i));
            lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
            *(Hk+kBaseIndex+sourceID[ihopping]*norbs+targetID[ihopping])+=tk;
        }
    }
    long long Hk_address=(long long)Hk;
    printf("in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(lapack_complex_double *));
    printf("pls. see if they are identical. Otherwise the calc. results with cdll are seriously wrong!\n");
    printf("in ham.dll the addr. of Hk is %p\n",Hk);
    /*printf("c Hk at %d th kpt:\n",ikpt_print);
    for(int i=0;i<norbs;i++){
        for(int j=0;j<norbs;j++){
            printf("(%.3f, %.3f) ",creal(*(Hk+ikpt_print*norbs*norbs+i*norbs+j)),cimag(*(Hk+ikpt_print*norbs*norbs+i*norbs+j)));
        }
        printf("\n");
    }*/
    return Hk_address;
}

int cleanKMeshAndHk(long long kmesh_address, long long Hk_address){
    double*kmesh=(double*)kmesh_address;
    lapack_complex_double*Hk=(lapack_complex_double*)Hk_address;
    printf("try to release mallocated kmesh at %p and Hk at %p\n",kmesh,Hk);
    free(kmesh);
    free(Hk);
    printf("done successfully\n");
    return 0;
}