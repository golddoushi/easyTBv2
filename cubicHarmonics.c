#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double Rnl(double r, int n, int l){
    if(r<1e-5) return 0;
    if(n==1){
        return exp(-r);
    }else if(n==2){
        if(l==0){
            return (1-r/2)*exp(-r/2);
        }else if(l==1){
            return r*exp(-r/2);
        }
    }else if(n==3){
        if(l==0){
            return (1-2/3*r+2/27*r*r)*exp(-r/3);
        }else if(l==1){
            return (1-r/6)*r*exp(-r/3);
        }else if(l==2){
            return (r*r)*exp(-r/3);
        }
    }else if(n==4){
        if(l==0){
            return (1-3/4*r+r*r/8-r*r*r/192)*exp(-r/4);
        }else if(l==1){
            return (1-r/4+r*r/80)*r*exp(-r/4);
        }else if(l==2){
            return (1-r/12)*r*r*exp(-r/4);
        }else if(l==3){
            return r*r*r*exp(-r/4);
        }
    }
    printf("Error occured in Rnl in .so");
    return 0;
}

double Clm(int l, char m[], double x, double y, double z, double r){
    //printf("inner %c%c %.6f %.6f\n",m[0],m[1],y,r);
    if(r<0) r=sqrt(x*x+y*y+z*z);
    if(r<1e-5) return 0;
        
    if(l==0){
        return 1;
    }else if(l==1){
        if(m[0]=='z'){
            return z/r;
        }else if(m[0]=='x'){
            return x/r;
        }else if(m[0]=='y'){
            return y/r;
        }
    }else if(l==2){
        if(m[0]=='z'){
            return (3*z*z-r*r)/2/sqrt(3)/(r*r);
        }else if(m[0]=='x' && m[1]=='z'){
            return x*z/(r*r);
        }else if(m[0]=='y' && m[1]=='z'){
            return y*z/(r*r);
        }else if(m[0]=='x' && m[1]=='y'){
            return x*y/(r*r);
        }else if(m[0]=='x' && m[1]=='2'){
            //printf("innner: x:%.6f y:%.6f\n",x,y);
            return (x*x-y*y)/2/(r*r);
        }
    }else if(l==3){
        if(m[0]=='z' && m[1]=='3'){
            return z*(5*z*z-3*r*r)/2/sqrt(15)/(r*r*r);
        }else if(m[0]=='x' && m[1]=='z'){
            return x*(5*z*z-r*r)/2/sqrt(10)/(r*r*r);
        }else if(m[0]=='y' && m[1]=='z'){
            return y*(5*z*z-r*r)/sqrt(10)/(r*r*r);
        }else if(m[0]=='x' && m[1]=='y'){
            return x*y*z/(r*r*r);
        }else if(m[0]=='z' && m[1]=='('){
            return z*(x*x-y*y)/2/(r*r*r);
        }else if(m[0]=='x' && m[1]=='('){
            return x*(x*x-3*y*y)/2/sqrt(6)/(r*r*r);
        }else if(m[0]=='y' && m[1]=='('){
            return y*(3*x*x-y*y)/2/sqrt(6)/(r*r*r);
        }
    }
}
 
double * vecDotTensor(int m, int n, double vec[m], double M[m][n]){
    double *result=(double*)malloc(n*sizeof(double));
    int i,j;
    for(i=0;i<n;i++){
        result[i]=0;
        for(j=0;j<m;j++) result[i]+=vec[j]*M[j][i];
    }
    return result;
}

double ** tensorDotTensor(int a, int b, int c, double L[a][b], double R[b][c]){
    double **result=(double**)malloc(a*sizeof(double*));
    int i,j;
    for(i=0;i<a;i++){
        double *row=(double*)malloc(c*sizeof(double));
        for(j=0;j<c;j++){
            row[j]=0;
            //printf("%d\n",i*a+j);
            for(int k=0;k<b;k++) row[j]+=L[i][k]*R[k][j];
            //printf("%.3f\n",row[j]);
        }
        result[i]=row;
    }
    return result;
}

PyObject * generateCubicOrbital(int n, int l, char m[], int ng, double L, double Trans[3][3]){
    //printf("%c%c\n",m[0],m[1]);
    //for(int i=0;i<3;i++) for(int j=0;j<3;j++) printf("%.6f\n",Trans[i][j]);
    double dl=L/ng;
    double shift=L/2;
    int icnt=0;
    int ix,iy,iz;
    double x,y,z,_x,_y,_z,r,_rnl,_clm,value,integral;
    //printf("before Data ng=%d\n",ng);
    double *Data=(double*)malloc(ng*ng*ng*sizeof(double));
    //printf("after Data\n");
    double xyz[3];
    integral=0;
    //printf("initialized\n");
    for(iz=0;iz<ng;iz++){
        for(iy=0;iy<ng;iy++){
            for (ix=0;ix<ng;ix++){
                //printf("%d %d %d, %.6f %.6f\n",ix,iy,iz,dl,shift);
                xyz[0]=dl*ix-shift;
                xyz[1]=dl*iy-shift;
                xyz[2]=dl*iz-shift;
                //printf("x:%.6f y:%.6f z:%.6f\n",xyz[0],xyz[1],xyz[2]);
                //printf("%c%c\n",m[0],m[1]);
                double *xyz_rotated=vecDotTensor(3,3,xyz,Trans);
                double x=xyz_rotated[0];
                double y=xyz_rotated[1];
                double z=xyz_rotated[2];
                free(xyz_rotated);
                //printf("x:%.6f y:%.6f z:%.6f\n",x,y,z);
                r=sqrt(x*x+y*y+z*z);

                _rnl=Rnl(r, n, l);
                _clm=Clm(l, m, x, y, z, r);
                //printf("%c%c\n",m[0],m[1]);
                value=_rnl*_clm;
                //printf("r %.6f c %.6f value %.6f\n",_rnl,_clm,value);
                integral+=value*value;
                Data[icnt]=value;
                icnt+=1;
            }
        }
    
    }

    PyObject *PyData;
    PyData=PyTuple_New(ng*ng*ng);
    double moduli=sqrt(integral);
    for(icnt=0;icnt<ng*ng*ng;icnt++) PyTuple_SetItem(PyData, icnt, PyFloat_FromDouble(Data[icnt]/moduli));
    return PyData;
}
