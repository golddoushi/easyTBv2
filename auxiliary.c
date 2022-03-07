#include <stdio.h>
#include <complex.h>
#include <Python.h>

#define TwoPI 6.28318530717958647692

/*
COMPILATION:
gcc --std=c99 -shared -fPIC -o auxiliary.dll .\auxiliary.c -I"D:\Program Files\Python38\include" -L"D:\Program Files\Python38\libs" -lpython3
*/

PyObject * readVelocity(char rpath[],int nspin, int nkpt, int nbands)
{
    printf("nspin: %d nkpt: %d nbands: %d\n",nspin,nkpt,nbands);
    printf("estimated memory: %d MB\n",2*6*nspin*nkpt*nbands*nbands*sizeof(double)/1024/1024);
    complex double * velocity_matrix=(complex double*)malloc(3*nspin*nkpt*nbands*nbands*sizeof(complex double));
    complex double cj=_Complex_I;

    float velocity_real;
    float velocity_imag;
    PyObject *velocity_array_real;
    PyObject *velocity_array_imag;
    velocity_array_real=PyTuple_New(3*nspin*nkpt*nbands*nbands);
    velocity_array_imag=PyTuple_New(3*nspin*nkpt*nbands*nbands);

    FILE*fp;
    fp=fopen(rpath,"r");
    char *buff;
    buff=(char*)malloc(200*sizeof(char));

    int cnt=0;
    for(int idir=0;idir<3;idir++){
        fgets(buff,100,fp);
        for(int ispin=0;ispin<nspin;ispin++){
            fgets(buff,100,fp);
            for(int ikpt=0;ikpt<nkpt;ikpt++){
                fgets(buff,100,fp);
                for(int nb1=0;nb1<nbands;nb1++){
                    for(int nb2=0;nb2<nbands;nb2++){
                        fgets(buff,100,fp);
                        sscanf(buff,"%18E%18E",&velocity_real,&velocity_imag);
                        velocity_matrix[cnt]=velocity_real+cj*velocity_imag;
                        PyTuple_SetItem(velocity_array_real, cnt,   PyFloat_FromDouble(velocity_real));
                        PyTuple_SetItem(velocity_array_imag, cnt,   PyFloat_FromDouble(velocity_imag));
                        cnt=cnt+1;
                    }//bands
                }//bands
                fgets(buff,100,fp);
            }// kpoints
        }// spin
    }// derivation direction
    fclose(fp);
    long long velocity_addr=(long long)velocity_matrix;
    printf("in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(complex double *));
    printf("pls. see if they are identical. Otherwise the calc. results with cdll are seriously wrong!\n");
    printf("in auxiliary.dll the addr. of velocity_matrix is %p\n",velocity_matrix);
    return Py_BuildValue("(OOL)",velocity_array_real,velocity_array_imag,velocity_addr);
}

PyObject * calcShiftBPVC(int nphoton, double EphotonList[nphoton], double erange[2], double tau, long long velocity_addr,
                    int nspin, int nkpt, int nbands, double eigs[nspin][nkpt][nbands], double occupations[nspin][nkpt][nbands],int nthreads)
{
    //complex double velocity_matrix[3][nspin][nkpt][nbands][nbands];
    complex double*velocity_matrix=(complex double*)velocity_addr;
    //printf("%f %f\n",creal(velocity_matrix[2]),cimag(velocity_matrix[2]));
    //exit(1);
    complex double *cx_xx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_xx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_yy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_yy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_xy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_xy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_yx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_yx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    for(int i=0;i<nphoton*nkpt;i++){
        cx_xx[i]=0;
        cy_xx[i]=0;
        cx_yy[i]=0;
        cy_yy[i]=0;
        cx_xy[i]=0;
        cy_xy[i]=0;
        cx_yx[i]=0;
        cy_yx[i]=0;
    }
    
    complex double itau=tau*_Complex_I;
    int bnd_base=nbands;
    int kpt_base=nbands*nbands;
    int spin_base=nkpt*kpt_base;
    int dir_base=nspin*spin_base;
    printf("entering Shift BPVC main loop...\n");
    /*
    for(int iband=0;iband<nbands;iband++){
        vx_nl=velocity_matrix[         0*spin_base+17*kpt_base+1*bnd_base+iband];
        vy_nl=velocity_matrix[dir_base+0*spin_base+17*kpt_base+1*bnd_base+iband];
        printf("vx: %E %E, vy: %E %E\n",creal(vx_nl),cimag(vx_nl),creal(vy_nl),cimag(vy_nl));
    }
    exit(1);
    */
    #pragma omp parallel num_threads(nthreads)
    {
    double El,Em,En,fl,fn,fln,omega;
    complex double Enl,Eln,Eml,Enm; // this is essnetial to consider energy difference as complex
    complex double vx_nl,vy_nl,vx_ln,vy_ln,vx_nn,vy_nn,vx_ll,vy_ll;
    complex double rx_nl,ry_nl,rxx_ln,rxy_ln,ryx_ln,ryy_ln;
    complex double SIGMA_xx_ln, SIGMA_xy_ln, SIGMA_yx_ln, SIGMA_yy_ln;
    complex double vx_lm,vy_lm,vx_mn,vy_mn;
    complex double factor_global1,factor_global2;
    int ipos;
    #pragma omp for
    for(int ikpt=0;ikpt<nkpt;ikpt++){
        for(int ispin=0;ispin<nspin;ispin++){
            for(int ibandl=0;ibandl<nbands;ibandl++){
                El=eigs[ispin][ikpt][ibandl];
                if(El<erange[0] | El>erange[1]) continue;
                fl=occupations[ispin][ikpt][ibandl];
                //printf("ikpt %d iband %d eig: %.6f occ: %.2f\n",ikpt,ibandl,El,fl);
                for(int ibandn=0;ibandn<nbands;ibandn++){
                    En=eigs[ispin][ikpt][ibandn];
                    if(En<erange[0] | En>erange[1]) continue;
                    fn=occupations[ispin][ikpt][ibandn];
                    fln=fl-fn;
                    if(fabs(fln)<0.1) continue;
                    Enl=En-El;
                    Eln=-Enl;
                    vx_nl=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandl];
                    vy_nl=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandl];
                    vx_ln=conj(vx_nl);
                    vy_ln=conj(vy_nl);
                    vx_nn=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandn];
                    vy_nn=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandn];
                    vx_ll=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandl];
                    vy_ll=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandl];

                    rx_nl=_Complex_I*vx_nl/Enl;
                    ry_nl=_Complex_I*vy_nl/Enl;

                    SIGMA_xx_ln=0;
                    SIGMA_xy_ln=0;
                    SIGMA_yx_ln=0;
                    SIGMA_yy_ln=0;
                    for(int ibandm=0;ibandm<nbands;ibandm++){
                        if(ibandm==ibandn|ibandm==ibandl)continue;
                        Em=eigs[ispin][ikpt][ibandm];
                        Eml=Em-El+itau; // to avoid singularity
                        Enm=En-Em-itau; // to avoid singularity

                        vx_lm=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandm];
                        vy_lm=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandm];
                        vx_mn=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandm*bnd_base+ibandn];
                        vy_mn=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandm*bnd_base+ibandn];
                        
                        
                        SIGMA_xx_ln+=vx_lm*vx_mn*(1/Eml+1/Enm);
                        SIGMA_xy_ln+=vx_lm*vy_mn/Eml+vy_lm*vx_mn/Enm;
                        SIGMA_yx_ln+=vy_lm*vx_mn/Eml+vx_lm*vy_mn/Enm;
                        SIGMA_yy_ln+=vy_lm*vy_mn*(1/Eml+1/Enm);
                    }

                    rxx_ln=_Complex_I/Eln*(
                        (vx_ln*(vx_ll-vx_nn)+vx_ln*(vx_ll-vx_nn))/Eln+SIGMA_xx_ln);
                    rxy_ln=_Complex_I/Eln*(
                        (vx_ln*(vy_ll-vy_nn)+vy_ln*(vx_ll-vx_nn))/Eln+SIGMA_xy_ln);
                    ryx_ln=_Complex_I/Eln*(
                        (vy_ln*(vx_ll-vx_nn)+vx_ln*(vy_ll-vy_nn))/Eln+SIGMA_yx_ln);
                    ryy_ln=_Complex_I/Eln*(
                        (vy_ln*(vy_ll-vy_nn)+vy_ln*(vy_ll-vy_nn))/Eln+SIGMA_yy_ln);


                    //printf("vx: %f %f, vy: %f %f\n",creal(vx_nl),cimag(vx_nl),creal(vy_nl),cimag(vy_nl));
                    
                        for(int iphoton=0;iphoton<nphoton;iphoton++){
                            omega=EphotonList[iphoton];
                            factor_global1=fln*(1/(Enl-omega-itau))/2;
                            factor_global2=fln*(1/(Enl+omega+itau))/2;

                            ipos=iphoton*nkpt+ikpt;
                            cx_xx[ipos]+=(rx_nl*rxx_ln*factor_global1+rx_nl*rxx_ln*factor_global2);
                            cy_xx[ipos]+=(rx_nl*rxy_ln*factor_global1+rx_nl*rxy_ln*factor_global2);
                            cx_yy[ipos]+=(ry_nl*ryx_ln*factor_global1+ry_nl*ryx_ln*factor_global2);
                            cy_yy[ipos]+=(ry_nl*ryy_ln*factor_global1+ry_nl*ryy_ln*factor_global2);
                            cx_xy[ipos]+=(rx_nl*ryx_ln*factor_global1+ry_nl*rxx_ln*factor_global2);
                            cy_xy[ipos]+=(rx_nl*ryy_ln*factor_global1+ry_nl*rxy_ln*factor_global2);
                            cx_yx[ipos]+=(ry_nl*rxx_ln*factor_global1+rx_nl*ryx_ln*factor_global2);
                            cy_yx[ipos]+=(ry_nl*rxy_ln*factor_global1+rx_nl*ryy_ln*factor_global2);
                        }
                    
                }
            }
        }
    }
    }
    printf("Shift BPVC finished\n");

    PyObject *cx_xx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yx_imag=PyTuple_New(nphoton*nkpt);
    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            int ipos=iphoton*nkpt+ikpt;
            PyTuple_SetItem(cx_xx_real, ipos,   PyFloat_FromDouble(creal(cx_xx[ipos])));
            PyTuple_SetItem(cx_xx_imag, ipos,   PyFloat_FromDouble(cimag(cx_xx[ipos])));
            PyTuple_SetItem(cy_xx_real, ipos,   PyFloat_FromDouble(creal(cy_xx[ipos])));
            PyTuple_SetItem(cy_xx_imag, ipos,   PyFloat_FromDouble(cimag(cy_xx[ipos])));
            PyTuple_SetItem(cx_yy_real, ipos,   PyFloat_FromDouble(creal(cx_yy[ipos])));
            PyTuple_SetItem(cx_yy_imag, ipos,   PyFloat_FromDouble(cimag(cx_yy[ipos])));
            PyTuple_SetItem(cy_yy_real, ipos,   PyFloat_FromDouble(creal(cy_yy[ipos])));
            PyTuple_SetItem(cy_yy_imag, ipos,   PyFloat_FromDouble(cimag(cy_yy[ipos])));
            PyTuple_SetItem(cx_xy_real, ipos,   PyFloat_FromDouble(creal(cx_xy[ipos])));
            PyTuple_SetItem(cx_xy_imag, ipos,   PyFloat_FromDouble(cimag(cx_xy[ipos])));
            PyTuple_SetItem(cy_xy_real, ipos,   PyFloat_FromDouble(creal(cy_xy[ipos])));
            PyTuple_SetItem(cy_xy_imag, ipos,   PyFloat_FromDouble(cimag(cy_xy[ipos])));
            PyTuple_SetItem(cx_yx_real, ipos,   PyFloat_FromDouble(creal(cx_yx[ipos])));
            PyTuple_SetItem(cx_yx_imag, ipos,   PyFloat_FromDouble(cimag(cx_yx[ipos])));
            PyTuple_SetItem(cy_yx_real, ipos,   PyFloat_FromDouble(creal(cy_yx[ipos])));
            PyTuple_SetItem(cy_yx_imag, ipos,   PyFloat_FromDouble(cimag(cy_yx[ipos])));
            //if(ikpt==18){
            //    printf("cxxx %f %f\n",creal(cx_xx[ipos]),cimag(cx_xx[ipos]));
            //    printf("cyxx %f %f\n",creal(cy_xx[ipos]),cimag(cy_xx[ipos]));
            //    printf("cxyy %f %f\n",creal(cx_yy[ipos]),cimag(cx_yy[ipos]));
            //    printf("cyyy %f %f\n",creal(cy_yy[ipos]),cimag(cy_yy[ipos]));
            //}
        }
    }
    free(cx_xx);free(cy_xx);free(cx_yy);free(cy_yy);free(cx_xy);free(cy_xy);free(cx_yx);free(cy_yx);
    return Py_BuildValue("(OOOOOOOOOOOOOOOO)",
    cx_xx_real,cx_xx_imag,cy_xx_real,cy_xx_imag,
    cx_yy_real,cx_yy_imag,cy_yy_real,cy_yy_imag,
    cx_xy_real,cx_xy_imag,cy_xy_real,cy_xy_imag,
    cx_yx_real,cx_yx_imag,cy_yx_real,cy_yx_imag);
}

PyObject * calcBPVC(int nphoton, double EphotonList[nphoton], double erange[2], double tau, long long velocity_addr,
                    int nspin, int nkpt, int nbands, double eigs[nspin][nkpt][nbands], double occupations[nspin][nkpt][nbands],int nthreads)
{
    //complex double velocity_matrix[3][nspin][nkpt][nbands][nbands];
    complex double*velocity_matrix=(complex double*)velocity_addr;
    //printf("%f %f\n",creal(velocity_matrix[2]),cimag(velocity_matrix[2]));
    //exit(1);
    complex double *cx_xx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_xx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_yy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_yy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_xy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_xy=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cx_yx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    complex double *cy_yx=(complex double*)malloc(nphoton*nkpt*sizeof(complex double));
    for(int i=0;i<nphoton*nkpt;i++){
        cx_xx[i]=0;
        cy_xx[i]=0;
        cx_yy[i]=0;
        cy_yy[i]=0;
        cx_xy[i]=0;
        cy_xy[i]=0;
        cx_yx[i]=0;
        cy_yx[i]=0;
    }
    
    int bnd_base=nbands;
    int kpt_base=nbands*nbands;
    int spin_base=nkpt*kpt_base;
    int dir_base=nspin*spin_base;
    printf("entering BPVC main loop...\n");
    /*
    for(int iband=0;iband<nbands;iband++){
        vx_nl=velocity_matrix[         0*spin_base+17*kpt_base+1*bnd_base+iband];
        vy_nl=velocity_matrix[dir_base+0*spin_base+17*kpt_base+1*bnd_base+iband];
        printf("vx: %E %E, vy: %E %E\n",creal(vx_nl),cimag(vx_nl),creal(vy_nl),cimag(vy_nl));
    }
    exit(1);
    */
    #pragma omp parallel num_threads(nthreads)
    {
    double El,Em,En,fl,fn,fln,omega;
    complex double vx_nl,vy_nl,vx_lm,vy_lm,vx_mn,vy_mn,factor_global1,factor_global2;
    complex double itau=tau*_Complex_I;
    int ipos;
    #pragma omp for
    for(int ikpt=0;ikpt<nkpt;ikpt++){
        for(int ispin=0;ispin<nspin;ispin++){
            for(int ibandl=0;ibandl<nbands;ibandl++){
                El=eigs[ispin][ikpt][ibandl];
                if(El<erange[0] | El>erange[1]) continue;
                fl=occupations[ispin][ikpt][ibandl];
                //printf("ikpt %d iband %d eig: %.6f occ: %.2f\n",ikpt,ibandl,El,fl);
                for(int ibandn=0;ibandn<nbands;ibandn++){
                    En=eigs[ispin][ikpt][ibandn];
                    if(En<erange[0] | En>erange[1]) continue;
                    fn=occupations[ispin][ikpt][ibandn];
                    fln=fl-fn;
                    if(fabs(fln)<0.1) continue;
                    vx_nl=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandl];
                    vy_nl=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandn*bnd_base+ibandl];
                    //printf("vx: %f %f, vy: %f %f\n",creal(vx_nl),cimag(vx_nl),creal(vy_nl),cimag(vy_nl));
                    for(int ibandm=0;ibandm<nbands;ibandm++){
                        Em=eigs[ispin][ikpt][ibandm];
                        if(Em<erange[0] | Em>erange[1]) continue;
                        //if(ibandm==ibandn) continue;
                        //if(ibandl==ibandm) continue;
                        vx_lm=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandm];
                        vy_lm=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandl*bnd_base+ibandm];
                        vx_mn=velocity_matrix[         ispin*spin_base+ikpt*kpt_base+ibandm*bnd_base+ibandn];
                        vy_mn=velocity_matrix[dir_base+ispin*spin_base+ikpt*kpt_base+ibandm*bnd_base+ibandn];
                        for(int iphoton=0;iphoton<nphoton;iphoton++){
                            omega=EphotonList[iphoton];
                            factor_global1=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                            //factor_global2=fln/(omega*omega)/(Em-El-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau))/2;
                            ipos=iphoton*nkpt+ikpt;
                            cx_xx[ipos]+=vx_nl*vx_lm*vx_mn*factor_global1;//-vx_nl*vx_lm*vx_mn*factor_global2;
                            cy_xx[ipos]+=vx_nl*vx_lm*vy_mn*factor_global1;//-vx_nl*vy_lm*vx_mn*factor_global2;
                            cx_yy[ipos]+=vy_nl*vy_lm*vx_mn*factor_global1;//-vy_nl*vx_lm*vy_mn*factor_global2;
                            cy_yy[ipos]+=vy_nl*vy_lm*vy_mn*factor_global1;//-vy_nl*vy_lm*vy_mn*factor_global2;
                            cx_xy[ipos]+=vx_nl*vy_lm*vx_mn*factor_global1;//-vx_nl*vx_lm*vy_mn*factor_global2;
                            cy_xy[ipos]+=vx_nl*vy_lm*vy_mn*factor_global1;//-vx_nl*vy_lm*vy_mn*factor_global2;
                            cx_yx[ipos]+=vy_nl*vx_lm*vx_mn*factor_global1;//-vy_nl*vx_lm*vx_mn*factor_global2;
                            cy_yx[ipos]+=vy_nl*vx_lm*vy_mn*factor_global1;//-vy_nl*vy_lm*vx_mn*factor_global2;
                            //if(ikpt==18){
                            //    printf("omega %f fac %f %f cxxx %f %f\n",
                            //    omega,creal(factor_global),cimag(factor_global),creal(cx_xx[iphoton*nphoton+ikpt]),cimag(cx_xx[iphoton*nphoton+ikpt]));
                            //}
                            //printf("%d\n",ikpt);
                            //exit(1);
                        }
                    }
                }
            }
        }
    }
    }
    printf("BPVC finished\n");

    PyObject *cx_xx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_xy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xy_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_xy_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cx_yx_imag=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yx_real=PyTuple_New(nphoton*nkpt);
    PyObject *cy_yx_imag=PyTuple_New(nphoton*nkpt);
    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            int ipos=iphoton*nkpt+ikpt;
            PyTuple_SetItem(cx_xx_real, ipos,   PyFloat_FromDouble(creal(cx_xx[ipos])));
            PyTuple_SetItem(cx_xx_imag, ipos,   PyFloat_FromDouble(cimag(cx_xx[ipos])));
            PyTuple_SetItem(cy_xx_real, ipos,   PyFloat_FromDouble(creal(cy_xx[ipos])));
            PyTuple_SetItem(cy_xx_imag, ipos,   PyFloat_FromDouble(cimag(cy_xx[ipos])));
            PyTuple_SetItem(cx_yy_real, ipos,   PyFloat_FromDouble(creal(cx_yy[ipos])));
            PyTuple_SetItem(cx_yy_imag, ipos,   PyFloat_FromDouble(cimag(cx_yy[ipos])));
            PyTuple_SetItem(cy_yy_real, ipos,   PyFloat_FromDouble(creal(cy_yy[ipos])));
            PyTuple_SetItem(cy_yy_imag, ipos,   PyFloat_FromDouble(cimag(cy_yy[ipos])));
            PyTuple_SetItem(cx_xy_real, ipos,   PyFloat_FromDouble(creal(cx_xy[ipos])));
            PyTuple_SetItem(cx_xy_imag, ipos,   PyFloat_FromDouble(cimag(cx_xy[ipos])));
            PyTuple_SetItem(cy_xy_real, ipos,   PyFloat_FromDouble(creal(cy_xy[ipos])));
            PyTuple_SetItem(cy_xy_imag, ipos,   PyFloat_FromDouble(cimag(cy_xy[ipos])));
            PyTuple_SetItem(cx_yx_real, ipos,   PyFloat_FromDouble(creal(cx_yx[ipos])));
            PyTuple_SetItem(cx_yx_imag, ipos,   PyFloat_FromDouble(cimag(cx_yx[ipos])));
            PyTuple_SetItem(cy_yx_real, ipos,   PyFloat_FromDouble(creal(cy_yx[ipos])));
            PyTuple_SetItem(cy_yx_imag, ipos,   PyFloat_FromDouble(cimag(cy_yx[ipos])));
            //if(ikpt==18){
            //    printf("cxxx %f %f\n",creal(cx_xx[ipos]),cimag(cx_xx[ipos]));
            //    printf("cyxx %f %f\n",creal(cy_xx[ipos]),cimag(cy_xx[ipos]));
            //    printf("cxyy %f %f\n",creal(cx_yy[ipos]),cimag(cx_yy[ipos]));
            //    printf("cyyy %f %f\n",creal(cy_yy[ipos]),cimag(cy_yy[ipos]));
            //}
        }
    }
    free(cx_xx);free(cy_xx);free(cx_yy);free(cy_yy);free(cx_xy);free(cy_xy);free(cx_yx);free(cy_yx);
    return Py_BuildValue("(OOOOOOOOOOOOOOOO)",
    cx_xx_real,cx_xx_imag,cy_xx_real,cy_xx_imag,
    cx_yy_real,cx_yy_imag,cy_yy_real,cy_yy_imag,
    cx_xy_real,cx_xy_imag,cy_xy_real,cy_xy_imag,
    cx_yx_real,cx_yx_imag,cy_yx_real,cy_yx_imag);
}
