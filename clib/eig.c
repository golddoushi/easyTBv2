#define PY_SSIZE_T_CLEAN
#include<stdio.h>
#include<complex.h>
#include<Python.h>
#include<time.h>
#include<fftw3.h>
#ifdef MKL_ILP64
#define MKL_Complex16 complex double
#include<mkl.h>
#else
#include<lapacke.h>
#include<cblas.h>
#endif

#define TwoPI 6.28318530717958647692
#define PI 3.14159265358979323846
typedef unsigned int uint;
typedef unsigned long long ulong;

int omp_get_thread_num();

FILE*fp;

int output(const char* s){
    int myrank=omp_get_thread_num();
    if(myrank==0){
        fp=fopen("./eig.log","a");
        fprintf(fp,s);
        fclose(fp);
    }
    return 0;
}

PyObject * calcFullBandsAndVecs(PyObject* self, PyObject* args){
    // read in all parameters
    PyObject* py_nkpt;
    PyObject* py_kmesh;
    PyObject* py_norbs;
    PyObject* py_nhoppings;
    PyObject* py_sourceID;
    PyObject* py_targetID;
    PyObject* py_overLat;
    PyObject* py_hopping_real;
    PyObject* py_hopping_imag;
    PyObject* py_nthreads;
    PyObject* callback;  // callback function
    printf("start parsing...\n");
    PyArg_ParseTuple(args,"OOOOOOOOOOO",
        &py_nkpt,&py_kmesh,&py_norbs,&py_nhoppings,&py_sourceID,&py_targetID,
        &py_overLat,&py_hopping_real,&py_hopping_imag,&py_nthreads,&callback);
    uint nkpt=(uint)PyLong_AS_LONG(py_nkpt);
    uint norbs=(uint)PyLong_AS_LONG(py_norbs);
    uint nhoppings=(uint)PyLong_AS_LONG(py_nhoppings);
    uint nthreads=(uint)PyLong_AS_LONG(py_nthreads);
    printf("nkpt=%d norbs=%d nhoppings=%d nthreads=%d\n",nkpt,norbs,nhoppings,nthreads);
    double *kmesh=(double*)malloc(nkpt*3*sizeof(double));
    for(uint ikpt=0;ikpt<nkpt;ikpt++)for(uint idir=0;idir<3;idir++)
        kmesh[ikpt*3+idir]=PyFloat_AsDouble(PyList_GetItem(py_kmesh,ikpt*3+idir));
    uint *sourceID=(uint*)malloc(nhoppings*sizeof(uint));
    uint *targetID=(uint*)malloc(nhoppings*sizeof(uint));
    int *overLat=(int*)malloc(nhoppings*3*sizeof(int));
    double*hopping_real=(double*)malloc(nhoppings*sizeof(double));
    double*hopping_imag=(double*)malloc(nhoppings*sizeof(double));
    for(uint ihopping=0;ihopping<nhoppings;ihopping++){
        sourceID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_sourceID,ihopping));
        targetID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_targetID,ihopping));
        for(uint idir=0;idir<3;idir++)
            overLat[ihopping*3+idir]=PyLong_AS_LONG(PyList_GetItem(py_overLat,ihopping*3+idir));
        hopping_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_hopping_real,ihopping));
        hopping_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_hopping_imag,ihopping));
    }
    double * Spectra_tmp=(double*)malloc(nkpt*norbs*sizeof(double));
    double * Vector_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    
    if(nkpt<nthreads*3)nthreads=1;
    printf("use %d cores in ham.dll::calcFullBandsAndVecs\n",nthreads);
    // calculate matrix elements
    uint index;
    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk
        double * eigs=(double*)malloc(norbs*sizeof(double)); // output eigenvalues of Hk
        #pragma omp for
        for(uint ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //printf("kpt%d %.3f %.3f %.3f\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2]);
            //int kBaseIndex=ikpt*norbs*norbs;
            for(uint i=0;i<norbs;i++)for(uint j=0;j<norbs;j++)hk[i*norbs+j]=0;
            //printf("hoppings\n");
            for(uint ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (uint i=0;i<3;i++) phase+=overLat[ihopping*3+i]*kmesh[ikpt*3+i];
                
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*_Complex_I*phase);
                hk[sourceID[ihopping]*norbs+targetID[ihopping]]+=tk;
            }
            
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,hk,norbs,eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigs: %.4f %.4f\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2],eigs[0],eigs[1]);
            for(uint iorb=0;iorb<norbs;iorb++) Spectra_tmp[ikpt*norbs+iorb]=eigs[iorb];
            for(uint iorb=0;iorb<norbs;iorb++){
                for(uint jorb=0;jorb<norbs;jorb++){
                    index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Vector_r_tmp[index]=creal(hk[iorb*norbs+jorb]);
                    Vector_i_tmp[index]=cimag(hk[iorb*norbs+jorb]);
                }
            }
        }
        free(hk);free(eigs);
    }

    PyObject * Spectra=PyTuple_New(nkpt*norbs);
    PyObject * Vector_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_i=PyTuple_New(nkpt*norbs*norbs);

    for(uint ikpt=0;ikpt<nkpt;ikpt++){
        for(uint iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(Spectra, ikpt*norbs+iorb, PyFloat_FromDouble(Spectra_tmp[ikpt*norbs+iorb]));
        for(uint iorb=0;iorb<norbs;iorb++){
            for(uint jorb=0;jorb<norbs;jorb++){
                index=ikpt*norbs*norbs+iorb*norbs+jorb;
                PyTuple_SetItem(Vector_r, index, PyFloat_FromDouble(Vector_r_tmp[index]));
                PyTuple_SetItem(Vector_i, index, PyFloat_FromDouble(Vector_i_tmp[index]));
                //if(ikpt==0)printf("%.6e %.6e, ",creal(Dhk[iorb][jorb][0]),cimag(Dhk[iorb][jorb][0]));
            }
        }
    }
    free(kmesh);free(sourceID);free(targetID);free(overLat);free(hopping_real);free(hopping_imag);
    return Py_BuildValue("(OOO)",Spectra,Vector_r,Vector_i);
}

PyObject * calcFullBandsAndVecs_OVLCorrection_DoubelSpin(PyObject* self, PyObject* args){
    // read in all parameters
    PyObject* py_nkpt;
    PyObject* py_kmesh;
    PyObject* py_norbs;
    PyObject* py_nhoppings;
    PyObject* py_sourceID;
    PyObject* py_targetID;
    PyObject* py_overLat;
    PyObject* py_treal;
    PyObject* py_timag;
    PyObject* py_oreal;
    PyObject* py_oimag;
    PyObject* py_nthreads;
    PyObject* callback;  // callback function
    printf("start parsing...\n");
    PyArg_ParseTuple(args,"OOOOOOOOOOOOO",
        &py_nkpt,&py_kmesh,&py_norbs,&py_nhoppings,&py_sourceID,&py_targetID,
        &py_overLat,
        &py_treal,&py_timag,&py_oreal,&py_oimag,
        &py_nthreads,&callback);
    uint nkpt=(uint)PyLong_AS_LONG(py_nkpt);
    uint norbs=(uint)PyLong_AS_LONG(py_norbs);
    uint nhoppings=(uint)PyLong_AS_LONG(py_nhoppings);
    uint nthreads=(uint)PyLong_AS_LONG(py_nthreads);
    printf("nkpt=%d norbs=%d nhoppings=%d nthreads=%d\n",nkpt,norbs,nhoppings,nthreads);
    double *kmesh=(double*)malloc(nkpt*3*sizeof(double));
    for(uint ikpt=0;ikpt<nkpt;ikpt++)for(uint idir=0;idir<3;idir++)
        kmesh[ikpt*3+idir]=PyFloat_AsDouble(PyList_GetItem(py_kmesh,ikpt*3+idir));
    uint *sourceID=(uint*)malloc(nhoppings*sizeof(uint));
    uint *targetID=(uint*)malloc(nhoppings*sizeof(uint));
    int *overLat=(int*)malloc(nhoppings*3*sizeof(int));
    double*hopping_real=(double*)malloc(nhoppings*sizeof(double));
    double*hopping_imag=(double*)malloc(nhoppings*sizeof(double));
    double*overlap_real=(double*)malloc(nhoppings*sizeof(double));
    double*overlap_imag=(double*)malloc(nhoppings*sizeof(double));
    for(uint ihopping=0;ihopping<nhoppings;ihopping++){
        sourceID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_sourceID,ihopping));
        targetID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_targetID,ihopping));
        for(uint idir=0;idir<3;idir++)
            overLat[ihopping*3+idir]=PyLong_AS_LONG(PyList_GetItem(py_overLat,ihopping*3+idir));
        hopping_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_treal,ihopping));
        hopping_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_timag,ihopping));
        overlap_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_oreal,ihopping));
        overlap_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_oimag,ihopping));
    }
    double * Spectra_tmp=(double*)malloc(nkpt*norbs*sizeof(double));
    double * Vector_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    
    if(nkpt<nthreads*3)nthreads=1;
    printf("use %d cores in eig.pyd::calcFullBandsAndVecs_OVLCorrection_DoubelSpin\n",nthreads);
    unsigned long long index; // the index for all states includes kpts and bands
    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * Hk     =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init original Hk
        lapack_complex_double * HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * Ok=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init overlap matrix
        lapack_complex_double * OkHalf=(lapack_complex_double*)malloc((norbs/2)*(norbs/2)*sizeof(lapack_complex_double)); // for spin-polarized and noncolinear
        lapack_complex_double * W=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init the eigen-vecs of Ok
        double * eigs=(double*)malloc(norbs*sizeof(double)); // eigenvalues of Hk
        double * Oklam=(double*)malloc(norbs*sizeof(double)); // eigenvalues of overlap
        double * OklamHalf=(double*)malloc(norbs/2*sizeof(double)); // for spin-polarized or noncolinear case
        #pragma omp for
        for(uint ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //printf("kpt%d %.3f %.3f %.3f\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //int kBaseIndex=ikpt*norbs*norbs;
            for(uint i=0;i<norbs;i++){
                for(uint j=0;j<norbs;j++){
                    Hk[i*norbs+j]=0;
                    HkWave[i*norbs+j]=0;
                    Ok[i*norbs+j]=0;
                }
            }
            for(uint i=0;i<norbs/2;i++)for(uint j=0;j<norbs/2;j++)OkHalf[i*norbs/2+j]=0;
            //printf("hoppings\n");
            double phase;
            lapack_complex_double tk, sk;
            for(uint ihopping=0;ihopping<nhoppings;ihopping++){
                phase=0;
                for (uint i=0;i<3;i++) phase+=overLat[ihopping*3+i]*kmesh[ikpt*3+i];
                tk=(hopping_real[ihopping]+_Complex_I*hopping_imag[ihopping])*cexp(TwoPI*_Complex_I*phase);
                Hk[sourceID[ihopping]*norbs+targetID[ihopping]]+=tk;
                if(sourceID[ihopping]<norbs/2 & targetID[ihopping]<norbs/2){
                    sk=(overlap_real[ihopping]+_Complex_I*overlap_imag[ihopping])*cexp(TwoPI*_Complex_I*phase);
                    OkHalf[sourceID[ihopping]*norbs/2+targetID[ihopping]]+=sk;
                }
            }
            //printf("kpt%d %.3f %.3f %.3f Overlap:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(ovlk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //printf("Hamiltonian:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(hk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs/2,OkHalf,norbs/2,OklamHalf);
            if(info!=0)printf("LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigvals of overlap matrix:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs/2;iorb++)printf(" %.3f",koHalf[iorb]);
            //printf("\n");
            //printf("kpt%d %.3f %.3f %.3f Overlap eivenvecs:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(ovlk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);
            // Set full ovlk
            for(uint i=0;i<norbs/2;i++){
                Oklam[i]        =OklamHalf[i];
                Oklam[norbs/2+i]=OklamHalf[i];
                for(uint j=0;j<norbs/2;j++){
                    W[i*norbs          +j]          =OkHalf[i*norbs/2+j];
                    W[(norbs/2+i)*norbs+(norbs/2+j)]=OkHalf[i*norbs/2+j];
                }
            }
            // HkWave= 1/sqrt(Olam) @ W^dagger @ Hk @ W @ 1/sqrt(Olam)
            for(uint iorb=0;iorb<norbs;iorb++)
               for(uint jorb=0;jorb<norbs;jorb++)
                  for(uint icontract=0;icontract<norbs;icontract++)
                    for(uint jcontract=0;jcontract<norbs;jcontract++)
                        HkWave[iorb*norbs+jorb]+=conj(W[icontract*norbs+iorb])*Hk[icontract*norbs+jcontract]*W[jcontract*norbs+jorb];
                     //B[iorb*norbs+jorb]+=ovlk[icontract*norbs+iorb]*hk[icontract*norbs+jorb];
            //printf("Real part of center matrix:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(B[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);
            for(uint iorb=0;iorb<norbs;iorb++)for(uint jorb=0;jorb<norbs;jorb++)
               HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb])/sqrt(Oklam[jorb]);

            info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,HkWave,norbs,eigs);
            if(info!=0)printf("LAPACKE failed for the eigen. eq. for D, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigvals:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++)printf(" %.3f",eigs[iorb]);
            //printf("\n");
            //exit(1);
            // V=W @ 1/sqrt(Olam) @ U
            for(uint iorb=0;iorb<norbs;iorb++){
                for(uint jorb=0;jorb<norbs;jorb++){
                   HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb]);
                   Hk[iorb*norbs+jorb]=0;
               }
            }
            
            for(uint iorb=0;iorb<norbs;iorb++)
               for(uint jorb=0;jorb<norbs;jorb++)
                  for(uint icontract=0;icontract<norbs;icontract++)
                     Hk[iorb*norbs+jorb]+=W[iorb*norbs+icontract]*HkWave[icontract*norbs+jorb];
            
            //printf("Real part of eigen-vecs:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",cimag(hk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);

            for(uint iorb=0;iorb<norbs;iorb++) Spectra_tmp[ikpt*norbs+iorb]=eigs[iorb];
            for(uint iorb=0;iorb<norbs;iorb++){
                for(uint jorb=0;jorb<norbs;jorb++){
                    index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Vector_r_tmp[index]=creal(Hk[iorb*norbs+jorb]);
                    Vector_i_tmp[index]=cimag(Hk[iorb*norbs+jorb]);
                }
            }
        }
        free(Hk);free(HkWave);free(Ok);free(OkHalf);free(W);free(eigs);free(Oklam);free(OklamHalf);
    }

    PyObject * Spectra=PyTuple_New(nkpt*norbs);
    PyObject * Vector_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_i=PyTuple_New(nkpt*norbs*norbs);

    for(uint ikpt=0;ikpt<nkpt;ikpt++){
        for(uint iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(Spectra, ikpt*norbs+iorb, PyFloat_FromDouble(Spectra_tmp[ikpt*norbs+iorb]));
        for(uint iorb=0;iorb<norbs;iorb++){
            for(uint jorb=0;jorb<norbs;jorb++){
                index=ikpt*norbs*norbs+iorb*norbs+jorb;
                PyTuple_SetItem(Vector_r, index, PyFloat_FromDouble(Vector_r_tmp[index]));
                PyTuple_SetItem(Vector_i, index, PyFloat_FromDouble(Vector_i_tmp[index]));
                //if(ikpt==0)printf("%.6e %.6e, ",creal(Dhk[iorb][jorb][0]),cimag(Dhk[iorb][jorb][0]));
            }
        }
    }
    free(kmesh);free(sourceID);free(targetID);free(overLat);free(hopping_real);free(hopping_imag);free(overlap_real);free(overlap_imag);
    return Py_BuildValue("(OOO)",Spectra,Vector_r,Vector_i);
}

//***************************************************//
//             Green's function part                 //
//***************************************************//
// Auxiliary func, mod, especially for negative mod
int mod(int a, int b){
    int result=a%b;
    if (result<0)result+=b;
    return result;
}

// generate k-axis in 3 directions, with convention exp(-1j*2pi*dk)
int constructKAxis(int nk1, int nk2, int nk3, lapack_complex_double k1_axis[nk1], lapack_complex_double k2_axis[nk2], lapack_complex_double k3_axis[nk3]){
    int nkpt=nk1*nk2*nk3;//kgrid[0]*kgrid[1]*kgrid[2];
    double dk[3]={1./nk1,1./nk2,1./nk3};
    //int kxy=nk1*nk2;
    lapack_complex_double k1_a=cexp(TwoPI*_Complex_I*dk[0]);
    lapack_complex_double k2_a=cexp(TwoPI*_Complex_I*dk[1]);
    lapack_complex_double k3_a=cexp(TwoPI*_Complex_I*dk[2]);
    lapack_complex_double base=1.0;
    for(int ik=0;ik<nk1;ik++){
        k1_axis[ik]=base;
        base*=k1_a;
    }
    base=1.0;
    for(int ik=0;ik<nk2;ik++){
        k2_axis[ik]=base;
        base*=k2_a;
    }
    base=1.0;
    for(int ik=0;ik<nk3;ik++){
        k3_axis[ik]=base;
        base*=k3_a;
    }
    return 0;
}

PyObject * calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin(PyObject* self, PyObject* args){
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    fclose(fp);
    // read in all parameters
    PyObject* py_kgrid;
    PyObject* py_norbs;
    PyObject* py_nhoppings;
    PyObject* py_sourceID;
    PyObject* py_targetID;
    PyObject* py_overLat;
    PyObject* py_treal;
    PyObject* py_timag;
    PyObject* py_overlap;
    PyObject* py_nthreads;
    PyObject* callback;  // callback function
    printf("start parsing...\n");
    PyArg_ParseTuple(args,"OOOOOOOOOOO",
        &py_kgrid,&py_norbs,&py_nhoppings,&py_sourceID,&py_targetID,
        &py_overLat,
        &py_treal,&py_timag,&py_overlap,
        &py_nthreads,&callback);
    uint kgrid[3];
    kgrid[0]=(uint)PyLong_AS_LONG(PyList_GetItem(py_kgrid,0));
    kgrid[1]=(uint)PyLong_AS_LONG(PyList_GetItem(py_kgrid,1));
    kgrid[2]=(uint)PyLong_AS_LONG(PyList_GetItem(py_kgrid,2));
    uint nkpt=kgrid[0]*kgrid[1]*kgrid[2];
    uint kxy=kgrid[0]*kgrid[1];
    uint norbs=(uint)PyLong_AS_LONG(py_norbs);
    uint nhoppings=(uint)PyLong_AS_LONG(py_nhoppings);
    uint nthreads=(uint)PyLong_AS_LONG(py_nthreads);
    printf("nkpt=%d norbs=%d nhoppings=%d nthreads=%d\n",nkpt,norbs,nhoppings,nthreads);
    uint *sourceID=(uint*)malloc(nhoppings*sizeof(uint));
    uint *targetID=(uint*)malloc(nhoppings*sizeof(uint));
    int *overLat=(int*)malloc(nhoppings*3*sizeof(int));
    double*hopping_real=(double*)malloc(nhoppings*sizeof(double));
    double*hopping_imag=(double*)malloc(nhoppings*sizeof(double));
    double*overlap=(double*)malloc(nhoppings*sizeof(double));
    for(uint ihopping=0;ihopping<nhoppings;ihopping++){
        sourceID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_sourceID,ihopping));
        targetID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_targetID,ihopping));
        for(uint idir=0;idir<3;idir++)
            overLat[ihopping*3+idir]=PyLong_AS_LONG(PyList_GetItem(py_overLat,ihopping*3+idir));
        hopping_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_treal,ihopping));
        hopping_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_timag,ihopping));
        overlap[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_overlap,ihopping));
    }

    lapack_complex_double *k1_axis=(lapack_complex_double*)malloc(kgrid[0]*sizeof(lapack_complex_double));
    lapack_complex_double *k2_axis=(lapack_complex_double*)malloc(kgrid[1]*sizeof(lapack_complex_double));
    lapack_complex_double *k3_axis=(lapack_complex_double*)malloc(kgrid[2]*sizeof(lapack_complex_double));
    constructKAxis(kgrid[0],kgrid[1],kgrid[2],k1_axis,k2_axis,k3_axis);

    double * spectra=(double*)malloc((unsigned long long)nkpt*norbs*sizeof(double));
    lapack_complex_double * vecs=(lapack_complex_double*)malloc((unsigned long long)nkpt*norbs*norbs*sizeof(lapack_complex_double));
    
    if(nkpt<nthreads*3)nthreads=1;
    printf("use %d cores in eig.pyd::calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"use %d cores in eig.pyd::calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin\n",nthreads);
    fclose(fp);
    unsigned long long index; // the index for all states includes kpts and bands
    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * Hk     =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init original Hk
        lapack_complex_double * HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * Ok=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init overlap matrix
        lapack_complex_double * OkHalf=(lapack_complex_double*)malloc((norbs/2)*(norbs/2)*sizeof(lapack_complex_double)); // for spin-polarized and noncolinear
        lapack_complex_double * W=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init the eigen-vecs of Ok
        double * eigs=(double*)malloc(norbs*sizeof(double)); // eigenvalues of Hk
        double * Oklam=(double*)malloc(norbs*sizeof(double)); // eigenvalues of overlap
        double * OklamHalf=(double*)malloc(norbs/2*sizeof(double)); // for spin-polarized or noncolinear case
        uint ikpt, ihopping, iorb, jorb, icontract, jcontract, i, j; // necessary indexes for loops

        uint my_rank=omp_get_thread_num();
        uint my_nkpt_portion=nkpt/nthreads/50;
        uint my_icnt=0;
        if(my_nkpt_portion==0)my_nkpt_portion=1;
        if(my_rank==0){
            fp=fopen("./c_log","a");
            for(i=0;i<51;i++){
                printf("-");
                fprintf(fp,"_");
            }
            printf("\n");
            fprintf(fp,"\n");
            fclose(fp);
        }
        #pragma omp for
        for(ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            // ikpt = ik1 + ik2*kgrid[0] + ik3*kgrid[0]*kgrid[1]
            uint ik1=ikpt%kgrid[0];
            uint ik2=(ikpt-ik1)%kxy/kgrid[0];
            uint ik3=ikpt/kxy;
            // initialize matrix elements to zero for further contractions
            for(i=0;i<norbs;i++){
                for(j=0;j<norbs;j++){
                    Hk[i*norbs+j]=0;
                    HkWave[i*norbs+j]=0;
                    Ok[i*norbs+j]=0;
                }
            }
            for(i=0;i<norbs/2;i++)for(uint j=0;j<norbs/2;j++)OkHalf[i*norbs/2+j]=0;
            //printf("hoppings\n");
            lapack_complex_double phase;
            lapack_complex_double tk, sk;
            for(ihopping=0;ihopping<nhoppings;ihopping++){
                phase= k1_axis[mod(ik1*overLat[ihopping*3+0],kgrid[0])]
                      *k2_axis[mod(ik2*overLat[ihopping*3+1],kgrid[1])]
                      *k3_axis[mod(ik3*overLat[ihopping*3+2],kgrid[2])];
                tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*phase;
                Hk[sourceID[ihopping]*norbs+targetID[ihopping]]+=tk;
                if(sourceID[ihopping]<norbs/2 & targetID[ihopping]<norbs/2){
                    sk=overlap[ihopping]*phase;
                    OkHalf[sourceID[ihopping]*norbs/2+targetID[ihopping]]+=sk;
                }
            }
            //printf("kpt%d %.3f %.3f %.3f Overlap:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(ovlk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //printf("Hamiltonian:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(hk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs/2,OkHalf,norbs/2,OklamHalf);
            if(info!=0)printf("LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigvals of overlap matrix:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs/2;iorb++)printf(" %.3f",koHalf[iorb]);
            //printf("\n");
            //printf("kpt%d %.3f %.3f %.3f Overlap eivenvecs:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(ovlk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);
            // Set full ovlk
            for(i=0;i<norbs/2;i++){
                Oklam[i]        =OklamHalf[i];
                Oklam[norbs/2+i]=OklamHalf[i];
                for(uint j=0;j<norbs/2;j++){
                    W[i*norbs          +j]          =OkHalf[i*norbs/2+j];
                    W[(norbs/2+i)*norbs+(norbs/2+j)]=OkHalf[i*norbs/2+j];
                }
            }
            // HkWave= 1/sqrt(Olam) @ W^dagger @ Hk @ W @ 1/sqrt(Olam)
            for(iorb=0;iorb<norbs;iorb++)
               for(jorb=0;jorb<norbs;jorb++)
                  for(icontract=0;icontract<norbs;icontract++)
                    for(jcontract=0;jcontract<norbs;jcontract++)
                        HkWave[iorb*norbs+jorb]+=conj(W[icontract*norbs+iorb])*Hk[icontract*norbs+jcontract]*W[jcontract*norbs+jorb];
                     //B[iorb*norbs+jorb]+=ovlk[icontract*norbs+iorb]*hk[icontract*norbs+jorb];
            //printf("Real part of center matrix:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",creal(B[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);
            for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
               HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb])/sqrt(Oklam[jorb]);

            info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,HkWave,norbs,eigs);
            if(info!=0)printf("LAPACKE failed for the eigen. eq. for D, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigvals:\n",ikpt,kmesh[ikpt*3+0],kmesh[ikpt*3+1],kmesh[ikpt*3+2]);
            //for(uint iorb=0;iorb<norbs;iorb++)printf(" %.3f",eigs[iorb]);
            //printf("\n");
            //exit(1);
            // Calc. the true eigen-vecs. V=W @ 1/sqrt(Olam) @ U
            for(iorb=0;iorb<norbs;iorb++){
                for(jorb=0;jorb<norbs;jorb++){
                   HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb]);
                   Hk[iorb*norbs+jorb]=0;
               }
            }
            for(iorb=0;iorb<norbs;iorb++)
               for(jorb=0;jorb<norbs;jorb++)
                  for(icontract=0;icontract<norbs;icontract++)
                     Hk[iorb*norbs+jorb]+=W[iorb*norbs+icontract]*HkWave[icontract*norbs+jorb];
            
            //printf("Real part of eigen-vecs:\n");
            //for(uint iorb=0;iorb<norbs;iorb++){
            //    for(uint jorb=0;jorb<norbs;jorb++)printf(" %.3f",cimag(hk[iorb*norbs+jorb]));
            //    printf("\n");
            //}
            //exit(1);

            for(iorb=0;iorb<norbs;iorb++) spectra[ikpt*norbs+iorb]=eigs[iorb];
            for(iorb=0;iorb<norbs;iorb++){
                for(jorb=0;jorb<norbs;jorb++){
                    index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    vecs[index]=Hk[iorb*norbs+jorb];
                }
            }

            my_icnt+=1;
            if(my_rank==0 & my_icnt%my_nkpt_portion==0){
                printf(">");
                fp=fopen("./c_log","a");
                fprintf(fp,">");
                fclose(fp);
            }
        }
        free(Hk);free(HkWave);free(Ok);free(OkHalf);free(W);free(eigs);free(Oklam);free(OklamHalf);
    }
    free(sourceID);free(targetID);free(overLat);free(hopping_real);free(hopping_imag);free(overlap);
    free(k1_axis);free(k2_axis);free(k3_axis);
    long long eigs_addr=(long long)spectra;
    long long vecs_addr=(long long)vecs;
    printf("\nin ham.dll the addr. of eigs and vecs are %p, %p\n",eigs_addr,vecs_addr);
    return Py_BuildValue("(LL)",eigs_addr,vecs_addr);
}

PyObject * calcFullBandsAndVecsInMemory_OVLCorrection_FFT(PyObject* self, PyObject* args){
    // read in all parameters
    PyObject* py_kgrid;
    PyObject* py_norbs;
    PyObject* py_nhoppings;
    PyObject* py_sourceID;
    PyObject* py_targetID;
    PyObject* py_overLat;
    PyObject* py_treal;
    PyObject* py_timag;
    PyObject* py_overlap;
    PyObject* py_ispin;
    PyObject* py_nthreads;
    PyObject* callback;  // callback function
    uint icomp, iorb, jorb;
    ulong ihopping, ikpt, ielement;
    output("Args parsing into calcFullBandsAndVecsInMemory_OVLCorrection_FFT\n");
    char output_info[512];
    PyArg_ParseTuple(args,"OOOOOOOOOOOO",
        &py_kgrid,&py_norbs,&py_nhoppings,&py_sourceID,&py_targetID,
        &py_overLat,
        &py_treal,&py_timag,&py_overlap,&py_ispin,
        &py_nthreads,&callback);
    uint ispin=(uint)PyLong_AS_LONG(py_ispin);
    uint kgrid[3];
    for(icomp=0;icomp<3;icomp++)
        kgrid[icomp]=(uint)PyLong_AS_LONG(PyList_GetItem(py_kgrid,icomp));
    
    uint nKgrid=kgrid[0]*kgrid[1]*kgrid[2];
    uint norbs=(uint)PyLong_AS_LONG(py_norbs);
    uint nhoppings=(uint)PyLong_AS_LONG(py_nhoppings);
    sprintf(output_info,"nkpt=%d norbs=%d ispin=%d nhoppings=%d\n",nKgrid,norbs,ispin,nhoppings);output(output_info);
    uint nthreads=(uint)PyLong_AS_LONG(py_nthreads);
    sprintf(output_info,"use %d cores in eig.pyd::calcFullBandsAndVecsInMemory_OVLCorrection_FFT\n",nthreads);output(output_info);

    uint *sourceID=(uint*)malloc(nhoppings*sizeof(uint));
    uint *targetID=(uint*)malloc(nhoppings*sizeof(uint));
    int *overLat=(int*)malloc(nhoppings*3*sizeof(int));
    double*hopping_real=(double*)malloc(nhoppings*sizeof(double));
    double*hopping_imag=(double*)malloc(nhoppings*sizeof(double));
    double*overlap=(double*)malloc(nhoppings*sizeof(double));
    for(ihopping=0;ihopping<nhoppings;ihopping++){
        sourceID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_sourceID,ihopping));
        targetID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_targetID,ihopping));
        for(uint idir=0;idir<3;idir++)
            overLat[ihopping*3+idir]=PyLong_AS_LONG(PyList_GetItem(py_overLat,ihopping*3+idir));
        hopping_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_treal,ihopping));
        hopping_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_timag,ihopping));
        overlap[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_overlap,ihopping));
    }
    output("Args. parsing finished.\n");
    sprintf(output_info,"estimate memory cost (per thread) %d MB\n",
                         (nKgrid*norbs*sizeof(double)+
                         nKgrid*norbs*norbs*sizeof(lapack_complex_double)+
                         norbs*norbs*nKgrid*sizeof(lapack_complex_double)+
                         nKgrid*norbs*norbs*sizeof(lapack_complex_double))/1024/1024);
    output(output_info);
    clock_t time0=clock();
    double * spectra=(double*)malloc(nKgrid*norbs*sizeof(double));                               // eigen values
    lapack_complex_double *Hrs=(lapack_complex_double*)malloc(norbs*norbs*nKgrid*sizeof(lapack_complex_double)); // eigen vectors
    lapack_complex_double *Hks=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double)); // eigen vectors
    lapack_complex_double *Ors=(lapack_complex_double*)malloc(norbs*norbs*nKgrid*sizeof(lapack_complex_double));      // overlap matrix in k-space
    lapack_complex_double *Oks=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double));      // overlap matrix in k-space
    
    // fftw_plan use global variables. So, we should create them outside the parallel loop
    fftw_complex *tR=(fftw_complex*)malloc(nKgrid*sizeof(fftw_complex));
    fftw_complex *tK=(fftw_complex*)malloc(nKgrid*sizeof(fftw_complex));
    fftw_plan p_Ham=fftw_plan_dft_3d(kgrid[0],kgrid[1],kgrid[2],tR,tK,FFTW_BACKWARD,FFTW_MEASURE);
    fftw_complex *oR=(fftw_complex*)malloc(nKgrid*sizeof(fftw_complex));
    fftw_complex *oK=(fftw_complex*)malloc(nKgrid*sizeof(fftw_complex));
    fftw_plan p_Overlap=fftw_plan_dft_3d(kgrid[0],kgrid[1],kgrid[2],oR,oK,FFTW_BACKWARD,FFTW_MEASURE);
    output("FFTW ready.\n");
    
    // set real-space Ham. and overlaps
    int Rvec0, Rvec1, Rvec2; // lattice point
    uint isource, itarget; // orbital indexes
    ulong index; // the index for all states includes kpts and bands
    for(ikpt=0;ikpt<nKgrid;ikpt++){for(iorb=0;iorb<norbs;iorb++){for(jorb=0;jorb<norbs;jorb++)
        {Hrs[ikpt*norbs*norbs+iorb*norbs+jorb]=0;Ors[ikpt*norbs*norbs+iorb*norbs+jorb]=0;}}}
    for(ihopping=0;ihopping<nhoppings;ihopping++){
        Rvec0=overLat[ihopping*3+0];
        Rvec1=overLat[ihopping*3+1];
        Rvec2=overLat[ihopping*3+2];

        if(Rvec0<0) Rvec0+=kgrid[0];
        if(Rvec1<0) Rvec1+=kgrid[1];
        if(Rvec2<0) Rvec2+=kgrid[2];

        if(Rvec0<0 | Rvec1<0 | Rvec2<0){
            sprintf(output_info,"kmesh %dx%dx%d too slim to accomodate Bravais vector %d,%d,%d\n",kgrid[0],kgrid[1],kgrid[2],overLat[ihopping*3+0],overLat[ihopping*3+1],overLat[ihopping*3+2]);output(output_info);
            exit(1);
        }
        isource=sourceID[ihopping];
        itarget=targetID[ihopping];
        index=(isource*norbs+itarget)*nKgrid+Rvec0*kgrid[1]*kgrid[2]+Rvec1*kgrid[2]+Rvec2;
        //if(index>nKgrid*norbs*norbs){
        //    printf("Error: index %d > %d, orb(%d-%d) Bravais [%d %d %d]\n",index,nKgrid*norbs*norbs,isource,itarget,Rvec0,Rvec1,Rvec2);
        //}
        Hrs[index]=hopping_real[ihopping]+_Complex_I*hopping_imag[ihopping];
        Ors[index]=overlap[ihopping];
        //if(isource==0 & itarget==0 & overlap[ihopping]>1E-7){
        //    printf("%d-%d [%d %d %d] Ovl:%.2E\n",isource,itarget,Rvec0,Rvec1,Rvec2,creal(Ors[index]));
        //}
    }
    output("real-space hoppings and overlaps are set.\n");
    uint iorb_debug=53;
    uint jorb_debug=53;
    //for(uint ik0=0;ik0<kgrid[0];ik0++){
    //    for(uint ik1=0;ik1<kgrid[1];ik1++)printf("%.2E ",creal(Ors[ik0*kgrid[1]*kgrid[2]+ik1*kgrid[2]]));
    //    printf("\n");
    //}
    //exit(1);
    // obtain hoppings and overlaps in k-space
    //#pragma omp parallel num_threads(nthreads)
    {
        uint iOrbPair, ik1, ik2, ik3;
        ulong igrid, ik, index;
        //#pragma omp for
        for(iOrbPair=0;iOrbPair<norbs*norbs;iOrbPair++){
            // NOTE: it is necessary to copy the data for FFT, since the data type are different in principle
            for(igrid=0;igrid<nKgrid;igrid++){
                tR[igrid]=Hrs[iOrbPair*nKgrid+igrid];
                oR[igrid]=Ors[iOrbPair*nKgrid+igrid];
                //if(iOrbPair==iorb_debug*norbs+jorb_debug & igrid%kgrid[1]==0)printf("\n");
                //if(iOrbPair==iorb_debug*norbs+jorb_debug)printf("%.1E,%.1E ",creal(oR[igrid]),cimag(oR[igrid]));
                
            }
            //if(iOrbPair==iorb_debug*norbs+jorb_debug)printf("\nFFTW");
            //exit(1);
            fftw_execute(p_Ham);
            fftw_execute(p_Overlap);
            /*for(igrid=0;igrid<nKgrid;igrid++){
                if(iOrbPair==iorb_debug*norbs+jorb_debug & igrid%kgrid[1]==0)printf("\n");
                if(iOrbPair==iorb_debug*norbs+jorb_debug)printf("%.1E,%.1E ",creal(oK[igrid]),cimag(oK[igrid]));
            }*/
            //if(iOrbPair==iorb_debug*norbs+jorb_debug)exit(1);
            //printf("fft iOrbPair %d finished\n",iOrbPair);
            for(ik1=0;ik1<kgrid[0];ik1++){
            for(ik2=0;ik2<kgrid[1];ik2++){
            for(ik3=0;ik3<kgrid[2];ik3++){
                igrid=ik1*kgrid[1]*kgrid[2]+ik2*kgrid[2]+ik3; // ik3 run the fastest for real-grid
                ik   =ik1+ik2*kgrid[0]+ik3*kgrid[0]*kgrid[1]; // ik1 run the fastest for reciprocal-grid
                index=ik*norbs*norbs+iOrbPair;
                Hks[index]=tK[igrid];
                Oks[index]=oK[igrid];
                //printf("%d %d-%d %.1E,%.1E \n",index,iOrbPair/norbs,iOrbPair%norbs,creal(Oks[index]),cimag(Oks[index]));
            }
            }
            }
            //exit(1);
        }
    }
    output("k-space hoppings and overlaps are set.\n");
    /*for(uint igrid=0;igrid<nKgrid;igrid++){
        if(igrid%kgrid[1]==0)printf("\n");
        ulong index=igrid*norbs*norbs+(iorb_debug*norbs+jorb_debug);
        printf("%.1E ",creal(Oks[index]));
    }
    exit(1);*/
    free(tR);free(tK);free(oR);free(oK);
    fftw_destroy_plan(p_Ham);fftw_destroy_plan(p_Overlap);
    output("FFT finished\n");

    // solve generalized eigen-eqs.
    #pragma omp parallel num_threads(nthreads)
    {
        uint ik, iorb, jorb, icontract, jcontract, ieig;
        lapack_complex_double * _HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * Wk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init the eigen-vecs of Ok
        lapack_complex_double * WkHalf=(lapack_complex_double*)malloc((norbs/2)*(norbs/2)*sizeof(lapack_complex_double)); // one quarter the eigen-vecs of Ok for double spin case
        double * eigs=(double*)malloc(norbs*sizeof(double)); // eigenvalues of Hk
        double * Oklam=(double*)malloc(norbs*sizeof(double)); // eigenvalues of overlap
        double alpha, beta;
        alpha=1;beta=0; // for matrix multiply

        int my_rank=omp_get_thread_num();
        uint my_nkpt_portion=nKgrid/nthreads/50;
        uint my_icnt=0;
        if(my_nkpt_portion==0)my_nkpt_portion=1;
        if(my_rank==0){sprintf(output_info,"> corresponds to ~%d kpts\n",my_nkpt_portion*50);output(output_info);}

        #pragma omp for
        for(ik=0;ik<nKgrid;ik++){
            if(ispin==2){ // only one quarter overlap will be solved, and copy to the other diagonal region
                for(iorb=0;iorb<norbs/2;iorb++)for(jorb=0;jorb<norbs/2;jorb++)
                    WkHalf[iorb*(norbs/2)+jorb]=Oks[ik*norbs*norbs+iorb*norbs+jorb];
                
                int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs/2,WkHalf,norbs/2,Oklam);
                if(info!=0){printf(output_info,"LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);exit(1);}
                // Set full ovlk
                for(iorb=0;iorb<norbs/2;iorb++){
                    Oklam[norbs/2+iorb]=Oklam[iorb];
                    for(jorb=0;jorb<norbs/2;jorb++){
                        Wk[          iorb*norbs+          jorb]=WkHalf[iorb*(norbs/2)+jorb];
                        Wk[(iorb+norbs/2)*norbs+(jorb+norbs/2)]=WkHalf[iorb*(norbs/2)+jorb];
                        Wk[(iorb+norbs/2)*norbs          +jorb]=0;
                        Wk[          iorb*norbs+(jorb+norbs/2)]=0;
                    }
                }
            }else{ // solve the whole overlap matrix
                for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
                   Wk[iorb*norbs+jorb]=Oks[ik*norbs*norbs+iorb*norbs+jorb];
                int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,Wk,norbs,Oklam);
                if(info!=0){printf("LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);exit(1);}
            }
            // to check the positivity of overlap eigenvals
            if(Oklam[0]<=0){
                output("Error, eigenvalues of overlap is not positive:");
                for(ieig=0;ieig<norbs;ieig++){sprintf(output_info," %.3f",Oklam[ieig]);output(output_info);}
                output("\n");
                exit(1);
            }
                
            // HkWave= 1/sqrt(Olam) @ W^dagger @ Hk @ W @ 1/sqrt(Olam)
            // 1. _HkWave = Hk @ W
            // it is essential to initialize the matrix before cblas
            for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)_HkWave[iorb*norbs+jorb]=0;
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, norbs, norbs, &alpha, Hks+ik*norbs*norbs, norbs, Wk, norbs, &beta, _HkWave, norbs);
            // 2. HkWave = W^dagger @ _HkWave
            for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)HkWave[iorb*norbs+jorb]=0;
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, norbs, norbs, norbs, &alpha, Wk, norbs, _HkWave, norbs, &beta, HkWave, norbs);
                
            for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
                HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb])/sqrt(Oklam[jorb]);

            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,HkWave,norbs,eigs);
            if(info!=0){printf("LAPACKE failed for the eigen. eq. for HkWave, info: %d\n",info);exit(1);}
            // Calc. the normalized eigen-vecs. V=W @ 1/sqrt(Olam) @ U
            // 1. HkWave = 1/sqrt(Olam) @ U
            for(iorb=0;iorb<norbs;iorb++){
                spectra[ik*norbs+iorb]=eigs[iorb];
                for(jorb=0;jorb<norbs;jorb++){
                   HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb]);
                }
            }
            // 2. V=W @ HkWave
            for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)Hks[ik*norbs*norbs+iorb*norbs+jorb]=0;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, norbs, norbs, norbs, &alpha, Wk, norbs, HkWave, norbs, &beta, Hks+ik*norbs*norbs, norbs);
        
            my_icnt+=1;
            if(my_icnt%my_nkpt_portion==0)output(">");
        }
        output("\n");
        free(HkWave);free(_HkWave);free(Wk);free(WkHalf);free(eigs);free(Oklam);
    }
    output("free all inputs...\n");
    free(sourceID);free(targetID);free(overLat);free(hopping_real);free(hopping_imag);free(overlap);
    
    sprintf(output_info,"Hk constructed in %.3fs\n",(double)(clock()-time0)/CLOCKS_PER_SEC);output(output_info);
    long long eigs_addr=(long long)spectra;
    long long vecs_addr=(long long)Hks;
    sprintf(output_info,"\nin eig.pyd the addr. of eigs, vecs, hk and ok are %p, %p\n",eigs_addr,vecs_addr);output(output_info);
    return Py_BuildValue("(LL)",eigs_addr,vecs_addr);
}

PyObject * calcFullBandsAndVecsInMemory_OVLCorrection_mixFFT(PyObject* self, PyObject* args){
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    fclose(fp);
    // read in all parameters
    PyObject* py_kgrid;
    PyObject* py_rgrid;
    PyObject* py_qgrid;
    PyObject* py_norbs;
    PyObject* py_nhoppings;
    PyObject* py_sourceID;
    PyObject* py_targetID;
    PyObject* py_overLat;
    PyObject* py_treal;
    PyObject* py_timag;
    PyObject* py_overlap;
    PyObject* py_ispin;
    PyObject* py_nthreads;
    PyObject* callback;  // callback function
    printf("start parsing...\n");
    PyArg_ParseTuple(args,"OOOOOOOOOOOOOO",
        &py_kgrid,&py_rgrid,&py_qgrid,&py_norbs,&py_nhoppings,&py_sourceID,&py_targetID,
        &py_overLat,
        &py_treal,&py_timag,&py_overlap,&py_ispin,
        &py_nthreads,&callback);
    uint ispin=(uint)PyLong_AS_LONG(py_ispin);
    uint kgrid[3];
    uint rgrid[3];
    uint qgrid[3];
    for(uint icomp=0;icomp<3;icomp++){
        kgrid[icomp]=(uint)PyLong_AS_LONG(PyList_GetItem(py_kgrid,icomp));
        rgrid[icomp]=(uint)PyLong_AS_LONG(PyList_GetItem(py_rgrid,icomp));
        qgrid[icomp]=(uint)PyLong_AS_LONG(PyList_GetItem(py_qgrid,icomp));
    }
    uint nKgrid=kgrid[0]*kgrid[1]*kgrid[2];
    uint nRgrid=rgrid[0]*rgrid[1]*rgrid[2];
    uint nQgrid=qgrid[0]*qgrid[1]*qgrid[2];
    uint norbs=(uint)PyLong_AS_LONG(py_norbs);
    uint nhoppings=(uint)PyLong_AS_LONG(py_nhoppings);
    printf("nkpt=%d nrpt=%d nqpt=%d norbs=%d ispin=%d nhoppings=%d\n",nKgrid,nRgrid,nQgrid,norbs,ispin,nhoppings);
    uint nthreads=(uint)PyLong_AS_LONG(py_nthreads);
    if(2*nQgrid<nthreads)nthreads=1;
    printf("Solving %d slow girds, using %d threads\n",nQgrid,nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"use %d cores in eig.pyd::calcFullBandsAndVecsInMemory_OVLCorrection_mixFFT\n",nthreads);
    fclose(fp);

    uint *sourceID=(uint*)malloc(nhoppings*sizeof(uint));
    uint *targetID=(uint*)malloc(nhoppings*sizeof(uint));
    int *overLat=(int*)malloc(nhoppings*3*sizeof(int));
    double*hopping_real=(double*)malloc(nhoppings*sizeof(double));
    double*hopping_imag=(double*)malloc(nhoppings*sizeof(double));
    double*overlap=(double*)malloc(nhoppings*sizeof(double));
    for(uint ihopping=0;ihopping<nhoppings;ihopping++){
        sourceID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_sourceID,ihopping));
        targetID[ihopping]=(uint)PyLong_AS_LONG(PyList_GetItem(py_targetID,ihopping));
        for(uint idir=0;idir<3;idir++)
            overLat[ihopping*3+idir]=PyLong_AS_LONG(PyList_GetItem(py_overLat,ihopping*3+idir));
        hopping_real[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_treal,ihopping));
        hopping_imag[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_timag,ihopping));
        overlap[ihopping]=PyFloat_AsDouble(PyList_GetItem(py_overlap,ihopping));
    }
    printf("Args. parsing finished.\n");
    // create hopping on slow grids
    clock_t time0=clock();
    double * spectra=(double*)malloc((unsigned long long)nKgrid*norbs*sizeof(double));                               // eigen values
    lapack_complex_double *tk_list=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double)); // eigen vectors
    lapack_complex_double *hk=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double));      // energy matrix in k-space
    lapack_complex_double *hk_a=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double));    // hamiltonian factor matrix in k-space
    lapack_complex_double *ok=(lapack_complex_double*)malloc(nKgrid*norbs*norbs*sizeof(lapack_complex_double));      // overlap matrix in k-space
    
    // fftw_plan use global variables. So, we should create them outside the parallel loop
    fftw_complex *tR=(fftw_complex*)malloc(nthreads*nRgrid*sizeof(fftw_complex));
    fftw_complex *tk=(fftw_complex*)malloc(nthreads*nRgrid*sizeof(fftw_complex));
    fftw_plan*p_list  =(fftw_plan*)malloc(nthreads*sizeof(fftw_plan));
    fftw_complex *sR=(fftw_complex*)malloc(nthreads*nRgrid*sizeof(fftw_complex));
    fftw_complex *sk=(fftw_complex*)malloc(nthreads*nRgrid*sizeof(fftw_complex));
    fftw_plan*ps_list  =(fftw_plan*)malloc(nthreads*sizeof(fftw_plan));
    for(uint ithread=0;ithread<nthreads;ithread++){
        p_list[ithread]=fftw_plan_dft_3d(rgrid[0],rgrid[1],rgrid[2],tR+ithread*nRgrid,tk+ithread*nRgrid,FFTW_BACKWARD,FFTW_MEASURE);
        ps_list[ithread]=fftw_plan_dft_3d(rgrid[0],rgrid[1],rgrid[2],sR+ithread*nRgrid,sk+ithread*nRgrid,FFTW_BACKWARD,FFTW_MEASURE);
    }
    printf("FFTW ready.\n");
    unsigned long long index; // the index for all states includes kpts and bands
    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double _tq, _sq;
        lapack_complex_double phase;
        uint isource, itarget;
        int Rvec0, Rvec1, Rvec2; // Rvec can be negative!
        double d0, d1, d2, alpha, beta;
        uint iOrbPair, iR, iR1, iR2, iR3, iq, iq1, iq2, iq3, ik1, ik2, ik3, ik, iorb, jorb, icontract, jcontract;
        lapack_complex_double *tq=(lapack_complex_double*)malloc(norbs*norbs*nRgrid*sizeof(lapack_complex_double));
        lapack_complex_double *sq=(lapack_complex_double*)malloc(norbs*norbs*nRgrid*sizeof(lapack_complex_double));
        lapack_complex_double * _HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * HkWave =(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init Hk wave the one can be solved
        lapack_complex_double * Wk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double)); // init the eigen-vecs of Ok
        lapack_complex_double * WkHalf=(lapack_complex_double*)malloc((norbs/2)*(norbs/2)*sizeof(lapack_complex_double)); // one quarter the eigen-vecs of Ok for double spin case
        double * eigs=(double*)malloc(norbs*sizeof(double)); // eigenvalues of Hk
        double * Oklam=(double*)malloc(norbs*sizeof(double)); // eigenvalues of overlap
        alpha=1;beta=0; // for matrix multiply

        int myrank=omp_get_thread_num();
        uint my_nqpt_portion=nQgrid/nthreads;
        uint my_icnt=0;
        if(my_nqpt_portion==0)my_nqpt_portion=1;
        if(myrank==0){
            fp=fopen("./c_log","a");
            for(iq=0;iq<my_nqpt_portion;iq++){
                printf("_");
                fprintf(fp,"_");
            }
            printf("\n");
            fprintf(fp,"\n");
            fclose(fp);
        }
        #pragma omp for
        for(iq=0;iq<nQgrid;iq++){
            iq3=iq%qgrid[2];
            iq2=((iq-iq3)/qgrid[2])%qgrid[1];
            iq1=iq/qgrid[1]/qgrid[2];
            //if(myrank==0)printf("q=(%d %d %d)\n",iq1,iq2,iq3);
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                //if(myrank==0)printf("hopping%d\n",ihopping);
                Rvec0=overLat[ihopping*3+0];
                Rvec1=overLat[ihopping*3+1];
                Rvec2=overLat[ihopping*3+2];
                //if(abs(Rvec0)>=rgrid[0] | abs(Rvec1)>=rgrid[1] | abs(Rvec2)>=rgrid[2])continue; // skip redundant hopping
                //if(myrank==0)printf("Original Rvec: %d %d %d in rgird (%d %d %d)\n",Rvec0,Rvec1,Rvec2,rgrid[0],rgrid[1],rgrid[2]);
                if(Rvec0<0) Rvec0+=rgrid[0];
                if(Rvec1<0) Rvec1+=rgrid[1];
                if(Rvec2<0) Rvec2+=rgrid[2];
                //if(Rvec0<0 | Rvec1<0 | Rvec2<0)continue; // skip redundant hopping
                iR=Rvec0*rgrid[1]*rgrid[2]+Rvec1*rgrid[2]+Rvec2;

                isource=sourceID[ihopping];
                itarget=targetID[ihopping];
                iOrbPair=isource*norbs+itarget;
                //printf("Rvec: %d %d %d; iOrbPair: %d; iR: %d; index for tq and sq: %d\n",Rvec0,Rvec1,Rvec2,iOrbPair,iR,iOrbPair*nRgrid+iR);
                phase=cexp(_Complex_I*TwoPI*((double)iq1/kgrid[0]*overLat[ihopping*3+0]+(double)iq2/kgrid[1]*overLat[ihopping*3+1]+(double)iq3/kgrid[2]*overLat[ihopping*3+2]));
                //if(myrank==0)printf("phase: %.3f %.3f\n",creal(phase),cimag(phase));
                //if(myrank==0)printf("hopping: %.3f %.3f\n",creal(hopping_real[ihopping]),cimag(hopping_real[ihopping]));
                //if(myrank==0)printf("overlap: %.3f\n",overlap[ihopping]);
                //exit(1);
                /*
                if(iOrbPair*nRgrid+iR>=norbs*norbs*nRgrid | iOrbPair*nRgrid+iR<0){
                    printf("Error!\n");
                    printf("Rvec: %d %d %d; isource,itarget,iOrbPair: %d %d %d; iR: %d; index for tq and sq: %d\n",Rvec0,Rvec1,Rvec2,isource,itarget,iOrbPair,iR,iOrbPair*nRgrid+iR);
                    printf("norbs*norbs*nRgrid=%d\n",norbs*norbs*nRgrid);
                    exit(1);
                }*/
                tq[iOrbPair*nRgrid+iR]=phase*(hopping_real[ihopping]+_Complex_I*hopping_imag[ihopping]);
                sq[iOrbPair*nRgrid+iR]=phase*overlap[ihopping];
            }
            //printf("tq and sq ready\n");
            //exit(1);
            // compute the reciprocal energy matrix and overlap matrix by FFT
            for(iorb=0;iorb<norbs;iorb++){
                for(jorb=0;jorb<norbs;jorb++){
                    iOrbPair=iorb*norbs+jorb;
                    for(iR=0;iR<nRgrid;iR++){
                        tR[myrank*nRgrid+iR]=tq[iOrbPair*nRgrid+iR];
                        sR[myrank*nRgrid+iR]=sq[iOrbPair*nRgrid+iR];
                    }
                    //printf("\n");
                    fftw_execute(p_list[myrank]);
                    fftw_execute(ps_list[myrank]);
                    for(iR1=0;iR1<rgrid[0];iR1++){
                    for(iR2=0;iR2<rgrid[1];iR2++){
                    for(iR3=0;iR3<rgrid[2];iR3++){
                        ik1=iR1*qgrid[0]+iq1;
                        ik2=iR2*qgrid[1]+iq2;
                        ik3=iR3*qgrid[2]+iq3;
                        ik=ik1+ik2*kgrid[0]+ik3*kgrid[0]*kgrid[1]; // ik1 run fastest
                        iR=iR1*rgrid[1]*rgrid[2]+iR2*rgrid[2]+iR3;
                        //if(myrank==0)printf("iq: %d %d %d iR: %d %d %d ik: %d %d %d\n",iq1,iq2,iq3,iR1,iR2,iR3,ik1,ik2,ik3);

                        tk_list[ik*norbs*norbs+iOrbPair]=tk[myrank*nRgrid+iR];
                        hk[ik*norbs*norbs+iOrbPair]=tk[myrank*nRgrid+iR];
                        ok[ik*norbs*norbs+iOrbPair]=sk[myrank*nRgrid+iR];
                    }
                    }
                    }
                }
            }
            //printf("Hk and Ok for all sub kpts of iq=%d are set.\n",iq);
            //exit(1);
            for(iR1=0;iR1<rgrid[0];iR1++){
            for(iR2=0;iR2<rgrid[1];iR2++){
            for(iR3=0;iR3<rgrid[2];iR3++){
                ik1=iR1*qgrid[0]+iq1;
                ik2=iR2*qgrid[1]+iq2;
                ik3=iR3*qgrid[2]+iq3;
                ik=ik1+ik2*kgrid[0]+ik3*kgrid[0]*kgrid[1]; // ik1 run fastest
                //printf("ik %d\n",ik);

                if(ispin==2){ // only one quarter overlap will be solved, and copy to the other diagonal region
                    for(iorb=0;iorb<norbs/2;iorb++)for(jorb=0;jorb<norbs/2;jorb++)
                       WkHalf[iorb*(norbs/2)+jorb]=ok[ik*norbs*norbs+iorb*norbs+jorb];
                    int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs/2,WkHalf,norbs/2,Oklam);
                    if(info!=0)printf("LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);
                    // Set full ovlk
                    for(iorb=0;iorb<norbs/2;iorb++){
                        Oklam[norbs/2+iorb]=Oklam[iorb];
                        for(jorb=0;jorb<norbs/2;jorb++){
                            Wk[          iorb*norbs+          jorb]=WkHalf[iorb*(norbs/2)+jorb];
                            Wk[(iorb+norbs/2)*norbs+(jorb+norbs/2)]=WkHalf[iorb*(norbs/2)+jorb];
                            Wk[(iorb+norbs/2)*norbs          +jorb]=0;
                            Wk[          iorb*norbs+(jorb+norbs/2)]=0;
                        }
                    }
                }else{ // solve the whole overlap matrix
                    for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
                       Wk[iorb*norbs+jorb]=ok[ik*norbs*norbs+iorb*norbs+jorb];
                    int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,Wk,norbs,Oklam);
                    if(info!=0)printf("LAPACKE failed for the eigen. eq. for ovlk, info: %d\n",info);
                }
                // to check the positivity of overlap eigenvals
                if(Oklam[0]<=0){
                    printf("Error, eigenvalues of overlap is not positive:");
                    for(uint ieig=0;ieig<norbs;ieig++)printf(" %.3f",Oklam[ieig]);
                    printf("\n");
                    exit(1);
                }
                
                // HkWave= 1/sqrt(Olam) @ W^dagger @ Hk @ W @ 1/sqrt(Olam)
                // 1. _HkWave = Hk @ W
                for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)_HkWave[ik*norbs*norbs+iorb*norbs+jorb]=0;
                cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, norbs, norbs, &alpha, tk_list+ik*norbs*norbs, norbs, Wk, norbs, &beta, _HkWave, norbs);
                // 2. HkWave = W^dagger @ _HkWave
                for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)HkWave[iorb*norbs+jorb]=0;
                cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, norbs, norbs, norbs, &alpha, Wk, norbs, _HkWave, norbs, &beta, HkWave, norbs);
                
                for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
                    HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb])/sqrt(Oklam[jorb]);

                int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,HkWave,norbs,eigs);
                if(info!=0){printf("LAPACKE failed for the eigen. eq. for HkWave, info: %d\n",info);exit(1);}
                // Calc. the normalized eigen-vecs. V=W @ 1/sqrt(Olam) @ U
                // 1. HkWave = 1/sqrt(Olam) @ U
                for(iorb=0;iorb<norbs;iorb++){
                    spectra[ik*norbs+iorb]=eigs[iorb];
                    for(jorb=0;jorb<norbs;jorb++){
                       HkWave[iorb*norbs+jorb]=HkWave[iorb*norbs+jorb]/sqrt(Oklam[iorb]);
                       //tk_list[ik*norbs*norbs+iorb*norbs+jorb]=0;
                   }
                }
                // 2. V=W @ HkWave
                for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)tk_list[ik*norbs*norbs+iorb*norbs+jorb]=0;
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, norbs, norbs, norbs, &alpha, Wk, norbs, HkWave, norbs, &beta, tk_list+ik*norbs*norbs, norbs);
            }
            }
            }
            if(myrank==0){
                printf(">");
                fp=fopen("./c_log","a");
                fprintf(fp,">");
                fclose(fp);
            }

        }
        if(myrank==0)printf("\n");
        //printf("thread%d finished\n",myrank);
        free(HkWave);free(_HkWave);free(tq);free(sq);free(Wk);free(WkHalf);free(eigs);free(Oklam);
    }
    printf("free all inputs...\n");
    free(sourceID);free(targetID);free(overLat);free(hopping_real);free(hopping_imag);free(overlap);
    printf("stop FFTW...\n");
    for(int ithread=0;ithread<nthreads;ithread++){
        fftw_destroy_plan(p_list[ithread]);
        fftw_destroy_plan(ps_list[ithread]);
    }
    free(p_list);free(ps_list);
    free(tk);free(tR);
    printf("Hk constructed in %.3fs\n",(double)(clock()-time0)/CLOCKS_PER_SEC);
    long long eigs_addr=(long long)spectra;
    long long vecs_addr=(long long)tk_list;
    long long hk_addr=(long long)hk;
    long long ok_addr=(long long)ok;
    printf("\nin ham.dll the addr. of eigs, vecs, hk and ok are %p, %p, %p, %p\n",eigs_addr,vecs_addr,hk_addr,ok_addr);
    return Py_BuildValue("(LLLL)",eigs_addr,vecs_addr,hk_addr,ok_addr);
}

static PyMethodDef module_methods[] = {
    {"calcFullBandsAndVecs", calcFullBandsAndVecs, METH_VARARGS, "calc. the band energy and eigenvectors via directly solving the eigen eq., using phase factor"},
    {"calcFullBandsAndVecs_OVLCorrection_DoubelSpin", calcFullBandsAndVecs_OVLCorrection_DoubelSpin, METH_VARARGS, "calc. the band energy and eigenvectors via solving the eigen eq. with overlap corrections, using phase factor"},
    {"calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin", calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin, METH_VARARGS, "calc. the band energy and eigenvectors via solving the eigen eq. with overlap corrections, using phase factor"},
    {"calcFullBandsAndVecsInMemory_OVLCorrection_FFT",calcFullBandsAndVecsInMemory_OVLCorrection_FFT,METH_VARARGS,"compute eigen-values and vectors by solving generalized eigen-eqs. in reciprocal with FFTW"},
    {"calcFullBandsAndVecsInMemory_OVLCorrection_mixFFT", calcFullBandsAndVecsInMemory_OVLCorrection_mixFFT, METH_VARARGS, "compute eigen-values, vectors, energy matrix and overlap matrix in reciprocal"},
    {NULL, NULL, 0, NULL} // neccessary to tell python compiler stop here
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "eig",
    "To solve the eigen-equation to obtain spectra and eigen-vectors",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_eig(void) {
    fp=fopen("./eig.log","w");
    output("Initializing lib of eig...\n");
    PyObject* m;
    m= PyModule_Create(&moduledef);
    return m;
}