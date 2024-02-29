#define PY_SSIZE_T_CLEAN
#include<stdio.h>
#include<complex.h>
#include<Python.h>
#include<lapacke.h>
#include<time.h>
#include<fftw3.h>

#define TwoPI 6.28318530717958647692
#define PI 3.14159265358979323846
typedef unsigned int uint;

int omp_get_thread_num();

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

static PyMethodDef module_methods[] = {
    {"calcFullBandsAndVecs", calcFullBandsAndVecs, METH_VARARGS, "calc. the band energy and eigenvectors via directly solving the eigen eq., using phase factor"},
    {"calcFullBandsAndVecs_OVLCorrection_DoubelSpin", calcFullBandsAndVecs_OVLCorrection_DoubelSpin, METH_VARARGS, "calc. the band energy and eigenvectors via solving the eigen eq. with overlap corrections, using phase factor"},
    {"calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin", calcFullBandsAndVecsInMemory_OVLCorrection_DoubelSpin, METH_VARARGS, "calc. the band energy and eigenvectors via solving the eigen eq. with overlap corrections, using phase factor"},
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
    printf("Initializing lib of eig...\n");
    PyObject* m;
    m= PyModule_Create(&moduledef);
    return m;
}