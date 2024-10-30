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
PyObject*callback_global;
int output(const char* s){
    int myrank=omp_get_thread_num();
    if(myrank==0){
        fp=fopen("./eig.log","a");
        fprintf(fp,s);
        fclose(fp);
    }
    return 0;
}
int output_py(const char* s){
    int myrank=omp_get_thread_num();
    if(myrank==0){
        fp=fopen("./eig.log","a");
        fprintf(fp,s);
        fclose(fp);
        PyObject_CallObject(callback_global, Py_BuildValue("(s)",s));
    }
    return 0;
}

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

PyObject * calcRetardedLatticeGreensFunction(PyObject* self, PyObject* args){
    clock_t time0=clock();
    PyObject* py_eigs_addr;
    PyObject* py_vecs_addr;
    PyObject* py_kgrid;
    PyObject* py_nTotalOrbs;
    PyObject* py_norbs;
    PyObject* py_orbIDMap;
    PyObject* py_nOmegaPoints;
    PyObject* py_omega_real;
    PyObject* py_omega_imag;
    PyObject* py_eta;
    PyObject* py_nLatticeVectors;
    PyObject* py_latVec;
    PyObject* py_spinor_tag;
    PyObject* py_nthreads;
    //PyObject* callback;  // callback function
    output("Args parsing into calcRetardedLatticeGreensFunction\n");
    char output_info[512];
    PyArg_ParseTuple(args,"OOOOOOOOOOOOOOO",
        &py_eigs_addr,&py_vecs_addr,
        &py_kgrid,&py_nTotalOrbs,&py_norbs,&py_orbIDMap,&py_nOmegaPoints,&py_omega_real,&py_omega_imag,
        &py_eta,&py_nLatticeVectors,&py_latVec,&py_spinor_tag,
        &py_nthreads,&callback_global);
    long long eigs_addr=PyLong_AsLongLong(py_eigs_addr);
    long long vecs_addr=PyLong_AsLongLong(py_vecs_addr);
    sprintf(output_info,"Obtain eigs and vecs @ %p, %p\n",eigs_addr,vecs_addr);output_py(output_info);
    uint kgrid[3];
    for(uint icomp=0;icomp<3;icomp++){
        kgrid[icomp]=(uint)PyLong_AsLong(PyList_GetItem(py_kgrid,icomp));
    }
    uint nTotalOrbs=(uint)PyLong_AsLong(py_nTotalOrbs);
    uint norbs=(uint)PyLong_AsLong(py_norbs);
    uint orbIDMap[norbs];
    for(uint iorb=0;iorb<norbs;iorb++)orbIDMap[iorb]=(uint)PyLong_AsLong(PyList_GetItem(py_orbIDMap,iorb));
    uint nOmegaPoints=(uint)PyLong_AsLong(py_nOmegaPoints);
    double omega_real[nOmegaPoints];
    double omega_imag[nOmegaPoints];
    for(uint iomega=0;iomega<nOmegaPoints;iomega++){
        omega_real[iomega]=PyFloat_AsDouble(PyList_GetItem(py_omega_real,iomega));
        omega_imag[iomega]=PyFloat_AsDouble(PyList_GetItem(py_omega_imag,iomega));
    }
    double eta=PyFloat_AsDouble(py_eta);
    uint nLatticeVectors=(uint)PyLong_AsLong(py_nLatticeVectors);
    int latVec[nLatticeVectors][3];
    for(uint ilat=0;ilat<nLatticeVectors;ilat++)for(uint icomp=0;icomp<3;icomp++)
        latVec[ilat][icomp]=PyLong_AsLong(PyList_GetItem(py_latVec,ilat*3+icomp));
    uint spinor_tag=(uint)PyLong_AsLong(py_spinor_tag);
    uint _nthreads=(uint)PyLong_AsLong(py_nthreads);

    double*eigs=(double*)eigs_addr;
    lapack_complex_double*vecs=(lapack_complex_double*)vecs_addr;
    
    long long nkpt=kgrid[0]*kgrid[1]*kgrid[2];
    long long kxy=kgrid[0]*kgrid[1];
    lapack_complex_double *k1_axis=(lapack_complex_double*)malloc(kgrid[0]*sizeof(lapack_complex_double));
    lapack_complex_double *k2_axis=(lapack_complex_double*)malloc(kgrid[1]*sizeof(lapack_complex_double));
    lapack_complex_double *k3_axis=(lapack_complex_double*)malloc(kgrid[2]*sizeof(lapack_complex_double));
    constructKAxis(kgrid[0],kgrid[1],kgrid[2],k1_axis,k2_axis,k3_axis);
    
    uint nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads)nthreads=nkpt;
    sprintf(output_info,"nkpt %d, nOmega %d, nLatticeVectors %d, norbs %d, nTotalOrbs %d\n",nkpt,nOmegaPoints,nLatticeVectors,norbs,nTotalOrbs);output_py(output_info);
    sprintf(output_info,"use %d cores\n",nthreads);output_py(output_info);
    ulong latVec_len=norbs*norbs;
    ulong omega_len=nLatticeVectors*latVec_len;
    ulong thread_len=nOmegaPoints*omega_len;
    sprintf(output_info,"estimate memory cost: %d MB\n",
                    (nOmegaPoints*nLatticeVectors*norbs*norbs*sizeof(lapack_complex_double)+
                    nthreads*thread_len*sizeof(lapack_complex_double)+
                    nkpt*nOmegaPoints*nTotalOrbs*sizeof(lapack_complex_double)+
                    nkpt*nTotalOrbs*nTotalOrbs*sizeof(lapack_complex_double)+
                    nkpt*nLatticeVectors*sizeof(lapack_complex_double))/1024/1024
    );
    output_py(output_info);
    // the lattice retarded Green's function
    lapack_complex_double * G_w_lat=(lapack_complex_double*)malloc(nOmegaPoints*nLatticeVectors*norbs*norbs*sizeof(lapack_complex_double));
    // partially BZ integrated retarded Green's function (only integrated in each thread)
    lapack_complex_double * _G_w_lat=(lapack_complex_double*)malloc(nthreads*thread_len*sizeof(lapack_complex_double));
    // inversed Hk (in eigen rep.)
    lapack_complex_double *inversed_hks=(lapack_complex_double*)malloc(nkpt*nOmegaPoints*nTotalOrbs*sizeof(lapack_complex_double));
    // inversed vecs (in LCAO rep.)
    lapack_complex_double *inversed_vecs=(lapack_complex_double*)malloc(nkpt*nTotalOrbs*nTotalOrbs*sizeof(lapack_complex_double));
    // lattice Fourier basis
    lapack_complex_double *lat_factors=(lapack_complex_double*)malloc(nkpt*nLatticeVectors*sizeof(lapack_complex_double));
    #pragma omp parallel num_threads(nthreads)
    {
        uint my_rank=omp_get_thread_num();
        ulong i_rank=my_rank*thread_len;
        ulong iomega, ieig;
        ulong i_omega_k;
        ulong i_eig_k, i_kpt_base;
        ulong ikpt, ik1, ik2, ik3;
        ulong ilat, iorb, jorb;
        int info;
#ifdef MKL_ILP64
        long long *ipiv=(long long*)malloc(nTotalOrbs*sizeof(long long));
#else
        int *ipiv=(int*)malloc(nTotalOrbs*sizeof(int));
#endif
        //printf("thread %d initialized %.6f\n",my_rank,creal(_G_w_lat[my_rank*thread_len]));
        // initial G_w_lat
        for(iomega=0;iomega<nOmegaPoints;iomega++){
            for(ilat=0;ilat<nLatticeVectors;ilat++){
                for(iorb=0;iorb<norbs;iorb++){
                    for(jorb=0;jorb<norbs;jorb++){
                        _G_w_lat[i_rank+iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb]=0;
                    }
                }
            }
        }
        #pragma omp for
        for(ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            // ikpt = ik1 + ik2*kgrid[0] + ik3*kgrid[0]*kgrid[1]
            ik1=ikpt%kgrid[0];
            ik2=(ikpt-ik1)%kxy/kgrid[0];
            ik3=ikpt/kxy;
            // compute lattice fourier components
            for(ilat=0;ilat<nLatticeVectors;ilat++){
                lat_factors[ikpt*nLatticeVectors+ilat]=k1_axis[mod(-ik1*latVec[ilat][0],kgrid[0])]
                                            *k2_axis[mod(-ik2*latVec[ilat][1],kgrid[1])]
                                            *k3_axis[mod(-ik3*latVec[ilat][2],kgrid[2])];
            }
            // compute inversed hks
            i_eig_k=ikpt*nTotalOrbs;
            i_kpt_base=ikpt*nOmegaPoints*nTotalOrbs;
            for(iomega=0;iomega<nOmegaPoints;iomega++){
                lapack_complex_double omega=omega_real[iomega]+_Complex_I*omega_imag[iomega];
                i_omega_k=iomega*nTotalOrbs;
                for(ieig=0;ieig<nTotalOrbs;ieig++){
                    inversed_hks[i_kpt_base+i_omega_k+ieig]=1/(omega-eigs[i_eig_k+ieig]+_Complex_I*eta);
                }
            }
            // compute inversed vec
            for(iorb=0;iorb<nTotalOrbs;iorb++)for(jorb=0;jorb<nTotalOrbs;jorb++)
                inversed_vecs[ikpt*nTotalOrbs*nTotalOrbs+iorb*nTotalOrbs+jorb]=vecs[ikpt*nTotalOrbs*nTotalOrbs+iorb*nTotalOrbs+jorb];
            info=LAPACKE_zgetrf(LAPACK_ROW_MAJOR,nTotalOrbs,nTotalOrbs,inversed_vecs+ikpt*nTotalOrbs*nTotalOrbs,nTotalOrbs,ipiv);
            if(info!=0){printf("Failed zgetrf (LU decomposition) info=%d\n",info);exit(1);}
            info=LAPACKE_zgetri(LAPACK_ROW_MAJOR,nTotalOrbs,inversed_vecs+ikpt*nTotalOrbs*nTotalOrbs,nTotalOrbs,ipiv);
            if(info!=0){printf("Failed zgetri (matrix inversion) info=%d\n",info);exit(1);}
        }
    }
    sprintf(output_info,"inversed hk, inversed vec and lattice Fourier basis constructed in %ds\n",(clock()-time0)/CLOCKS_PER_SEC);output_py(output_info);
    time0=clock();
    // calculate matrix elements
    //int index;
    #pragma omp parallel num_threads(nthreads)
    {
        uint my_rank=omp_get_thread_num();
        ulong i_rank=my_rank*thread_len;
        ulong iomega, ilat, ieig, iorb, jorb, ikpt;
        ulong i_omega_k, i_lat_k;
        ulong i_eig_k, i_vec_k;
        lapack_complex_double factor, inverse_h;
        double alpha, beta;
        alpha=1;beta=0;

        uint my_nkpt_portion=nkpt/nthreads/50;
        uint my_icnt=0;
        if(my_nkpt_portion==0)my_nkpt_portion=1;
        if(my_rank==0){sprintf(output_info,"> corresponds to ~%d kpts\n",my_nkpt_portion*50);output_py(output_info);}

        lapack_complex_double *g0_inversed_vecs=(lapack_complex_double*)malloc(nTotalOrbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double *Gw_k=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
        //lapack_complex_double *inversed_veck=(lapack_complex_double*)malloc(nTotalOrbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double *veck=(lapack_complex_double*)malloc(norbs*nTotalOrbs*sizeof(lapack_complex_double));
        //printf("thread %d initialized %.6f\n",my_rank,creal(_G_w_lat[my_rank*thread_len]));
        #pragma omp for // k-point parallel is assumed to exhibit linear scaling
        for(ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //i_eig_k=ikpt*norbs;
            i_vec_k=ikpt*nTotalOrbs*nTotalOrbs;
            // get out veck^-1
            for(ieig=0;ieig<nTotalOrbs;ieig++)for(iorb=0;iorb<norbs;iorb++)
                //inversed_veck[ieig*norbs+iorb]=inversed_vecs[i_vec_k+ieig*nTotalOrbs+orbIDMap[iorb]];
                veck[iorb*nTotalOrbs+ieig]=vecs[i_vec_k+orbIDMap[iorb]*nTotalOrbs+ieig];
            for(ilat=0;ilat<nLatticeVectors;ilat++){
                //printf("rank%d, ikpt*nLatticeVectors+ilat %d\n",my_rank,ikpt*nLatticeVectors+ilat);
                factor=lat_factors[ikpt*nLatticeVectors+ilat];
                i_lat_k=ilat*latVec_len;
                for(iomega=0;iomega<nOmegaPoints;iomega++){
                    i_omega_k=iomega*omega_len;
                    // Gw_k = veck @ Gw_lam @ veck.conj().T
                    // 1. g0_inversed_vecs = Gw_lam @ vec.conj().T
                    //cblas_ztrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, Cblas)
                    for(ieig=0;ieig<nTotalOrbs;ieig++)for(iorb=0;iorb<norbs;iorb++)
                        //g0_inversed_vecs[ieig*norbs+iorb]=inversed_hks[ikpt*nOmegaPoints*nTotalOrbs+iomega*nTotalOrbs+ieig]*inversed_veck[ieig*norbs+iorb];
                        g0_inversed_vecs[ieig*norbs+iorb]=inversed_hks[ikpt*nOmegaPoints*nTotalOrbs+iomega*nTotalOrbs+ieig]*conj(veck[iorb*nTotalOrbs+ieig]);
                    // 2. Gw_k = veck @ g0_inversed_vecs
                    // it is essential to initialize Gw_k
                    for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)Gw_k[iorb*norbs+jorb]=0;
                    //cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, norbs, norbs, nTotalOrbs, &alpha, inversed_veck, norbs, g0_inversed_vecs, norbs, &beta, Gw_k, norbs);
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, norbs, norbs, nTotalOrbs, &alpha, veck, nTotalOrbs, g0_inversed_vecs, norbs, &beta, Gw_k, norbs);
                    
                    // inversed Fourier transformations to real-space
                    for(iorb=0;iorb<norbs;iorb++)for(jorb=0;jorb<norbs;jorb++)
                        _G_w_lat[i_rank+i_omega_k+i_lat_k+iorb*norbs+jorb]+=factor*Gw_k[iorb*norbs+jorb];
                }
            }
            my_icnt+=1;
            if(my_icnt%my_nkpt_portion==0)output_py(">");
        }
        free(g0_inversed_vecs);free(Gw_k);free(veck);
    }
    sprintf(output_info,"\nLattice G constructed in %ds\n",(clock()-time0)/CLOCKS_PER_SEC);output_py(output_info);
    
    ulong id_of_combine, id_of_sub_thread;
    for(uint iomega=0;iomega<nOmegaPoints;iomega++){
        for(uint ilat=0;ilat<nLatticeVectors;ilat++){
            for(uint iorb=0;iorb<norbs;iorb++){
                for(uint jorb=0;jorb<norbs;jorb++){
                    id_of_combine=iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb;
                    G_w_lat[iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb]=0;
                    for(uint ithread=0;ithread<nthreads;ithread++){
                        id_of_sub_thread=ithread*thread_len+id_of_combine;
                        G_w_lat[iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb]+=_G_w_lat[ithread*thread_len+iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb];
                    }
                }
            }
        }
    }
    free(_G_w_lat);free(inversed_hks);free(inversed_vecs);free(lat_factors);free(k1_axis);free(k2_axis);free(k3_axis);
    
    PyObject * GreenFunc=PyTuple_New(2*thread_len);
    for(uint iomega=0;iomega<nOmegaPoints;iomega++){
        for(uint ilat=0;ilat<nLatticeVectors;ilat++){
            for(uint iorb=0;iorb<norbs;iorb++){
                for(uint jorb=0;jorb<norbs;jorb++){
                    PyTuple_SetItem(GreenFunc, 0*thread_len+iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb, PyFloat_FromDouble(creal(G_w_lat[iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb]/nkpt)));
                    PyTuple_SetItem(GreenFunc, 1*thread_len+iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb, PyFloat_FromDouble(cimag(G_w_lat[iomega*omega_len+ilat*latVec_len+iorb*norbs+jorb]/nkpt)));
                }
            }
        }
    }
    free(G_w_lat);
    return Py_BuildValue("(O)",GreenFunc);
}

static PyMethodDef module_methods[] = {
    {"calcRetardedLatticeGreensFunction", calcRetardedLatticeGreensFunction, METH_VARARGS, "calc. the retarded lattice green's funciton using FFT"},
    {NULL, NULL, 0, NULL} // neccessary to tell python compiler stop here
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "green",
    "To solve the green's function",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_green(void) {
    fp=fopen("./green.log","w");
    output("Initializing lib of green...\n");
    PyObject* m;
    m= PyModule_Create(&moduledef);
    return m;
}