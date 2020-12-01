#include <stdio.h>
#include <complex.h>
#include <lapacke.h>
#include <Python.h>
#include <omp.h>

#define TwoPI 6.28318530717958647692

lapack_complex_double * create_zero_complex_matrix(int order){
    lapack_complex_double *matrix=(lapack_complex_double*)malloc(order*order*sizeof(lapack_complex_double));
    for(int i=0;i<order;i++) for (int j=0;j<order;j++) matrix[i*order+j]=0;
    return matrix;
}

lapack_complex_double * create_constant_complex_matrix(int order, lapack_complex_double element){
    lapack_complex_double *matrix=create_zero_complex_matrix(order);
    for(int i=0;i<order;i++) matrix[i*order+i]=element;
    return matrix;
}

lapack_complex_double * conventional_tensor_contraction(int lda, int nrhs, int ldb,
                                                        lapack_complex_double *leftTensor,
                                                        lapack_complex_double *rightTensor
                                                        ){
    lapack_complex_double *matrix=(lapack_complex_double*)malloc(lda*ldb*sizeof(lapack_complex_double));
    for(int i=0;i<lda;i++){
        for(int j=0;j<ldb;j++){
            *(matrix+i*ldb+j)=0;
            for(int k=0;k<nrhs;k++) *(matrix+i*ldb+j)+=*(leftTensor+i*nrhs+k)*(*(rightTensor+k*ldb+j));
        }
    }
    return matrix;
}



PyObject * getEOnCircle(int nthreads,
                        double theta, double r, double o, double eta,
                        int nkpt, double kmesh[nkpt][3], 
                        int norb, double HkList[nkpt][norb*norb*2],
                        int scatteringOrder, 
                        int relevantOrbR[scatteringOrder][3], int relevantOrbIndex[scatteringOrder],
                        double scatteringDoubleCenterOrigin[scatteringOrder*scatteringOrder*2],
                        double scatteringSingleCenterOrigin[scatteringOrder*scatteringOrder*2])
{
    lapack_complex_double z=r*cexp(theta*_Complex_I);
    lapack_complex_double E=z+o;

    lapack_complex_double* g_internal=(lapack_complex_double*)malloc(scatteringOrder*scatteringOrder*sizeof(lapack_complex_double));
    lapack_complex_double* g_external_contracted=(lapack_complex_double*)malloc(scatteringOrder*scatteringOrder*sizeof(lapack_complex_double));
    lapack_complex_double* scatteringDoubleCenter=(lapack_complex_double*)malloc(scatteringOrder*scatteringOrder*sizeof(lapack_complex_double));
    lapack_complex_double* scatteringSingleCenter=(lapack_complex_double*)malloc(scatteringOrder*scatteringOrder*sizeof(lapack_complex_double));
    
    // initialize green's functions and scattering matrix
    for(int i=0;i<scatteringOrder;i++){
        for(int j=0;j<scatteringOrder;j++){
            *(g_internal+i*scatteringOrder+j)=0;
            *(g_external_contracted+i*scatteringOrder+j)=0;
            *(scatteringDoubleCenter+i*scatteringOrder+j)=*(scatteringDoubleCenterOrigin+i*scatteringOrder*2+j*2)+(*(scatteringDoubleCenterOrigin+i*scatteringOrder*2+j*2+1))*_Complex_I;
            *(scatteringSingleCenter+i*scatteringOrder+j)=*(scatteringSingleCenterOrigin+i*scatteringOrder*2+j*2)+(*(scatteringSingleCenterOrigin+i*scatteringOrder*2+j*2+1))*_Complex_I;
        }
    }

    // loop for BZ integration
    //printf("total threads in use: %d\n",nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        //printf("start working thread id: %d\n",omp_get_thread_num());
        // private variables for each thread itself
        lapack_complex_double g_internal_private[scatteringOrder][scatteringOrder];
        lapack_complex_double g_external_contracted_private[scatteringOrder][scatteringOrder];
        for(int i=0;i<scatteringOrder;i++){
            for(int j=0;j<scatteringOrder;j++){
                g_internal_private[i][j]=0;
                g_external_contracted_private[i][j]=0;
            }
        }
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            //printf("kpt%d thread id: %d\n",ikpt, omp_get_thread_num());
            int i,j,info;
            //printf("kpt%d\n",ikpt);
            double *kpt=kmesh[ikpt];
            //printf("kpt (%.2f %.2f %.2f)\n",*kpt,*(kpt+1),*(kpt+2));
            //printf("Hk:\n");
            double *hk=HkList[ikpt];
            /*for(i=0;i<norb;i++){
                for(j=0;j<norb;j++) printf("(%.6f, %.6f) ",*(hk+i*norb*2+j*2),*(hk+i*norb*2+j*2+1));
                printf("\n");
            }*/
            // calculate gk using LAPACK
            lapack_complex_double gk[norb][norb];
            lapack_complex_double coreMatrix[norb][norb];
            for(i=0;i<norb;i++){
                for(j=0;j<norb;j++){
                    coreMatrix[i][j]=-*(hk+i*norb*2+j*2)-*(hk+i*norb*2+j*2+1)*_Complex_I;
                    gk[i][j]=0;
                    if(i==j){
                        coreMatrix[i][j]+=E+eta*_Complex_I;
                        gk[i][j]=1;
                    }
                }
            }
            int ipiv[norb];
            info=LAPACKE_zgesv(LAPACK_ROW_MAJOR,norb,norb,*coreMatrix,norb,ipiv,*gk,norb);
            /*printf("Gk:\n");
            for(i=0;i<4;i++){
                for(j=0;j<4;j++) printf("(%.6f, %.6f) ",creal(gk[i][j]),cimag(gk[i][j]));
                printf("\n");
            }*/
        
            /*printf("g_internal before step%d\n",ikpt);
            for(int irow=0;irow<scatteringOrder;irow++){
                for(int icol=0;icol<4;icol++) printf("(%.6f, %.6f) ",creal(*(g_internal+irow*scatteringOrder+icol)),cimag(*(g_internal+irow*scatteringOrder+icol)));
                printf("\n");
            }
            printf("g_external before step%d\n",ikpt);
            for(int irow=0;irow<scatteringOrder;irow++){
                for(int icol=0;icol<4;icol++) printf("(%.6f, %.6f) ",creal(*(g_external_contracted+irow*scatteringOrder+icol)),cimag(*(g_external_contracted+irow*scatteringOrder+icol)));
                printf("\n");
            }*/
        
            for(int irow=0;irow<scatteringOrder;irow++){ // loop for g_external contraction
                int orbIDLeft=relevantOrbIndex[irow];
                for(int icol=0;icol<scatteringOrder;icol++){
                    int orbIDRight=relevantOrbIndex[icol];
                    int deltaR[3];
                    for(i=0;i<3;i++) deltaR[i]=relevantOrbR[irow][i]-relevantOrbR[icol][i];
                    double kDotDR=0;
                    for(i=0;i<3;i++) kDotDR+=(deltaR[i]*(*(kpt+i)));
                    double phase=TwoPI*kDotDR;
                    double realPhase=cos(phase);
                    double imagPhase=sin(phase);

                    lapack_complex_double contracted_gk=0;
                    for(i=0;i<norb;i++)contracted_gk+=gk[orbIDRight][i]*gk[i][orbIDLeft];
                
                    g_internal_private[irow][icol]+=gk[orbIDLeft][orbIDRight]*(realPhase - imagPhase*_Complex_I)/nkpt;
                    g_external_contracted_private[irow][icol]+=contracted_gk*(realPhase + imagPhase*_Complex_I)/nkpt;
                    //*(g_internal+irow*scatteringOrder+icol)+=gk[orbIDLeft][orbIDRight]*(realPhase - imagPhase*_Complex_I)/nkpt;
                    //*(g_external_contracted+irow*scatteringOrder+icol)+=contracted_gk*(realPhase + imagPhase*_Complex_I)/nkpt;
                }
            }// end loop for g_external contraction
            //printf("kpt%d end thread id: %d\n",ikpt, omp_get_thread_num());
        }// end loop for BZ integration for each thread
        //printf("ready for end sum --> thread id: %d\n",omp_get_thread_num());
        // should be wary of the final summation (into global variables)
        #pragma omp critical
        {
            //printf("entering the end sum --> thread id: %d\n",omp_get_thread_num());
            //printf("%d \n",omp_get_thread_num());
            for(int icol=0;icol<scatteringOrder;icol++){
                for(int irow=0;irow<scatteringOrder;irow++){
                    //printf("(%.6f, %.6f) ",creal(g_internal_private[irow][icol]),cimag(g_internal_private[irow][icol]));
                    //*(g_external_contracted+irow*scatteringOrder+icol)+=g_external_contracted_private[irow][icol];
                    *(g_internal+irow*scatteringOrder+icol)+=g_internal_private[irow][icol];
                    *(g_external_contracted+irow*scatteringOrder+icol)+=g_external_contracted_private[irow][icol];
                }
            }
        }
    }// end of parrallization
    //printf("end of parrallization\n");
    //printf("contracted g external:\n");
    //for(i=0;i<scatteringOrder;i++){
    //  for(j=0;j<scatteringOrder;j++) printf("(%.6f, %.6f) ",creal(*(g_external_contracted+i*scatteringOrder+j)),cimag(*(g_external_contracted+i*scatteringOrder+j)));
    //  printf("\n");
    //}
    //printf("g internal:\n");
    //for(i=0;i<scatteringOrder;i++){
    //  for(j=0;j<scatteringOrder;j++) printf("(%.6f, %.6f) ",creal(*(g_internal+i*scatteringOrder+j)),cimag(*(g_internal+i*scatteringOrder+j)));
    //  printf("\n");
    //}
    //G(V1+V2)G(V1+V2)G
    
    /*printf("left tensor:\n");
    for(i=0;i<scatteringOrder;i++){
      for(j=0;j<4;j++) printf("(%.6f, %.6f) ",creal(*(g_external_contracted+i*scatteringOrder+j)),cimag(*(g_external_contracted+i*scatteringOrder+j)));
      printf("\n");
    }
    printf("right tensor:\n");
    for(i=0;i<scatteringOrder;i++){
      for(j=0;j<4;j++) printf("(%.6f, %.6f) ",creal(*(scatteringDoubleCenter+i*scatteringOrder+j)),cimag(*(scatteringDoubleCenter+i*scatteringOrder+j)));
      printf("\n");
    }*/
    lapack_complex_double*contract1=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    g_external_contracted,
                                                                    scatteringDoubleCenter);
    /*printf("result tensor:\n");
    for(i=0;i<scatteringOrder;i++){
      for(j=0;j<4;j++) printf("(%.6f, %.6f) ",creal(*(contract1+i*scatteringOrder+j)),cimag(*(contract1+i*scatteringOrder+j)));
      printf("\n");
    }*/
    lapack_complex_double*contract2=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    contract1,
                                                                    g_internal);
    lapack_complex_double*contract3=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    contract2,
                                                                    scatteringDoubleCenter);
    //G(V1)G(V1)G
    lapack_complex_double*contract4=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    g_external_contracted,
                                                                    scatteringSingleCenter);
    lapack_complex_double*contract5=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    contract4,
                                                                    g_internal);
    lapack_complex_double*contract6=conventional_tensor_contraction(scatteringOrder,scatteringOrder,scatteringOrder,
                                                                    contract5,
                                                                    scatteringSingleCenter);
    //trance
    lapack_complex_double delta_trace=0;
    for(int i=0;i<scatteringOrder;i++) delta_trace+=(*(contract3+i*scatteringOrder+i)-2*(*(contract6+i*scatteringOrder+i)));
    free(contract1);free(contract2);free(contract3);free(contract4);free(contract5);free(contract6);
    free(g_internal);free(g_external_contracted);
    lapack_complex_double delta_dos=-2*delta_trace/TwoPI;
    double dE_onCircle=cimag(E*delta_dos*z*_Complex_I);

    //printf("dE on circle=%.3f\n",dE_onCircle);
    PyObject * result=PyFloat_FromDouble(dE_onCircle);
    return result;
}