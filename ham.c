#include<stdio.h>
#include<complex.h>
#include<Python.h>
#include<lapacke.h>

#define TwoPI 6.28318530717958647692
#define PI 3.14159265358979323846

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
                       double hopping_imag[nhoppings],
                       int nthreads){//,
                       //int ikpt_print){
    double * kmesh=(double*)kmesh_address;
    printf("in ham.dll the addr. of kmesh is %p\n",kmesh);
    int numOfElements=nkpt*norbs*norbs;
    lapack_complex_double * Hk=(lapack_complex_double *)malloc(numOfElements*sizeof(lapack_complex_double));
    // initialize
    for(int i=0;i<numOfElements;i++) *(Hk+i)=0;

    // calculate matrix elements
    #pragma omp parallel num_threads(nthreads)
    {
        //lapack_complex_double Hk_private[numOfElements];
        //for(int i=0;i<numOfElements;i++) Hk_private[i]=0;
        
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int kBaseIndex=ikpt*norbs*norbs;
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*(*(kmesh+ikpt*3+i));
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
                *(Hk+kBaseIndex+sourceID[ihopping]*norbs+targetID[ihopping])+=tk;
            }
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

void quickSort(double* input_list, int low, int high){
    int l=low;
    int h=high;
    double split=*(input_list+low);
        
    while(l<h){
        while(*(input_list+h)>split) h-=1;
                
        if(l<h){
            *(input_list+l)=*(input_list+h);
            *(input_list+h)=split;
                
            //int sort_tmp=sort_list[l];
            //sort_list[l]=sort_list[h];
            //sort_list[h]=sort_tmp;
                
            l+=1;
        }
            
        while(*(input_list+l)<split) l+=1;
            
        if(l<h){
            *(input_list+h)=*(input_list+l);
            *(input_list+l)=split;
                
            //int sort_tmp=sort_list[l];
            //sort_list[l]=sort_list[h];
            //sort_list[h]=sort_tmp;
                
            h-=1;
        }
    }
    if(l>low) quickSort(input_list,low,h-1);
    if(h<high) quickSort(input_list,l+1,high);
    return;
}

PyObject * calcFullDOS(long long Hk_address, int nkpt, int norb, int NEDOS, double eta, double fermi, int nthreads){
    lapack_complex_double * HkList=(lapack_complex_double *)Hk_address;
    printf("calculating full DOS in ham.dll. addr. of Hk: %p\n",HkList);
    int totalEigs=nkpt*norb;
    double fullEigs[totalEigs];

    // parallel amongst kpoints for calculations of eigs
    #pragma omp parallel num_threads(nthreads)
    {   
        // start parallel
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            lapack_complex_double hk[norb][norb]; // copy Hk, since LAPACK transform it into eigvecs or destroy
            double eigs[norb]; // output eigenvalues of Hk
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    hk[i][j]=(*(HkList+ikpt*norb*norb+i*norb+j));
                }
            }
            
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'N','U',norb,*hk,norb,eigs);
            for(int i=0;i<norb;i++) fullEigs[ikpt*norb+i]=eigs[i]; // store eigvalues
        }
    }

    printf("sorting %d eigs...\n",totalEigs);
    quickSort(fullEigs,0,totalEigs-1);
    double emin=fullEigs[0];
    double emax=fullEigs[totalEigs-1];
    printf("band range: (%.6f, %.6f)\n",emin,emax);

    // suggest optimal Fermi level
    int ilevel=0;
    for(ilevel=0;ilevel<totalEigs;ilevel++){
        if(fullEigs[ilevel]>fermi) break;
    }
    double avgOccPerK=(double)ilevel/(double)nkpt;
    int avgOccPerK_Int=(int)avgOccPerK;
    double res=avgOccPerK-avgOccPerK_Int;
    int optimal_avgOccPerK=avgOccPerK_Int;
    if(res>0.5) optimal_avgOccPerK=avgOccPerK_Int+1;
    int optimalNOcc=optimal_avgOccPerK*nkpt;
    double homo=fullEigs[optimalNOcc-1];
    double lumo=fullEigs[optimalNOcc];
    double optimalFermi=(homo+lumo)/2;
    //double res=avgOccPerK%1;
    printf("%d states below Fermi level %.6f, %.6f occupied per k, res: %.6f\n",ilevel,fermi,avgOccPerK,res);
    printf("optimal Fermi=%.6f, with %d states occupied per kpt, between levels %.6f and %.6f\n",optimalFermi,optimal_avgOccPerK,homo,lumo);

    // calc DOS
    if(NEDOS<2){
        printf("Warning: NEDOS<2, setting with nonsense! And I have change NEDOS to 1000\n");
        NEDOS=1000;
    }
    
    double deltaE=(emax-emin)/(NEDOS-1);
    double energy,dos,energyDistance;
    int startIEnergy=0;
    int searchIndex, startIEnergy_updated;
    PyObject * energyAxis=PyTuple_New(NEDOS);
    PyObject * dosAxis=PyTuple_New(NEDOS);
    for(int ienergy=0;ienergy<NEDOS;ienergy++){
        energy=emin+ienergy*deltaE;
        PyTuple_SetItem(energyAxis, ienergy, PyFloat_FromDouble(energy));

        dos=0;
        searchIndex=startIEnergy; // from the startIEnergy
        startIEnergy_updated=0; // reset the tag of startIEnergy_upedated
        while(searchIndex<totalEigs){
            energyDistance=energy - fullEigs[searchIndex];
            if(fabs(energyDistance)<=10*eta){
                if(startIEnergy_updated==0){ // flush start position
                    startIEnergy=searchIndex;
                    startIEnergy_updated=1;
                }
                dos-=cimag(1./(energyDistance+eta*_Complex_I))/PI;
            }else if(energyDistance<0){
                //printf("ienergy %d, energy=%.6f, last involved node is node%d with energy=%.6f\n",
                //        ienergy,energy,currentNode->id,currentNode->eig);
                break;
            }
            searchIndex+=1;
        }
        PyTuple_SetItem(dosAxis, ienergy, PyFloat_FromDouble(dos));
    }
    
    // return data to python
    PyObject * Data=PyTuple_New(3);
    PyTuple_SetItem(Data, 0, energyAxis);
    PyTuple_SetItem(Data, 1, dosAxis);
    PyTuple_SetItem(Data, 2, PyFloat_FromDouble(optimalFermi));

    return Data;
}

PyObject * calcFullBandsAndVecs(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int _nthreads){
    double * Spectra_tmp=(double*)malloc(nkpt*norbs*sizeof(double));
    double * Hk_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk1_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk1_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk2_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk2_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk3_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dk3_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    // calculate matrix elements
    int index;
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //int kBaseIndex=ikpt*norbs*norbs;
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double Dhk[norbs][norbs][3]; // init DHk
            for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++){hk[i][j]=0;for(int k=0;k<3;k++)Dhk[i][j][k]=0;}
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                Dhk[sourceID[ihopping]][targetID[ihopping]][0]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][0]);
                Dhk[sourceID[ihopping]][targetID[ihopping]][1]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][1]);
                Dhk[sourceID[ihopping]][targetID[ihopping]][2]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][2]);
            }
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Hk_r_tmp[index]=creal(hk[iorb][jorb]);
                    Hk_i_tmp[index]=cimag(hk[iorb][jorb]);
                    Hk_dk1_r_tmp[index]=creal(Dhk[iorb][jorb][0]);
                    Hk_dk1_i_tmp[index]=cimag(Dhk[iorb][jorb][0]);
                    Hk_dk2_r_tmp[index]=creal(Dhk[iorb][jorb][1]);
                    Hk_dk2_i_tmp[index]=cimag(Dhk[iorb][jorb][1]);
                    Hk_dk3_r_tmp[index]=creal(Dhk[iorb][jorb][2]);
                    Hk_dk3_i_tmp[index]=cimag(Dhk[iorb][jorb][2]);
                }
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            for(int iorb=0;iorb<norbs;iorb++) Spectra_tmp[ikpt*norbs+iorb]=eigs[iorb];
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Vector_r_tmp[index]=creal(hk[iorb][jorb]);
                    Vector_i_tmp[index]=cimag(hk[iorb][jorb]);
                }
            }
        }
    }

    PyObject * Spectra=PyTuple_New(nkpt*norbs);
    PyObject * Hk_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk1_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk1_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk2_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk2_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk3_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dk3_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_i=PyTuple_New(nkpt*norbs*norbs);

    for(int ikpt=0;ikpt<nkpt;ikpt++){
        for(int iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(Spectra, ikpt*norbs+iorb, PyFloat_FromDouble(Spectra_tmp[ikpt*norbs+iorb]));
        for(int iorb=0;iorb<norbs;iorb++){
            for(int jorb=0;jorb<norbs;jorb++){
                index=ikpt*norbs*norbs+iorb*norbs+jorb;
                PyTuple_SetItem(Hk_r, index, PyFloat_FromDouble(Hk_r_tmp[index]));
                PyTuple_SetItem(Hk_i, index, PyFloat_FromDouble(Hk_i_tmp[index]));
                PyTuple_SetItem(Hk_dk1_r, index, PyFloat_FromDouble(Hk_dk1_r_tmp[index]));
                PyTuple_SetItem(Hk_dk1_i, index, PyFloat_FromDouble(Hk_dk1_i_tmp[index]));
                PyTuple_SetItem(Hk_dk2_r, index, PyFloat_FromDouble(Hk_dk2_r_tmp[index]));
                PyTuple_SetItem(Hk_dk2_i, index, PyFloat_FromDouble(Hk_dk2_i_tmp[index]));
                PyTuple_SetItem(Hk_dk3_r, index, PyFloat_FromDouble(Hk_dk3_r_tmp[index]));
                PyTuple_SetItem(Hk_dk3_i, index, PyFloat_FromDouble(Hk_dk3_i_tmp[index]));
                PyTuple_SetItem(Vector_r, index, PyFloat_FromDouble(Vector_r_tmp[index]));
                PyTuple_SetItem(Vector_i, index, PyFloat_FromDouble(Vector_i_tmp[index]));
                //if(ikpt==0)printf("%.6e %.6e, ",creal(Dhk[iorb][jorb][0]),cimag(Dhk[iorb][jorb][0]));
            }
        }
    }
    return Py_BuildValue("(OOOOOOOOOOO)",Spectra,Vector_r,Vector_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i);
}

PyObject * calcSpinHallCond(int nkpt, double kmesh[nkpt][3], double dk[3], int nMaxRefine,
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nfermi,
                       double fermiList[nfermi],
                       double transform[3][3], // first double rows define kx ky, last row defines spin direction
                       double degenerateThreshold,
                       int _nthreads){
    double * spinBerryCurv_tmp=(double*)malloc(nfermi*nkpt*sizeof(double));
    
    double * kmesh_for_process=(double*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*3*sizeof(double));
    int * kpt_belongings=(int*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(int));
    int * kpt_belongings_nextProcess=(int*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(int));
    double * berryCurvOnKmesh=(double*)malloc(nkpt*sizeof(double));
    double * berryCurvOnKmesh_trial=(double*)malloc(nkpt*sizeof(double));
    double * berryCurvOnRefinedKmesh=(double*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(double));
    //
    //double dk=0;
    //for(int i=0;i<3;i++) if(fabs(kmesh[1][i]-kmesh[0][i])>dk)dk=kmesh[1][i]-kmesh[0][i];
    //int nMaxRefine=4;

    // construct spin matrix
    lapack_complex_double sigma[norbs][norbs];
    for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++)sigma[i][j]=0;
    for(int i=0;i<norbs/2;i++){
        // contribution from sigma^z
        sigma[i][i]=transform[2][2];
        sigma[i+norbs/2][i+norbs/2]=-transform[2][2];
        // contribution from sigma^x
        sigma[i][i+norbs/2]=transform[2][0];
        sigma[i+norbs/2][i]=transform[2][0];
        // contribution from sigma^y
        sigma[i][i+norbs/2]=-transform[2][1]*_Complex_I;
        sigma[i+norbs/2][i]=transform[2][1]*_Complex_I;
    }
    //printf("sigma matrix\n");
    //for(int i=0;i<norbs;i++){
    //    for(int j=0;j<norbs;j++)printf("(%.1f,%.1f) ",creal(sigma[i][j]),cimag(sigma[i][j]));
    //    printf("\n");
    //}
    //printf("a1=(%.3f %.3f %.3f)\n",transform[0][0],transform[0][1],transform[0][2]);
    //printf("a2=(%.3f %.3f %.3f)\n",transform[1][0],transform[1][1],transform[1][2]);
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    // calculate matrix elements
    //#pragma omp parallel num_threads(nthreads)
    {
        for(int ifermi=0;ifermi<nfermi;ifermi++){
            // init processing kmesh
            int nkptForProcess=nkpt;
            for(int ikpt=0;ikpt<nkpt;ikpt++){
                kmesh_for_process[ikpt*3+0]=kmesh[ikpt][0];
                kmesh_for_process[ikpt*3+1]=kmesh[ikpt][1];
                kmesh_for_process[ikpt*3+2]=kmesh[ikpt][2];
                kpt_belongings[ikpt]=ikpt;
            }
            // init oritinal berry curv
            for(int ikpt=0;ikpt<nkpt;ikpt++)berryCurvOnKmesh[ikpt]=1000000.0; //init berryCurv
            int nrefine=1;
            while (1==1){
            //for(int nrefine=1;nrefine<=nMaxRefine;nrefine++){// loop over refinaries
                for(int ikpt=0;ikpt<nkptForProcess;ikpt++)berryCurvOnRefinedKmesh[ikpt]=0.0; // init berryCurv
                printf("calc on fermi: %.4f, with nkpt: %d\n",fermiList[ifermi],nkptForProcess);
                #pragma omp parallel num_threads(nthreads)
                {
                    #pragma omp for
                    for(int ikpt=0;ikpt<nkptForProcess;ikpt++){// loop over kmesh
                        lapack_complex_double hk[norbs][norbs]; // init Hk
                        lapack_complex_double Dhk[norbs][norbs][3]; // init DHk/Dk1,Dk2
                        lapack_complex_double Dhk_xy[norbs][norbs][2]; // init DHk/Dkx,Dky
                        lapack_complex_double SDhk_x[norbs][norbs]; // init S \dot DHk/Dkx
                        for(int i=0;i<norbs;i++){
                            for(int j=0;j<norbs;j++){
                                hk[i][j]=0;
                                Dhk[i][j][0]=0;
                                Dhk[i][j][1]=0;
                                Dhk[i][j][2]=0;
                                Dhk_xy[i][j][0]=0;
                                Dhk_xy[i][j][1]=0;
                                SDhk_x[i][j]=0;
                            }
                        }
                        double eigs[norbs]; // output eigenvalues of Hk
                        for(int ihopping=0;ihopping<nhoppings;ihopping++){
                            double phase=0;
                            //for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*subkmesh[isubkpt][i];
                            for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*kmesh_for_process[ikpt*3+i];
                            lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
                            hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                            Dhk[sourceID[ihopping]][targetID[ihopping]][0]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][0]);
                            Dhk[sourceID[ihopping]][targetID[ihopping]][1]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][1]);
                            Dhk[sourceID[ihopping]][targetID[ihopping]][2]+=tk*(-TwoPI*_Complex_I*overLat[ihopping][2]);
                        }
                        // construct DHk/Dx,Dy
                        for(int i=0;i<norbs;i++){
                            for(int j=0;j<norbs;j++){
                                for(int k=0;k<3;k++){
                                    Dhk_xy[i][j][0]+=transform[0][k]*Dhk[i][j][k];
                                    Dhk_xy[i][j][1]+=transform[1][k]*Dhk[i][j][k];
                                }
                            }
                        }
                        // construct S \dot DHk/Dx
                        for(int i=0;i<norbs;i++){
                            for(int j=0;j<norbs;j++){
                                for(int k=0;k<norbs;k++){
                                    SDhk_x[i][j]+=sigma[i][k]*Dhk_xy[k][j][0];
                                    SDhk_x[i][j]+=Dhk_xy[i][k][0]*sigma[k][j];
                                }
                            }
                        }
                        //printf("kpt%d, [%.1f %.1f %.1f] S Dhk/Dx matrix:\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2]);
                        //for(int i=0;i<norbs;i++){
                        //    for(int j=0;j<norbs;j++)printf("(%.3f,%.3f) ",creal(SDhk_x[i][j]),cimag(SDhk_x[i][j]));
                        //    printf("\n");
                        //}
                        //printf("Dhk/Dy matrix:\n");
                        //for(int i=0;i<norbs;i++){
                        //    for(int j=0;j<norbs;j++)printf("(%.3f,%.3f) ",creal(Dhk_xy[i][j][1]),cimag(Dhk_xy[i][j][1]));
                        //    printf("\n");
                        //}
                        int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
                        // vec[i]=hk[:][i]
                        if(info!=0)printf("LAPACKE failed, info: %d\n",info);

                        for(int nband=0;nband<norbs;nband++){
                            double En=eigs[nband];
                            if(En<=fermiList[ifermi]){
                                //printf("valence bandn: %d energy: %.6f vec: ",nband,En);
                                //for(int i=0;i<norbs;i++)printf("(%.3f,%.3f) ",creal(hk[i][nband]),cimag(hk[i][nband]));
                                //printf("\n");
                                lapack_complex_double vecnDotSDHkDx[norbs];
                                for(int i=0;i<norbs;i++)vecnDotSDHkDx[i]=0;
                                for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++)vecnDotSDHkDx[i]+=conj(hk[j][nband])*SDhk_x[j][i];
                                //printf("n dot SDHk/Dx: ");
                                //for(int i=0;i<norbs;i++)printf("(%.3f,%.3f) ",creal(vecnDotSDHkDx[i]),cimag(vecnDotSDHkDx[i]));
                                //printf("\n");
                                lapack_complex_double DHkDyDotVecn[norbs];
                                for(int i=0;i<norbs;i++)DHkDyDotVecn[i]=0;
                                for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++)DHkDyDotVecn[i]+=Dhk_xy[i][j][1]*hk[j][nband];
                                //printf("DHk/Dy dot n: ");
                                //for(int i=0;i<norbs;i++)printf("(%.3f,%.3f) ",creal(DHkDyDotVecn[i]),cimag(DHkDyDotVecn[i]));
                                //printf("\n");

                                for(int mband=0;mband<norbs;mband++){
                                    double Em=eigs[mband];
                                    double deltaE=Em-En;
                                    double deltaE_square_double=deltaE*deltaE*2;
                                    if(Em>fermiList[ifermi] & deltaE>degenerateThreshold){
                                        //printf("conduct bandm: %d energy: %.6f vec: ",mband,Em);
                                        //for(int i=0;i<norbs;i++)printf("(%.3f,%.3f) ",creal(hk[i][mband]),cimag(hk[i][mband]));
                                        //printf("\n");
                                        lapack_complex_double Amp1=0;
                                        lapack_complex_double Amp2=0;
                                        for(int i=0;i<norbs;i++){
                                            Amp1+=vecnDotSDHkDx[i]*hk[i][mband];
                                            Amp2+=conj(hk[i][mband])*DHkDyDotVecn[i];
                                        }
                                        //printf("amp1: %.3f %.3f; amp2: %.3f %.3f\n",creal(Amp1),cimag(Amp1),creal(Amp2),cimag(Amp2));
                                        //spinBerryCurv_tmp[ifermi*nkpt+ikpt]+=cimag(Amp1*Amp2)/deltaE_square_double;
                                        berryCurvOnRefinedKmesh[ikpt]+=cimag(Amp1*Amp2)/deltaE_square_double;
                                    }
                                }
                            }
                        }
                    }
                }
                printf("calc on fermi: %.4f, with nkpt: %d, done!\n",fermiList[ifermi],nkptForProcess);
                
                nrefine+=1;
                //accquire berry on original kmesh
                // first confirm the individual kpt groups
                int nkptGroups=0;
                int sizeOfGroups[nkpt];
                //for(int ikpt=0;ikpt<nkpt;ikpt++)sizeOfGroups[ikpt]=0;
                int nContribute=0;
                int ikpt_contribute_old=kpt_belongings[0];
                int ikpt_contribute_now;
                for(int ikpt=0;ikpt<nkptForProcess-1;ikpt++){
                    ikpt_contribute_now=kpt_belongings[ikpt];
                    if(ikpt_contribute_now==ikpt_contribute_old){
                        nContribute+=1;
                    }else{
                        ikpt_contribute_old=ikpt_contribute_now;
                        sizeOfGroups[nkptGroups]=nContribute;
                        nkptGroups+=1;
                        nContribute=1;
                    }
                }
                ikpt_contribute_now=kpt_belongings[nkptForProcess-1];
                //printf("last ikpt group index: %d, -- former index: %d\n",ikpt_contribute_now,ikpt_contribute_old);
                if(ikpt_contribute_now==ikpt_contribute_old){
                    sizeOfGroups[nkptGroups]=nContribute+1;
                }else{
                    sizeOfGroups[nkptGroups]=nContribute;
                    nkptGroups+=1;
                    sizeOfGroups[nkptGroups]=1;
                    nkptGroups+=1;
                }
                //printf("found %d kpt groups\n",nkptGroups);

                for(int ikpt=0;ikpt<nkpt;ikpt++)berryCurvOnKmesh_trial[ikpt]=0.0; //init berryCurv
                int nkptForNextProcess=0;
                int ikpt=0;
                for(int igroup=0;igroup<nkptGroups;igroup++){
                    ikpt_contribute_now=kpt_belongings[ikpt];
                    for(int ikpt_ingroup=0;ikpt_ingroup<sizeOfGroups[igroup];ikpt_ingroup++){
                        berryCurvOnKmesh_trial[ikpt_contribute_now]+=berryCurvOnRefinedKmesh[ikpt];
                        //printf("kpt%d group, kpt%d in refined mesh\n",ikpt_contribute_now,ikpt);
                        //printf("      contributing berry curv of %.6f\n",berryCurvOnRefinedKmesh[ikpt]);
                        ikpt+=1;
                    }
                    berryCurvOnKmesh_trial[ikpt_contribute_now]/=sizeOfGroups[igroup];
                    //printf("kpt%d group finished, with %d grids, berry curv: %.3f\n",ikpt_contribute_now,sizeOfGroups[igroup],berryCurvOnKmesh_trial[ikpt_contribute_now]);
                    // compare the result and check if we need to refine grid more
                    if(fabs(berryCurvOnKmesh_trial[ikpt_contribute_now])<0.1 | fabs((berryCurvOnKmesh_trial[ikpt_contribute_now]-berryCurvOnKmesh[ikpt_contribute_now])/berryCurvOnKmesh[ikpt_contribute_now])<0.1){
                        // update
                        berryCurvOnKmesh[ikpt_contribute_now]=berryCurvOnKmesh_trial[ikpt_contribute_now];
                        //printf("kpt%d group converged\n",ikpt_contribute_now);
                    }else{
                        // update
                        //printf("kpt%d group did not converge, ratio: %.3f/%.3f\n",ikpt_contribute_now,berryCurvOnKmesh_trial[ikpt_contribute_now],berryCurvOnKmesh[ikpt_contribute_now]);
                        if(nrefine<=nMaxRefine){
                           // create/add refined kmesh
                            double ddk[3];// interval of refined grid
                            for(int i=0;i<3;i++)ddk[i]=dk[i]/nrefine;
                            double odk[3];// extra shift to locate refined grid around center
                            for(int i=0;i<3;i++)odk[i]=ddk[i]*(nrefine-1)/2;
                            //printf("create refined grids for kpt%d group centered in %.3f %.3f %.3f\n",ikpt_contribute_now,kmesh[ikpt_contribute_now][0],kmesh[ikpt_contribute_now][1],kmesh[ikpt_contribute_now][2]);
                            if(dk[2]<0.6){ // dim==3
                                for(int i=0;i<nrefine;i++){
                                    for(int j=0;j<nrefine;j++){
                                        for(int k=0;k<nrefine;k++){
                                            kmesh_for_process[nkptForNextProcess*3+0]=kmesh[ikpt_contribute_now][0]+i*ddk[0]-odk[0];
                                            kmesh_for_process[nkptForNextProcess*3+1]=kmesh[ikpt_contribute_now][1]+j*ddk[1]-odk[1];
                                            kmesh_for_process[nkptForNextProcess*3+2]=kmesh[ikpt_contribute_now][2]+k*ddk[2]-odk[2];
                                            //printf("    include kpt%d, %.3f %.3f %.3f\n",nkptForNextProcess,kmesh_for_process[nkptForNextProcess*3+0],kmesh_for_process[nkptForNextProcess*3+1],kmesh_for_process[nkptForNextProcess*3+2]);
                                            kpt_belongings_nextProcess[nkptForNextProcess]=ikpt_contribute_now;
                                            nkptForNextProcess+=1;
                                        }
                                    }
                                }
                            }else{ //dim==2
                                for(int i=0;i<nrefine;i++){
                                    for(int j=0;j<nrefine;j++){
                                        kmesh_for_process[nkptForNextProcess*3+0]=kmesh[ikpt_contribute_now][0]+i*ddk[0]-odk[0];
                                        kmesh_for_process[nkptForNextProcess*3+1]=kmesh[ikpt_contribute_now][1]+j*ddk[1]-odk[1];
                                        kmesh_for_process[nkptForNextProcess*3+2]=kmesh[ikpt_contribute_now][2];//+k*ddk[2]-odk[2];
                                        //printf("    include kpt%d, %.3f %.3f %.3f\n",nkptForNextProcess,kmesh_for_process[nkptForNextProcess*3+0],kmesh_for_process[nkptForNextProcess*3+1],kmesh_for_process[nkptForNextProcess*3+2]);
                                        kpt_belongings_nextProcess[nkptForNextProcess]=ikpt_contribute_now;
                                        nkptForNextProcess+=1;
                                    }
                                }
                            }
                        }else{
                            printf("cvg. failed at fermi %.4f, ikpt %d, [%.4f %.4f %.4f], final ratio: %.6f/%.6f\n",fermiList[ifermi],ikpt_contribute_now,kmesh[ikpt_contribute_now][0],kmesh[ikpt_contribute_now][1],kmesh[ikpt_contribute_now][2],berryCurvOnKmesh_trial[ikpt_contribute_now],berryCurvOnKmesh[ikpt_contribute_now]);
                        }
                        berryCurvOnKmesh[ikpt_contribute_now]=berryCurvOnKmesh_trial[ikpt_contribute_now];
                    }
                }// analysis on data on refined kmesh fininished
                
                if(nkptForNextProcess==0 | nrefine>nMaxRefine){ // stop the loop due to either convergence or max refinement reached
                    for(int ikpt=0;ikpt<nkpt;ikpt++){
                        spinBerryCurv_tmp[ifermi*nkpt+ikpt]=berryCurvOnKmesh[ikpt];
                        //printf("finnal result in kpt%d: %.3f\n",ikpt,spinBerryCurv_tmp[ifermi*nkpt+ikpt]);
                    }
                    break;
                }else{ // do another try
                    nkptForProcess=nkptForNextProcess;
                    for(int ikpt=0;ikpt<nkptForProcess;ikpt++)kpt_belongings[ikpt]=kpt_belongings_nextProcess[ikpt];
                }
            }
        }
    }

    PyObject * spinBerryCurv=PyTuple_New(nfermi*nkpt);

    for(int ifermi=0;ifermi<nfermi;ifermi++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            PyTuple_SetItem(spinBerryCurv, ifermi*nkpt+ikpt, PyFloat_FromDouble(spinBerryCurv_tmp[ifermi*nkpt+ikpt]));
        }
    }
    return Py_BuildValue("(O)",spinBerryCurv);
}