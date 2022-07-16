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
            //printf("kpt%d %.3f %.3f %.3f\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2]);
            //int kBaseIndex=ikpt*norbs*norbs;
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double Dhk[norbs][norbs][3]; // init DHk
            for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++){hk[i][j]=0;for(int k=0;k<3;k++)Dhk[i][j][k]=0;}
            double eigs[norbs]; // output eigenvalues of Hk
            //printf("hoppings\n");
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*kmesh[ikpt][i];
                //printf("bond%d [",ihopping);
                //for (int i=0;i<3;i++) printf("%d ",overLat[ihopping][i]);
                //printf("] %d ~ %d",sourceID[ihopping],targetID[ihopping]);
                //double phase_tmp=0;
                //for (int i=0;i<3;i++) phase_tmp+=overLat[ihopping][i]*kmesh[ikpt][i];
                //printf(" phase: %.4f 2pi t: %.4f %.4f\n",phase_tmp,hopping_real[ihopping],hopping_imag[ihopping]);
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
            /*
            //printf("%.4f %.4f\n",creal(hk[0][1]*hk[1][0]),3+2*cos(TwoPI*kmesh[ikpt][0])+2*cos(TwoPI*kmesh[ikpt][1])+2*cos(TwoPI*(kmesh[ikpt][0]+kmesh[ikpt][1])));
            // t*(1 + exp(-i 2pi k1) + exp(-i 2pi k2) )
            lapack_complex_double hk_ele_trial=-1*(1+cexp(-TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(-TwoPI*_Complex_I*kmesh[ikpt][1]));
            printf("trial elements: %.4f %.4f\n",creal(hk_ele_trial),cimag(hk_ele_trial));
            // t*(1 + exp( i 2pi k1) + exp( i 2pi k2) )
            lapack_complex_double hk_ele_trial_conj=-1*(1+cexp(TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(TwoPI*_Complex_I*kmesh[ikpt][1]));
            lapack_complex_double hk_ele_trial_square=hk_ele_trial*conj(hk_ele_trial);
            lapack_complex_double hk_ele_trial_square2=hk_ele_trial*hk_ele_trial_conj;
            lapack_complex_double hk_ele_trial_square3=(1+cexp(-TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(-TwoPI*_Complex_I*kmesh[ikpt][1]))*(1+cexp(TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(TwoPI*_Complex_I*kmesh[ikpt][1]));
            hk_ele_trial_square3=3+cexp(TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(-TwoPI*_Complex_I*kmesh[ikpt][0])+cexp(TwoPI*_Complex_I*kmesh[ikpt][1])+cexp(-TwoPI*_Complex_I*kmesh[ikpt][1])+cexp(TwoPI*_Complex_I*(kmesh[ikpt][0]-kmesh[ikpt][1]))+cexp(-TwoPI*_Complex_I*(kmesh[ikpt][0]-kmesh[ikpt][1]));
            //t^2*(3 + 2cos k1 + 2cos k2 + 2cos(k1-k2))
            hk_ele_trial_square3=3+2*cos(TwoPI*kmesh[ikpt][0])+2*cos(TwoPI*kmesh[ikpt][1])+2*cos(TwoPI*(kmesh[ikpt][0]-kmesh[ikpt][1]));
            
            printf("|f(k)|^2: %.4f %.4f, %.4f %.4f, %.4f %.4f\n",creal(hk_ele_trial_square),cimag(hk_ele_trial_square),creal(hk_ele_trial_square2),cimag(hk_ele_trial_square2),creal(hk_ele_trial_square3),cimag(hk_ele_trial_square3));
            */
            //printf("Hk:\n");
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    //printf("(%.4f %.4f) ",creal(hk[iorb][jorb]),cimag(hk[iorb][jorb]));
                    /*
                    Hk_dk1_r_tmp[index]=creal(Dhk[iorb][jorb][0]);
                    Hk_dk1_i_tmp[index]=cimag(Dhk[iorb][jorb][0]);
                    Hk_dk2_r_tmp[index]=creal(Dhk[iorb][jorb][1]);
                    Hk_dk2_i_tmp[index]=cimag(Dhk[iorb][jorb][1]);
                    Hk_dk3_r_tmp[index]=creal(Dhk[iorb][jorb][2]);
                    Hk_dk3_i_tmp[index]=cimag(Dhk[iorb][jorb][2]);
                    */
                }
                //printf("\n");
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigs: %.4f %.4f %.4f\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2],eigs[0],eigs[1],eigs[2]);
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

PyObject * calcBandsVecsVelocityOps(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       int _nthreads){
    double * Spectra_tmp=(double*)malloc(nkpt*norbs*sizeof(double));
    double * Hk_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAx_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAx_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAy_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAy_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAz_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Hk_dAz_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_r_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    double * Vector_i_tmp=(double*)malloc(nkpt*norbs*norbs*sizeof(double));
    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    // calculate matrix elements
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double Vk[norbs][norbs][3]; // init DHkDk_1,2,3
            lapack_complex_double Vxyz[norbs][norbs][3]; // init DHkDk_1,2,3
            for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++){
                hk[i][j]=0;
                for(int k=0;k<3;k++){
                    Vk[i][j][k]=0;
                    Vxyz[i][j][k]=0;
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (int i=0;i<3;i++) phase+=(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i])*kmesh[ikpt][i];
                //for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                Vk[sourceID[ihopping]][targetID[ihopping]][0]+=tk*(-TwoPI*_Complex_I)*(overLat[ihopping][0]+orbCorr[targetID[ihopping]][0]-orbCorr[sourceID[ihopping]][0]);
                Vk[sourceID[ihopping]][targetID[ihopping]][1]+=tk*(-TwoPI*_Complex_I)*(overLat[ihopping][1]+orbCorr[targetID[ihopping]][1]-orbCorr[sourceID[ihopping]][1]);
                Vk[sourceID[ihopping]][targetID[ihopping]][2]+=tk*(-TwoPI*_Complex_I)*(overLat[ihopping][2]+orbCorr[targetID[ihopping]][2]-orbCorr[sourceID[ihopping]][2]);
            }
            // unitary transform into real space partial
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    for(int k=0;k<3;k++){
                        Vxyz[i][j][0]+=Vk[i][j][k]*lattice[k][0];
                        Vxyz[i][j][1]+=Vk[i][j][k]*lattice[k][1];
                        Vxyz[i][j][2]+=Vk[i][j][k]*lattice[k][2];
                    }
                }
            }
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    int index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Hk_r_tmp[index]=creal(hk[iorb][jorb]);
                    Hk_i_tmp[index]=cimag(hk[iorb][jorb]);
                    Hk_dAx_r_tmp[index]=creal(Vxyz[iorb][jorb][0]);
                    Hk_dAx_i_tmp[index]=cimag(Vxyz[iorb][jorb][0]);
                    Hk_dAy_r_tmp[index]=creal(Vxyz[iorb][jorb][1]);
                    Hk_dAy_i_tmp[index]=cimag(Vxyz[iorb][jorb][1]);
                    Hk_dAz_r_tmp[index]=creal(Vxyz[iorb][jorb][2]);
                    Hk_dAz_i_tmp[index]=cimag(Vxyz[iorb][jorb][2]);
                }
            }
            
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            //printf("kpt%d %.3f %.3f %.3f eigs: %.4f %.4f %.4f\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2],eigs[0],eigs[1],eigs[2]);
            for(int iorb=0;iorb<norbs;iorb++) Spectra_tmp[ikpt*norbs+iorb]=eigs[iorb];
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    int index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    Vector_r_tmp[index]=creal(hk[iorb][jorb]);
                    Vector_i_tmp[index]=cimag(hk[iorb][jorb]);
                }
            }
        }
    }

    PyObject * Spectra=PyTuple_New(nkpt*norbs);
    PyObject * Hk_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAx_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAx_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAy_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAy_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAz_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Hk_dAz_i=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_r=PyTuple_New(nkpt*norbs*norbs);
    PyObject * Vector_i=PyTuple_New(nkpt*norbs*norbs);

    for(int ikpt=0;ikpt<nkpt;ikpt++){
        for(int iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(Spectra, ikpt*norbs+iorb, PyFloat_FromDouble(Spectra_tmp[ikpt*norbs+iorb]));
        for(int iorb=0;iorb<norbs;iorb++){
            for(int jorb=0;jorb<norbs;jorb++){
                int index=ikpt*norbs*norbs+iorb*norbs+jorb;
                PyTuple_SetItem(Hk_r, index, PyFloat_FromDouble(Hk_r_tmp[index]));
                PyTuple_SetItem(Hk_i, index, PyFloat_FromDouble(Hk_i_tmp[index]));
                PyTuple_SetItem(Hk_dAx_r, index, PyFloat_FromDouble(Hk_dAx_r_tmp[index]));
                PyTuple_SetItem(Hk_dAx_i, index, PyFloat_FromDouble(Hk_dAx_i_tmp[index]));
                PyTuple_SetItem(Hk_dAy_r, index, PyFloat_FromDouble(Hk_dAy_r_tmp[index]));
                PyTuple_SetItem(Hk_dAy_i, index, PyFloat_FromDouble(Hk_dAy_i_tmp[index]));
                PyTuple_SetItem(Hk_dAz_r, index, PyFloat_FromDouble(Hk_dAz_r_tmp[index]));
                PyTuple_SetItem(Hk_dAz_i, index, PyFloat_FromDouble(Hk_dAz_i_tmp[index]));
                PyTuple_SetItem(Vector_r, index, PyFloat_FromDouble(Vector_r_tmp[index]));
                PyTuple_SetItem(Vector_i, index, PyFloat_FromDouble(Vector_i_tmp[index]));
                //if(ikpt==0)printf("%.6e %.6e, ",creal(Dhk[iorb][jorb][0]),cimag(Dhk[iorb][jorb][0]));
            }
        }
    }
    return Py_BuildValue("(OOOOOOOOOOO)",Spectra,Vector_r,Vector_i,Hk_r,Hk_i,Hk_dAx_r,Hk_dAx_i,Hk_dAy_r,Hk_dAy_i,Hk_dAz_r,Hk_dAz_i);
}

PyObject * calcFullBandsAndVecsInMemory(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3],
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       double orbCorr[norbs][3],
                       double lattice[3][3],
                       int _nthreads){
    double * eigs=(double*)malloc(nkpt*norbs*sizeof(double));
    lapack_complex_double * vecs=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDkx=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDky=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDkz=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDkxDkx=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDkyDky=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double * DHkDkxDky=(lapack_complex_double*)malloc(nkpt*norbs*norbs*sizeof(lapack_complex_double));
    printf("estimated memory: %d MB\n",(nkpt*norbs*(sizeof(double)+7*norbs*sizeof(lapack_complex_double)))/1024/1024);
    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * _hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double * _dhkdkxyz=(lapack_complex_double*)malloc(norbs*norbs*3*sizeof(lapack_complex_double));
        lapack_complex_double * _dhkdkxdky=(lapack_complex_double*)malloc(norbs*norbs*9*sizeof(lapack_complex_double));
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //printf("kpt%d %.3f %.3f %.3f\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2]);
            //int kBaseIndex=ikpt*norbs*norbs;
            //lapack_complex_double _hk[norbs][norbs]; // init Hk
            //lapack_complex_double _dhkdkxyz[norbs][norbs][3];
            //lapack_complex_double _dhkdkxdky[norbs][norbs][3][3];
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    _hk[i*norbs+j]=0;
                    
                    for(int k=0;k<3;k++)_dhkdkxyz[i*norbs*3+j*3+k]=0;
                    for(int x=0;x<3;x++)for(int y=0;y<3;y++)_dhkdkxdky[i*norbs*9+j*9+x*3+y]=0;
                }
            }
            double _eigs[norbs]; // output eigenvalues of Hk
            //printf("hoppings\n");
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                lapack_complex_double hopping_R[3];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);
                
                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt][i];
                
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                _hk[sourceID[ihopping]*norbs+targetID[ihopping]]+=tk;

                lapack_complex_double hopping_R_c[3];
                for(int i=0;i<3;i++)hopping_R_c[i]=0;
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)hopping_R_c[i]+=hopping_R[j]*lattice[j][i];

                for(int i=0;i<2;i++) _dhkdkxyz[sourceID[ihopping]*3*norbs+targetID[ihopping]*3+i]+=tk*hopping_R_c[i];
                for(int i=0;i<2;i++)for(int j=0;j<2;j++)_dhkdkxdky[sourceID[ihopping]*9*norbs+targetID[ihopping]*9+i*3+j]+=tk*hopping_R_c[i]*hopping_R_c[j];
            }
            //printf("kpt%d %.3f %.3f %.3f hamk:\n",ikpt,kmesh[ikpt][0],kmesh[ikpt][1],kmesh[ikpt][2]);
            //for(int i=0;i<norbs;i++){
            //    for(int j=0;j<norbs;j++)printf("%f %f,",creal(_hk[i][j]),cimag(_hk[i][j]));
            //    printf("\n");
            //}
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','U',norbs,_hk,norbs,_eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            //printf("eigs:");
            //for(int i=0;i<norbs;i++)printf(" %.4f",_eigs[i]);
            //printf(" vec:\n");
            //for(int i=0;i<norbs;i++){
            //    for(int j=0;j<norbs;j++)printf("%f %f,",creal(_hk[i][j]),cimag(_hk[i][j]));
            //    printf("\n");
            //}
            for(int iorb=0;iorb<norbs;iorb++) eigs[ikpt*norbs+iorb]=_eigs[iorb];
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    int index=ikpt*norbs*norbs+iorb*norbs+jorb;
                    vecs[index]=_hk[iorb*norbs+jorb];
                    DHkDkx[index]=_dhkdkxyz[iorb*3*norbs+jorb*3+0];
                    DHkDky[index]=_dhkdkxyz[iorb*3*norbs+jorb*3+1];
                    DHkDkz[index]=_dhkdkxyz[iorb*3*norbs+jorb*3+2];
                    DHkDkxDkx[index]=_dhkdkxdky[iorb*9*norbs+jorb*9+0*3+0];
                    DHkDkyDky[index]=_dhkdkxdky[iorb*9*norbs+jorb*9+1*3+1];
                    DHkDkxDky[index]=_dhkdkxdky[iorb*9*norbs+jorb*9+0*3+1];
                }
            }
        }
    }
    
    long long eigs_addr=(long long)eigs;
    long long vecs_addr=(long long)vecs;
    long long DHkDkx_addr=(long long)DHkDkx;
    long long DHkDky_addr=(long long)DHkDky;
    long long DHkDkz_addr=(long long)DHkDkz;
    long long DHkDkxDkx_addr=(long long)DHkDkxDkx;
    long long DHkDkyDky_addr=(long long)DHkDkyDky;
    long long DHkDkxDky_addr=(long long)DHkDkxDky;
    printf("in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(lapack_complex_double *));
    printf("pls. see if they are identical. Otherwise the calc. results with cdll are seriously wrong!\n");
    printf("in ham.dll the addr. of eigs and vecs are %p, %p\n",eigs_addr,vecs_addr);
    printf("DHkDkx,ky,kz: %p, %p, %p\n",DHkDkx_addr,DHkDky_addr,DHkDkz_addr);
    return Py_BuildValue("(LLLLLLLL)",eigs_addr,vecs_addr,DHkDkx_addr,DHkDky_addr,DHkDkz_addr,DHkDkxDkx_addr,DHkDkyDky_addr,DHkDkxDky_addr);
}

PyObject* derivateOnKMesh_double(long long data_addr, int dataDim, int kmesh[3], double lat[3][3], int nthreads){
    double*gridData=(double*)data_addr;
    printf("ham.dll is to derivate the double type data @ %p\n",gridData);

    int nkpt=kmesh[0]*kmesh[1]*kmesh[2];
    printf("on kmesh %dx%dx%d, assuming the k3 runs fastest, then k2, then k1\n",kmesh[0],kmesh[1],kmesh[2]);

    double dk[3];for(int i=0;i<3;i++)dk[i]=1/(double)kmesh[i];
    //printf("dk %f %f %f\n",dk[0],dk[1],dk[2]);

    double*dData_dkx=(double*)malloc((nkpt*dataDim)*sizeof(double));
    double*dData_dky=(double*)malloc((nkpt*dataDim)*sizeof(double));
    double*dData_dkz=(double*)malloc((nkpt*dataDim)*sizeof(double));

    int sizeK2=kmesh[2];
    int sizeK1=kmesh[2]*kmesh[1];
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int ik1=ikpt/sizeK1;
            int ik2=(ikpt-ik1*sizeK1)/sizeK2;
            int ik3=ikpt-ik1*sizeK1-ik2*sizeK2;

            int index_base=ikpt*dataDim;
            int index_dq1=((ik1+1)%kmesh[0]*sizeK1+ik2*sizeK2+ik3)%nkpt*dataDim;
            int index_dq2=(ik1*sizeK1+(ik2+1)%kmesh[1]*sizeK2+ik3)%nkpt*dataDim;
            int index_dq3=(ik1*sizeK1+ik2*sizeK2+(ik3+1)%kmesh[2])%nkpt*dataDim;
            //printf("k0: %d, k+dk1: %d, k+dk2: %d, k+dk3: %d\n",ikpt,index_dq1/dataDim,index_dq2/dataDim,index_dq3/dataDim);
            double data_dk123[3][dataDim];
            //printf("%f\n",gridData[index_base]);
            for(int idata=0;idata<dataDim;idata++){
                data_dk123[0][idata]=(gridData[index_dq1+idata]-gridData[index_base+idata])/dk[0];
                data_dk123[1][idata]=(gridData[index_dq2+idata]-gridData[index_base+idata])/dk[1];
                data_dk123[2][idata]=(gridData[index_dq3+idata]-gridData[index_base+idata])/dk[2];
                //printf("%f\n",data_dk123[0][0]);
            }
            // change into derivates to kx,y,z
            double data_dkxyz[3][dataDim];
            for(int i=0;i<3;i++)for(int idata=0;idata<dataDim;idata++)data_dkxyz[i][idata]=0;
            for(int idata=0;idata<dataDim;idata++){
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)data_dkxyz[i][idata]+=data_dk123[j][idata]*lat[j][i];
            }
            //printf("%f\n",data_dkxyz[0][0]);
            for(int idata=0;idata<dataDim;idata++){
                dData_dkx[index_base+idata]=data_dkxyz[0][idata];
                dData_dky[index_base+idata]=data_dkxyz[1][idata];
                dData_dkz[index_base+idata]=data_dkxyz[2][idata];
            }
        }
    }
    long long dData_dkx_addr=(long long)dData_dkx;
    long long dData_dky_addr=(long long)dData_dky;
    long long dData_dkz_addr=(long long)dData_dkz;
    if(sizeof(long long)!=sizeof(double*)){
        printf("Error: in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(double *));
        printf("they are different!\n");
        exit(1);
    }
    printf("in ham.dll the addr. of derivates are %p, %p, %p\n",dData_dkx_addr,dData_dky_addr,dData_dkz_addr);
    return Py_BuildValue("(LLL)",dData_dkx_addr,dData_dky_addr,dData_dkz_addr);
}

PyObject* derivateOnKMesh_complex(long long data_addr, int dataDim, int kmesh[3], double lat[3][3], int nthreads){
    lapack_complex_double*gridData=(lapack_complex_double*)data_addr;
    printf("ham.dll is to derivate the complex type data @ %p\n",gridData);

    int nkpt=kmesh[0]*kmesh[1]*kmesh[2];
    printf("on kmesh %dx%dx%d, assuming the k3 runs fastest, then k2, then k1\n",kmesh[0],kmesh[1],kmesh[2]);

    double dk[3];for(int i=0;i<3;i++)dk[i]=1/(double)kmesh[i];
    //printf("dk %f %f %f\n",dk[0],dk[1],dk[2]);

    lapack_complex_double*dData_dkx=(lapack_complex_double*)malloc((nkpt*dataDim)*sizeof(lapack_complex_double));
    lapack_complex_double*dData_dky=(lapack_complex_double*)malloc((nkpt*dataDim)*sizeof(lapack_complex_double));
    lapack_complex_double*dData_dkz=(lapack_complex_double*)malloc((nkpt*dataDim)*sizeof(lapack_complex_double));

    int sizeK2=kmesh[2];
    int sizeK1=kmesh[2]*kmesh[1];
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int ik1=ikpt/sizeK1;
            int ik2=(ikpt-ik1*sizeK1)/sizeK2;
            int ik3=ikpt-ik1*sizeK1-ik2*sizeK2;

            int index_base=ikpt*dataDim;
            int index_dq1=((ik1+1)%kmesh[0]*sizeK1+ik2*sizeK2+ik3)%nkpt*dataDim;
            int index_dq2=(ik1*sizeK1+(ik2+1)%kmesh[1]*sizeK2+ik3)%nkpt*dataDim;
            int index_dq3=(ik1*sizeK1+ik2*sizeK2+(ik3+1)%kmesh[2])%nkpt*dataDim;
            //printf("k0: %d, k+dk1: %d, k+dk2: %d, k+dk3: %d\n",ikpt,index_dq1/dataDim,index_dq2/dataDim,index_dq3/dataDim);
            lapack_complex_double data_dk123[3][dataDim];
            //printf("%f\n",gridData[index_base]);
            //printf("ikpt %d ik1 %d ik2 %d\n",ikpt,ik1,ik2);
            for(int idata=0;idata<dataDim;idata++){
                data_dk123[0][idata]=(gridData[index_dq1+idata]-gridData[index_base+idata])/dk[0];
                data_dk123[1][idata]=(gridData[index_dq2+idata]-gridData[index_base+idata])/dk[1];
                data_dk123[2][idata]=(gridData[index_dq3+idata]-gridData[index_base+idata])/dk[2];
            }
            //for(int i=0;i<2;i++){
            //    for(int j=0;j<2;j++)printf("%f %f,",creal(data_dk123[0][i*2+j]),cimag(data_dk123[0][i*2+j]));
            //    printf("\n");
            //}
            
            // change into derivates to kx,y,z
            lapack_complex_double data_dkxyz[3][dataDim];
            for(int i=0;i<3;i++)for(int idata=0;idata<dataDim;idata++)data_dkxyz[i][idata]=0;
            for(int idata=0;idata<dataDim;idata++){
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)data_dkxyz[i][idata]+=data_dk123[j][idata]*lat[j][i];
            }
            //printf("%f\n",data_dkxyz[0][0]);
            for(int idata=0;idata<dataDim;idata++){
                dData_dkx[index_base+idata]=data_dkxyz[0][idata];
                dData_dky[index_base+idata]=data_dkxyz[1][idata];
                dData_dkz[index_base+idata]=data_dkxyz[2][idata];
            }
        }
    }
    long long dData_dkx_addr=(long long)dData_dkx;
    long long dData_dky_addr=(long long)dData_dky;
    long long dData_dkz_addr=(long long)dData_dkz;
    if(sizeof(long long)!=sizeof(lapack_complex_double*)){
        printf("Error: in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(lapack_complex_double *));
        printf("they are different!\n");
        exit(1);
    }
    printf("in ham.dll the addr. of derivates are %p, %p, %p\n",dData_dkx_addr,dData_dky_addr,dData_dkz_addr);
    return Py_BuildValue("(LLL)",dData_dkx_addr,dData_dky_addr,dData_dkz_addr);
}

PyObject* derivateOnKMesh_eigenvector(long long vec_addr, long long DHkDk1_addr, long long DHkDk2_addr, long long DHkDk3_addr, long long eigs_addr, int norb, int nocc, int kmesh[3], double lat[3][3], int nthreads){
    lapack_complex_double*vecs=(lapack_complex_double*)vec_addr;
    double*eigs=(double*)eigs_addr;
    lapack_complex_double*DHkDk1=(lapack_complex_double*)DHkDk1_addr;
    lapack_complex_double*DHkDk2=(lapack_complex_double*)DHkDk2_addr;
    lapack_complex_double*DHkDk3=(lapack_complex_double*)DHkDk3_addr;

    printf("ham.dll is to derivate the eigenvectors @ %p, phases are corrected automatically\n",vecs);
    printf("Note that BZ boundary wrapping should be neglected due to the disturbulences of orb phases\n");

    int nkpt=kmesh[0]*kmesh[1]*kmesh[2];
    printf("on kmesh %dx%dx%d, assuming the k3 runs fastest, then k2, then k1\n",kmesh[0],kmesh[1],kmesh[2]);

    double dk[3];for(int i=0;i<3;i++)dk[i]=1/(double)kmesh[i];
    //printf("dk %f %f %f\n",dk[0],dk[1],dk[2]);
    int norb2=norb*norb;
    lapack_complex_double*dvec_dkx=(lapack_complex_double*)malloc((nkpt*norb2)*sizeof(lapack_complex_double));
    lapack_complex_double*dvec_dky=(lapack_complex_double*)malloc((nkpt*norb2)*sizeof(lapack_complex_double));
    lapack_complex_double*dvec_dkz=(lapack_complex_double*)malloc((nkpt*norb2)*sizeof(lapack_complex_double));

    int sizeK2=kmesh[2];
    int sizeK1=kmesh[1]*kmesh[2];
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int ik1=ikpt/sizeK1;
            int ik2=(ikpt-ik1*sizeK1)/sizeK2;
            int ik3=ikpt-ik1*sizeK1-ik2*sizeK2;

            int index_base=ikpt*norb2;
            int index_dq1=((ik1+1)%kmesh[0]*sizeK1+ik2*sizeK2+ik3)*norb2;
            int index_dq2=(ik1*sizeK1+(ik2+1)%kmesh[1]*sizeK2+ik3)*norb2;
            int index_dq3=(ik1*sizeK1+ik2*sizeK2+(ik3+1)%kmesh[2])*norb2;
            //if(ikpt<100)printf("k0: %d, k+dk1: %d, k+dk2: %d, k+dk3: %d\n",ikpt,index_dq1/norb2,index_dq2/norb2,index_dq3/norb2);
            lapack_complex_double vec_0[norb][norb];
            lapack_complex_double vec_1[norb][norb];
            lapack_complex_double vec_2[norb][norb];
            lapack_complex_double vec_3[norb][norb];
            //printf("vec:\n");
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    vec_0[i][j]=vecs[index_base+i*norb+j];
                    vec_1[i][j]=vecs[index_dq1+i*norb+j];
                    vec_2[i][j]=vecs[index_dq2+i*norb+j];
                    vec_3[i][j]=vecs[index_dq3+i*norb+j];
                    //printf("%f %f,",creal(vec_0[i][j]),cimag(vec_0[i][j]));
                }
                //printf("\n");
            }
            // calculate numerical overlaps
            lapack_complex_double overlap_numeric[3][norb][norb];
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    overlap_numeric[0][i][j]=0;
                    overlap_numeric[1][i][j]=0;
                    overlap_numeric[2][i][j]=0;
                }
            }
            for(int ivec=0;ivec<norb;ivec++){
                for(int jvec=0;jvec<norb;jvec++){
                    for(int icontract=0;icontract<norb;icontract++){
                        overlap_numeric[0][ivec][jvec]+=conj(vec_0[icontract][ivec])*vec_1[icontract][jvec];
                        overlap_numeric[1][ivec][jvec]+=conj(vec_0[icontract][ivec])*vec_2[icontract][jvec];
                        overlap_numeric[2][ivec][jvec]+=conj(vec_0[icontract][ivec])*vec_3[icontract][jvec];
                    }

                    //double abs1=cabs(overlap_numeric[0][ivec][jvec]);
                    //double abs2=cabs(overlap_numeric[1][ivec][jvec]);
                    //double abs3=cabs(overlap_numeric[2][ivec][jvec]);
                    //if(abs1>1e-6)overlap_numeric[0][ivec][jvec]/=abs1;
                    //if(abs2>1e-6)overlap_numeric[1][ivec][jvec]/=abs2;
                    //if(abs3>1e-6)overlap_numeric[2][ivec][jvec]/=abs3;
                }
            }
            /*
            if(ikpt<100){
            for(int idir=0;idir<3;idir++){
                printf("numeric overlap dir=%d\n",idir);
                for(int ivec=0;ivec<norb;ivec++){
                    for(int jvec=0;jvec<norb;jvec++)printf("%9.6f+%9.6fj ",creal(overlap_numeric[idir][ivec][jvec]),cimag(overlap_numeric[idir][ivec][jvec]));
                    printf("\n");
                }
            }}*/
            // calc analytic overlaps
            lapack_complex_double _dhkdk123[3][norb][norb];
            double _eig[norb];
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    _dhkdk123[0][i][j]=DHkDk1[index_base+i*norb+j];
                    _dhkdk123[1][i][j]=DHkDk2[index_base+i*norb+j];
                    _dhkdk123[2][i][j]=DHkDk3[index_base+i*norb+j];
                }
                _eig[i]=eigs[ikpt*norb+i];
            }
            
            lapack_complex_double overlap_analytic[3][norb][norb];
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    overlap_analytic[0][i][j]=0;
                    overlap_analytic[1][i][j]=0;
                    overlap_analytic[2][i][j]=0;
                }
            }
            for(int ivec=0;ivec<norb;ivec++){
                for(int jvec=0;jvec<norb;jvec++){
                    for(int icontract=0;icontract<norb;icontract++){
                        for(int jcontract=0;jcontract<norb;jcontract++){
                            overlap_analytic[0][ivec][jvec]+=conj(vec_0[icontract][ivec])*_dhkdk123[0][icontract][jcontract]*vec_0[jcontract][jvec];
                            overlap_analytic[1][ivec][jvec]+=conj(vec_0[icontract][ivec])*_dhkdk123[1][icontract][jcontract]*vec_0[jcontract][jvec];
                            overlap_analytic[2][ivec][jvec]+=conj(vec_0[icontract][ivec])*_dhkdk123[2][icontract][jcontract]*vec_0[jcontract][jvec];
                        }
                    }
                    //overlap_analytic[0][ivec][jvec]*=dk[0];
                    //overlap_analytic[1][ivec][jvec]*=dk[1];
                    //overlap_analytic[2][ivec][jvec]*=dk[2];
                    //double abs1=cabs(overlap_analytic[0][ivec][jvec]);
                    //double abs2=cabs(overlap_analytic[1][ivec][jvec]);
                    //double abs3=cabs(overlap_analytic[2][ivec][jvec]);
                    //if(abs1>1e-6)overlap_analytic[0][ivec][jvec]/=abs1;
                    //if(abs2>1e-6)overlap_analytic[1][ivec][jvec]/=abs2;
                    //if(abs3>1e-6)overlap_analytic[2][ivec][jvec]/=abs3;
                    /*if((ivec<=nocc & jvec>nocc)|(ivec>nocc & jvec<=nocc)){
                        overlap_analytic[0][ivec][jvec]/=(_eig[ivec]-_eig[jvec]);
                        overlap_analytic[1][ivec][jvec]/=(_eig[ivec]-_eig[jvec]);
                        overlap_analytic[2][ivec][jvec]/=(_eig[ivec]-_eig[jvec]);
                    }*/
                }
            }
            /*
            if(ikpt<100){
            for(int idir=0;idir<3;idir++){
                printf("analytic overlap dir=%d\n",idir);
                for(int ivec=0;ivec<norb;ivec++){
                    for(int jvec=0;jvec<norb;jvec++)printf("%9.6f+%9.6fj ",creal(overlap_analytic[idir][ivec][jvec]),cimag(overlap_analytic[idir][ivec][jvec]));
                    printf("\n");
                }
            }}*/
            // fix the phases
            int num_ref_states;
            lapack_complex_double ref_phase_shift;
            for(int ivec=0;ivec<norb;ivec++){
                //printf("band %d diag phase1: %f %f, abs: %f\n",ivec,creal(overlap_numeric[0]),cimag(overlap_numeric[0]),cabs(overlap_numeric[0]));
                //printf("band %d diag phase2: %f %f, abs: %f\n",ivec,creal(overlap_numeric[1]),cimag(overlap_numeric[1]),cabs(overlap_numeric[1]));
                /*
                overlap_numeric[0][ivec][ivec]/=cabs(overlap_numeric[0][ivec][ivec]);
                overlap_numeric[1][ivec][ivec]/=cabs(overlap_numeric[1][ivec][ivec]);
                overlap_numeric[2][ivec][ivec]/=cabs(overlap_numeric[2][ivec][ivec]);
                for(int i=0;i<norb;i++){
                    vec_1[i][ivec]/=overlap_numeric[0][ivec][ivec];
                    vec_2[i][ivec]/=overlap_numeric[1][ivec][ivec];
                    vec_3[i][ivec]/=overlap_numeric[2][ivec][ivec];
                }
                */
                //if(ikpt<100){
                num_ref_states=0;
                ref_phase_shift=0;
                //for(int i=0;i<norb;i++)ref_phase_shift[i]=0;
                //printf("phase shift for vec%d on k+dk1:\n",ivec);
                for(int jvec=0;jvec<norb;jvec++){ // jvec is the vector in k0
                    if(ivec==jvec)continue; // skip diagonal terms
                    if(abs(_eig[ivec]-_eig[jvec])<1e-6)continue; // skip degenerated terms
                    double numeric_abs=cabs(overlap_numeric[0][jvec][ivec]);
                    double analytic_abs=cabs(overlap_analytic[0][jvec][ivec]);
                    if(numeric_abs<1e-9 | analytic_abs<1e-9)continue; // skip zero amplitudes
                    ref_phase_shift+=overlap_analytic[0][jvec][ivec]/overlap_numeric[0][jvec][ivec]*dk[0]/(_eig[ivec]-_eig[jvec]);
                    //printf("%9.6f+%9.6fj (%9.6f)",creal(ref_phase_shift),cimag(ref_phase_shift),cabs(ref_phase_shift));
                    num_ref_states+=1;
                }
                //printf("\n");
                //}
                if(num_ref_states>0){
                    ref_phase_shift/=num_ref_states;
                    double ref_abs=cabs(ref_phase_shift);
                    if(ref_abs>1e-9)ref_phase_shift/=ref_abs;
                    for(int i=0;i<norb;i++)vec_1[i][ivec]*=ref_phase_shift;
                }

                num_ref_states=0;
                ref_phase_shift=0;
                for(int jvec=0;jvec<norb;jvec++){ // jvec is the vector in k0
                    if(ivec==jvec)continue; // skip diagonal terms
                    if(abs(_eig[ivec]-_eig[jvec])<1e-6)continue; // skip degenerated terms
                    double numeric_abs=cabs(overlap_numeric[1][jvec][ivec]);
                    double analytic_abs=cabs(overlap_analytic[1][jvec][ivec]);
                    if(numeric_abs<1e-9 | analytic_abs<1e-9)continue; // skip zero amplitudes
                    ref_phase_shift+=overlap_analytic[1][jvec][ivec]/overlap_numeric[1][jvec][ivec]*dk[1]/(_eig[ivec]-_eig[jvec]);
                    num_ref_states+=1;
                }
                if(num_ref_states>0){
                    ref_phase_shift/=num_ref_states;
                    double ref_abs=cabs(ref_phase_shift);
                    if(ref_abs>1e-9)ref_phase_shift/=ref_abs;
                    for(int i=0;i<norb;i++)vec_2[i][ivec]*=ref_phase_shift;
                }
            }
            
            lapack_complex_double dv_dk123[3][norb2];
            //printf("%f\n",vecs[index_base]);
            //printf("ikpt %d ik1 %d ik2 %d\n",ikpt,ik1,ik2);
            //int idata=0;
            for(int i=0;i<norb;i++){
                for(int j=0;j<norb;j++){
                    dv_dk123[0][i*norb+j]=(vec_1[i][j]-vec_0[i][j])/dk[0];
                    dv_dk123[1][i*norb+j]=(vec_2[i][j]-vec_0[i][j])/dk[1];
                    dv_dk123[2][i*norb+j]=(vec_3[i][j]-vec_0[i][j])/dk[2];
                    //idata+=1;
                }
            }
            //for(int i=0;i<2;i++){
            //    for(int j=0;j<2;j++)printf("%f %f,",creal(data_dk123[0][i*2+j]),cimag(data_dk123[0][i*2+j]));
            //    printf("\n");
            //}
            
            // change into derivates to kx,y,z
            lapack_complex_double dv_dkxyz[3][norb2];
            for(int i=0;i<3;i++)for(int idata=0;idata<norb2;idata++)dv_dkxyz[i][idata]=0;
            for(int idata=0;idata<norb2;idata++){
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)dv_dkxyz[i][idata]+=dv_dk123[j][idata]*lat[j][i];
            }
            //printf("%f\n",data_dkxyz[0][0]);
            for(int idata=0;idata<norb2;idata++){
                dvec_dkx[index_base+idata]=dv_dkxyz[0][idata];
                dvec_dky[index_base+idata]=dv_dkxyz[1][idata];
                dvec_dkz[index_base+idata]=dv_dkxyz[2][idata];
            }
        }
    }
    long long dvec_dkx_addr=(long long)dvec_dkx;
    long long dvec_dky_addr=(long long)dvec_dky;
    long long dvec_dkz_addr=(long long)dvec_dkz;
    if(sizeof(long long)!=sizeof(lapack_complex_double*)){
        printf("Error: in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(lapack_complex_double *));
        printf("they are different!\n");
        exit(1);
    }
    printf("in ham.dll the addr. of derivates are %p, %p, %p\n",dvec_dkx_addr,dvec_dky_addr,dvec_dkz_addr);
    return Py_BuildValue("(LLL)",dvec_dkx_addr,dvec_dky_addr,dvec_dkz_addr);
}

PyObject* readOutGridDataOnKMesh_double(long long data_addr, int dataDim, int kmesh[3]){
    double*gridData=(double*)data_addr;
    printf("ham.dll is to read out the double type data @ %p\n",gridData);

    int nkpt=kmesh[0]*kmesh[1]*kmesh[2];
    printf("kmesh, dim: %dx%dx%d x%d, data dim run fastest, then ik3, ik2, ik1\n",kmesh[0],kmesh[1],kmesh[2],dataDim);

    PyObject * outData=PyTuple_New(nkpt*dataDim);
    for(int ikpt=0;ikpt<nkpt;ikpt++)
    for(int idata=0;idata<dataDim;idata++)
    PyTuple_SET_ITEM(outData,ikpt*dataDim+idata,PyFloat_FromDouble(gridData[ikpt*dataDim+idata]));
    
    return Py_BuildValue("O",outData);
}

PyObject* readOutGridDataOnKMesh_complex(long long data_addr, int dataDim, int kmesh[3]){
    lapack_complex_double*gridData=(lapack_complex_double*)data_addr;
    printf("ham.dll is to read out the complex type data @ %p\n",gridData);

    int nkpt=kmesh[0]*kmesh[1]*kmesh[2];
    printf("kmesh, dim: %dx%dx%d x%d, data dim run fastest, then ik3, ik2, ik1\n",kmesh[0],kmesh[1],kmesh[2],dataDim);

    PyObject * outData_real=PyTuple_New(nkpt*dataDim);
    PyObject * outData_imag=PyTuple_New(nkpt*dataDim);
    for(int ikpt=0;ikpt<nkpt;ikpt++){
        for(int idata=0;idata<dataDim;idata++){
            PyTuple_SET_ITEM(outData_real,ikpt*dataDim+idata,PyFloat_FromDouble(creal(gridData[ikpt*dataDim+idata])));
            PyTuple_SET_ITEM(outData_imag,ikpt*dataDim+idata,PyFloat_FromDouble(cimag(gridData[ikpt*dataDim+idata])));
        }
    }
    return Py_BuildValue("(OO)",outData_real,outData_imag);
}

int * bubleSort_ABS(int len, double *data){
    int*idList=(int*)malloc(len*sizeof(int));
    for(int i=0;i<len;i++)idList[i]=i;
    //printf("init id: ");
    //for(int i=0;i<len;i++)printf("%d ",idList[i]);
    //printf("\n");
    int goON=1;
    int checkID,checkNextID;
    while(goON==1){
        goON=0;
        for(int i=0;i<len-1;i++){
            checkID=idList[i];
            checkNextID=idList[i+1];
            //printf("check %d and %d, %f<>%f\n",checkID,checkNextID,fabs(data[checkID]),fabs(data[checkNextID]));
            if(fabs(data[checkID])<fabs(data[checkNextID])){
                idList[i]=checkNextID;
                idList[i+1]=checkID;
                //printf("%d %d exchange to %d %d\n",checkID,checkNextID,idList[i],idList[i+1]);
                goON=1;
            }
        }
    }
    return idList;
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
                       double tau,
                       double summedCvgCriteria,
                       int _nthreads){
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    fclose(fp);
    
    double * spinBerryCurv_tmp=(double*)malloc(nfermi*nkpt*sizeof(double));
    
    double * kmesh_for_process=(double*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*3*sizeof(double));
    int * kpt_belongings=(int*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(int));
    int * kpt_belongings_nextProcess=(int*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(int));
    double * berryCurvOnKmesh=(double*)malloc(nkpt*sizeof(double));
    double * berryCurvOnKmesh_trial=(double*)malloc(nkpt*sizeof(double));
    double * berryCurvOnRefinedKmesh=(double*)malloc(nMaxRefine*nMaxRefine*nMaxRefine*nkpt*sizeof(double));
    
    int dim=1; // dimension of Brilloinzone
    if(dk[1]<0.6)dim=2;
    if(dk[2]<0.6)dim=3;
    int basic_factor=1; // basic multiply factor for grid refinement
    for(int i=0;i<dim;i++)basic_factor*=2;
    //printf("basic factor %d\n",basic_factor);
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
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"use %d cores in ham.dll\n",nthreads);
    fclose(fp);
    // calculate matrix elements
    //#pragma omp parallel num_threads(nthreads)
    {
        for(int ifermi=0;ifermi<nfermi;ifermi++){
            // init processing kmesh
            int nkptForProcess=nkpt;
            int nkptGroups;
            for(int ikpt=0;ikpt<nkpt;ikpt++){
                kmesh_for_process[ikpt*3+0]=kmesh[ikpt][0];
                kmesh_for_process[ikpt*3+1]=kmesh[ikpt][1];
                kmesh_for_process[ikpt*3+2]=kmesh[ikpt][2];
                kpt_belongings[ikpt]=ikpt;
            }
            // init original berry curv
            for(int ikpt=0;ikpt<nkpt;ikpt++)berryCurvOnKmesh[ikpt]=0.0; //init berryCurv
            int nrefine=1;
            int nStepRefine=0;
            int ngrids;
            // init original contribution
            int nDomination=0;
            int dominateKList[nkpt];
            for(int ikpt=0;ikpt<nkpt;ikpt++)dominateKList[ikpt]=0;
            while (0==0){
            //for(int nrefine=1;nrefine<=nMaxRefine;nrefine++){// loop over refinaries
                // first confirm the individual kpt groups
                int sizeOfGroups[nkpt];
                nkptGroups=0;
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
                    nkptGroups+=1;
                }else{
                    sizeOfGroups[nkptGroups]=nContribute;
                    nkptGroups+=1;
                    sizeOfGroups[nkptGroups]=1;
                    nkptGroups+=1;
                }
                //printf("found %d kpt groups\n",nkptGroups);

                int basic_factor_power_n=1;
                for(int i=0;i<nStepRefine;i++)basic_factor_power_n*=basic_factor;
                ngrids=(basic_factor_power_n-1)/(basic_factor-1);
                //
                
                // calc berry on (perharps) refined grids
                for(int ikpt=0;ikpt<nkptForProcess;ikpt++)berryCurvOnRefinedKmesh[ikpt]=0.0; // init berryCurv
                if(nStepRefine>0){
                    // quick check the size of refined grids
                    for(int igroup=0;igroup<nkptGroups;igroup++){
                        if(sizeOfGroups[igroup]!=basic_factor_power_n)printf("Error, size of refinements is not consistent with refinements steps\n");
                    }
                    printf("Iter: %d; on %d kpt groups; refine step %d with %d more grids, %d kpts in total...",nStepRefine,nkptGroups,nStepRefine,basic_factor_power_n,nkptForProcess);
                    fp=fopen("./c_log","a");
                    fprintf(fp,"Iter: %d; on %d kpt groups; refine step %d with %d more grids, %d kpts in total...",nStepRefine,nkptGroups,nStepRefine,basic_factor_power_n,nkptForProcess);
                    fclose(fp);
                }else{
                    printf("****  initial calc on fermi: %.4f, with %d kpt groups... ",fermiList[ifermi],nkptGroups);
                    fp=fopen("./c_log","a");
                    fprintf(fp,"****  initial calc on fermi: %.4f, with %d kpt groups... ",fermiList[ifermi],nkptGroups);
                    fclose(fp);
                }
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
                                    double deltaE2=deltaE*deltaE;
                                    double deltaE_square_double=deltaE2/(2*(deltaE2+tau*tau)*(deltaE2+tau*tau));
                                    if(Em>fermiList[ifermi]){
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
                                        berryCurvOnRefinedKmesh[ikpt]-=cimag(Amp1*Amp2)*deltaE_square_double;
                                    }
                                }
                            }
                        }
                    }
                }
                printf(" done\n");
                fp=fopen("./c_log","a");
                fprintf(fp," done\n");
                fclose(fp);
                double old_summed_berryCurv=0; // save the former berryCurv
                for(int ikpt=0;ikpt<nkpt;ikpt++)old_summed_berryCurv+=berryCurvOnKmesh[ikpt];
                old_summed_berryCurv/=(nkpt*TwoPI);

                if(nStepRefine==1){// here we check which kpts contribute mostly
                    //printf("start sorting...");
                    int*kIDList=bubleSort_ABS(nkpt,berryCurvOnKmesh);
                    double abs_sum_berryCurv=0;
                    for(int ikpt=0;ikpt<nkpt;ikpt++)abs_sum_berryCurv+=fabs(berryCurvOnKmesh[ikpt]);
                    //printf("done, %.6f\n",abs_sum_berryCurv);
                    
                    double cumulated_abs_berryCurv=0;
                    for(int ikpt=0;ikpt<nkpt;ikpt++){
                        cumulated_abs_berryCurv+=fabs(berryCurvOnKmesh[kIDList[ikpt]]);
                        //printf("id %d -> %d\n",ikpt,kIDList[ikpt]);
                        if(cumulated_abs_berryCurv<0.9*abs_sum_berryCurv){
                            dominateKList[kIDList[ikpt]]=1;
                            nDomination+=1;
                        }
                    }
                    if(nDomination>0){
                        printf("initial calc get <Curv>=%.6f, SUM_|Curv|=%.6E, 90 percent contributed by %d/%d kpts, cutoff |Curv|=%.6f\n",old_summed_berryCurv,abs_sum_berryCurv/TwoPI,nDomination,nkpt,fabs(berryCurvOnRefinedKmesh[kIDList[nDomination-1]])/TwoPI);
                        fp=fopen("./c_log","a");
                        fprintf(fp,"initial calc get <Curv>=%.6f SUM_|Curv|=%.6E, 90 percent contributed by %d/%d kpts, cutoff |Curv|=%.6f\n",old_summed_berryCurv,abs_sum_berryCurv/TwoPI,nDomination,nkpt,fabs(berryCurvOnRefinedKmesh[kIDList[nDomination-1]])/TwoPI);
                        fclose(fp);
                    }else{
                        printf("zero SUM_|Curv|\n");
                        fp=fopen("./c_log","a");
                        fprintf(fp,"zero SUM_|Curv|\n");
                        fclose(fp);
                    }
                    //for(int ikpt=0;ikpt<nkpt;ikpt++){
                    //    printf("kpt%d, contribution: %.6f, domination: %d\n",ikpt,berryCurvOnRefinedKmesh[ikpt],dominateKList[ikpt]);
                    //}
                }

                nStepRefine+=1;
                nrefine*=2;
                //accquire berry on original kmesh
                for(int ikpt=0;ikpt<nkpt;ikpt++)berryCurvOnKmesh_trial[ikpt]=0.0; //init berryCurv
                int nkptForNextProcess=0;
                int ikpt_in_refinedGrid=0;
                for(int igroup=0;igroup<nkptGroups;igroup++){
                    ikpt_contribute_now=kpt_belongings[ikpt_in_refinedGrid];
                    for(int ikpt_ingroup=0;ikpt_ingroup<sizeOfGroups[igroup];ikpt_ingroup++){
                        berryCurvOnKmesh_trial[ikpt_contribute_now]+=berryCurvOnRefinedKmesh[ikpt_in_refinedGrid];
                        //printf("kpt%d group, kpt%d in refined mesh\n",ikpt_contribute_now,ikpt_in_refinedGrid);
                        //printf("      contributing berry curv of %.6f\n",berryCurvOnRefinedKmesh[ikpt_in_refinedGrid]);
                        ikpt_in_refinedGrid+=1;
                    }
                    if(nStepRefine>1){ // calc avg. value in densed grids for non-initial steps
                        berryCurvOnKmesh_trial[ikpt_contribute_now]=(berryCurvOnKmesh[ikpt_contribute_now]*ngrids+berryCurvOnKmesh_trial[ikpt_contribute_now])/(ngrids+sizeOfGroups[igroup]);///=sizeOfGroups[igroup];
                    }
                    //printf("kpt%d group finished, with %d grids, berry curv: %.3f\n",ikpt_contribute_now,sizeOfGroups[igroup],berryCurvOnKmesh_trial[ikpt_contribute_now]);
                    // compare the result and check if we need to refine grid more
                    if(nStepRefine>1 & (dominateKList[ikpt_contribute_now]==0 | fabs((berryCurvOnKmesh_trial[ikpt_contribute_now]-berryCurvOnKmesh[ikpt_contribute_now])/berryCurvOnKmesh[ikpt_contribute_now])<0.1)){
                        // update
                        berryCurvOnKmesh[ikpt_contribute_now]=berryCurvOnKmesh_trial[ikpt_contribute_now];
                        //printf("kpt%d group converged\n",ikpt_contribute_now);
                    }else{
                        // update
                        if(nrefine<=nMaxRefine){
                           // create/add refined kmesh
                            double ddk[3];// interval of refined grid
                            for(int i=0;i<3;i++)ddk[i]=dk[i]/nrefine;
                            double odk[3];// extra shift to locate refined grid around center
                            for(int i=0;i<3;i++)odk[i]=ddk[i]*(nrefine-1)/2;
                            //if(nrefine>1)printf("create refined grids for kpt%d group centered in %.3f %.3f %.3f\n",ikpt_contribute_now,kmesh[ikpt_contribute_now][0],kmesh[ikpt_contribute_now][1],kmesh[ikpt_contribute_now][2]);
                            if(dim==3){ // dim==3
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
                            }else if(dim==2){
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
                            }else{
                                printf("Error, 1-dimensional BZ is still in development\n");
                                exit(1);
                            }
                        }else if(nrefine>2){
                            printf("-->  cvg. failed at fermi %.4f, ikpt %d, [%.4f %.4f %.4f], final ratio: %.6f/%.6f\n",fermiList[ifermi],ikpt_contribute_now,kmesh[ikpt_contribute_now][0],kmesh[ikpt_contribute_now][1],kmesh[ikpt_contribute_now][2],berryCurvOnKmesh_trial[ikpt_contribute_now],berryCurvOnKmesh[ikpt_contribute_now]);
                            fp=fopen("./c_log","a");
                            fprintf(fp,"-->  cvg. failed at fermi %.4f, ikpt %d, [%.4f %.4f %.4f], final ratio: %.6f/%.6f\n",fermiList[ifermi],ikpt_contribute_now,kmesh[ikpt_contribute_now][0],kmesh[ikpt_contribute_now][1],kmesh[ikpt_contribute_now][2],berryCurvOnKmesh_trial[ikpt_contribute_now],berryCurvOnKmesh[ikpt_contribute_now]);
                            fclose(fp);
                        }
                        berryCurvOnKmesh[ikpt_contribute_now]=berryCurvOnKmesh_trial[ikpt_contribute_now];
                    }
                }// analysis on data on refined kmesh fininished
                
                double new_summed_berryCurv=0; // save the former berryCurv
                for(int ikpt=0;ikpt<nkpt;ikpt++)new_summed_berryCurv+=berryCurvOnKmesh[ikpt];
                new_summed_berryCurv/=(nkpt*TwoPI);
                double d_summed_berryCurv=fabs(new_summed_berryCurv-old_summed_berryCurv);
                if(nStepRefine>1){
                    printf("--> <Curv>: %.6f on %d grids, and %.6f on %d grids, diff: %.4f\n",new_summed_berryCurv,ngrids+basic_factor_power_n,old_summed_berryCurv,ngrids,d_summed_berryCurv);
                    fp=fopen("./c_log","a");
                    fprintf(fp,"--> <Curv>: %.6f on %d grids, and %.6f on %d grids, diff: %.4f\n",new_summed_berryCurv,ngrids+basic_factor_power_n,old_summed_berryCurv,ngrids,d_summed_berryCurv);
                    fclose(fp);
                }

                if(nStepRefine>1 & (nkptForNextProcess==0 | nrefine>nMaxRefine | d_summed_berryCurv<summedCvgCriteria)){ // stop the loop due to either convergence or max refinement reached
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

PyObject * calcJointDOS(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       int _nthreads){
    printf("entering JointDOS module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.4f MB\n",(double)(nphoton*nkpt+norbs*nkpt)/1024/1024*sizeof(double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(double)(nphoton*nkpt+norbs*nkpt)/1024/1024*sizeof(double));
    
    lapack_complex_double itau=tau*_Complex_I;
    double *energy=(double*)malloc(nkpt*norbs*sizeof(double));
    double * JDOS_on_kmesh_on_omega=(double*)malloc(nphoton*nkpt*sizeof(double));
    for(int i=0;i<nphoton*nkpt;i++)JDOS_on_kmesh_on_omega[i]=0;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"solve eigen equations using %d cores in ham.dll\n",nthreads);
    fclose(fp);

    printf("solve eigen equations using %d cores in ham.dll\n",nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            lapack_complex_double hk[norbs][norbs]; // init Hk
            for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++)hk[i][j]=0;
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                lapack_complex_double hopping_R[3];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);
                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;   
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            if(info!=0){
                printf("LAPACKE failed, info: %d\n",info);
                exit(1);
            }
            for(int bandm=0;bandm<norbs;bandm++)energy[ikpt*norbs+bandm]=eigs[bandm];
        }
    }
    fp=fopen("./c_log","a");
    fprintf(fp,"Eigenvalues solved\ncalc JDOS...\n");
    fclose(fp);
    printf("Eigenvalues solved\nnow calc JDOS use %d cores in ham.dll\n",nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l the conduction bands
                double El=energy[ikpt*norbs+bandl];
                if(El<erange[0] | El>erange[1] | El<=fermi) continue;
                for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m the valence bands
                    double En=energy[ikpt*norbs+bandn];
                    if(En<erange[0] | En>erange[1] | En>fermi) continue;
                    // calc JDOS
                    for(int iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                        double omega=omegaList[iphoton];
                        JDOS_on_kmesh_on_omega[iphoton*nkpt+ikpt]+=cimag(1./(El-En-omega-itau));
                    }
                }
            }
        }
    }
    printf("JDOS done\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"JDOS done\n");
    fclose(fp);
    free(energy);

    PyObject * JDOS=PyTuple_New(nphoton*nkpt);
    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            PyTuple_SetItem(JDOS, iphoton*nkpt+ikpt, PyFloat_FromDouble(JDOS_on_kmesh_on_omega[iphoton*nkpt+ikpt]));
        }
    }
    free(JDOS_on_kmesh_on_omega);
    return Py_BuildValue("O",JDOS);
}

PyObject * calcBulkPhotovoltaicCond_memoryReduce(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       int ndivide, // num. of parts
                       int _nthreads){
    printf("entering BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.3f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*sizeof(lapack_complex_double));
    //exit(1);
    
    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(nphoton*nkpt*8*sizeof(lapack_complex_double));
    for(int i=0;i<nphoton*nkpt*8;i++)conductance_on_kmesh_on_omega[i]=0;
    
    int nkpt_part=nkpt/ndivide;
    int ikpt_head;
    if(nkpt%ndivide!=0){
        printf("Error: nkpt %d cannot be divide by %d, exiting now!\n",nkpt,ndivide);
        exit(0);
    }
    for(int ipart=0;ipart<ndivide;ipart++){// start kmesh partition
        ikpt_head=nkpt_part*ipart;
        lapack_complex_double *velocity=(lapack_complex_double*)malloc(2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
        double *energy=(double*)malloc(nkpt_part*norbs*sizeof(double));

        int kpt_width=norbs*norbs;
        int dir_width=nkpt_part*kpt_width;

        int nthreads;
        nthreads=_nthreads;
        if(nkpt_part<_nthreads*10)nthreads=1;
        printf("use %d cores in ham.dll\n",nthreads);
        fp=fopen("./c_log","a");
        fprintf(fp,"calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        fclose(fp);

        printf("calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        #pragma omp parallel num_threads(nthreads)
        {
            lapack_complex_double * hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
            lapack_complex_double * DhkDkf=(lapack_complex_double*)malloc(norbs*norbs*3*sizeof(lapack_complex_double));
            lapack_complex_double * DhkDkc=(lapack_complex_double*)malloc(norbs*norbs*2*sizeof(lapack_complex_double));
            int ipos,ipos1,ipos2,ipos3,ipos4,ipos1_s,ipos2_s,ipos3_s,ipos4_s;
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                //lapack_complex_double hk[norbs][norbs]; // init Hk
                //lapack_complex_double Dhk[norbs][norbs][3]; // init DHk/Dk1,Dk2
                //lapack_complex_double DhkDkxy[norbs][norbs][2]; // init DHk/Dkx,Dky
                double Em, En;
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        hk[i*norbs+j]=0;
                        //Dhk[i][j][0]=0;
                        //Dhk[i][j][1]=0;
                        //Dhk[i][j][2]=0;
                        //DhkDkxy[i][j][0]=0;
                        //DhkDkxy[i][j][1]=0;
                        DhkDkf[i*norbs*3+j*3+0]=0;
                        DhkDkf[i*norbs*3+j*3+1]=0;
                        DhkDkf[i*norbs*3+j*3+2]=0;
                        DhkDkc[i*norbs*2+j*2+0]=0;
                        DhkDkc[i*norbs*2+j*2+1]=0;
                    }
                }
                
                double eigs[norbs]; // output eigenvalues of Hk
                for(int ihopping=0;ihopping<nhoppings;ihopping++){
                    lapack_complex_double hopping_R[3];
                    for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);
                
                    lapack_complex_double phase=0;
                    for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt_head+ikpt][i];
                
                    lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                    //hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                    hk[sourceID[ihopping]*norbs+targetID[ihopping]]+=tk;
                    //for(int i=0;i<3;i++)Dhk[sourceID[ihopping]][targetID[ihopping]][i]+=tk*hopping_R[i];
                    for(int i=0;i<3;i++)DhkDkf[sourceID[ihopping]*norbs*3+targetID[ihopping]*3+i]+=tk*hopping_R[i];
                }
            
                // unitary transform into real space partial
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<3;k++){
                            //DhkDkxy[i][j][0]+=Dhk[i][j][k]*lattice[k][0];
                            //DhkDkxy[i][j][1]+=Dhk[i][j][k]*lattice[k][1];
                            DhkDkc[i*norbs*2+j*2+0]+=DhkDkf[i*norbs*3+j*3+k]*lattice[k][0];
                            DhkDkc[i*norbs*2+j*2+1]+=DhkDkf[i*norbs*3+j*3+k]*lattice[k][1];
                        }
                    }
                }
            
                //int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
                int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,hk,norbs,eigs);
                // vec[i]=hk[:][i]
                if(info!=0){printf("LAPACKE failed, info: %d\n",info);exit(1);}
                for(int bandm=0;bandm<norbs;bandm++){
                    Em=eigs[bandm];
                    energy[ikpt*norbs+bandm]=Em;
                    if(Em<erange[0] | Em>erange[1]) continue;
                    for(int bandn=0;bandn<norbs;bandn++){
                        En=eigs[bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        for(int idir=0;idir<2;idir++){
                            ipos=idir*dir_width+ikpt*kpt_width+bandm*norbs+bandn;
                            velocity[ipos]=0;
                            for(int icontract=0;icontract<norbs;icontract++){
                                for(int jcontract=0;jcontract<norbs;jcontract++){
                                    //velocity[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkxy[icontract][jcontract][idir]*hk[jcontract*norbs+bandn];
                                    velocity[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc[icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                }
                            }
                        }
                    }
                }  
            }
            free(hk);free(DhkDkf);free(DhkDkc);
        }
        fp=fopen("./c_log","a");
        fprintf(fp,"velocities part%d completed\ncalc BPVE...\n",ipart+1);
        fclose(fp);
        printf("velocities part%d completed\nnow calc BPVE use %d cores in ham.dll\n",ipart+1,nthreads);

        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                int ikpt_true=ikpt_head+ikpt;
                int writing_k_index=ikpt_true*nphoton*8;
                for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                    double El=energy[ikpt*norbs+bandl];
                    if(El<erange[0] | El>erange[1]) continue;
                    int fl=0;
                    if(El<=fermi)fl=1;
                    for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                        double En=energy[ikpt*norbs+bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        int fn=0;
                        if(En<=fermi)fn=1;
                        int fln=fl-fn;
                        if (fln==0) continue;

                        // calc overlap
                        lapack_complex_double px_nl=velocity[          ikpt*kpt_width+bandn*norbs+bandl];
                        lapack_complex_double py_nl=velocity[dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                        //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                        for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                            //if(bandn!=bandl)continue; // switch for bandn
                            double Em=energy[ikpt*norbs+bandm];
                            if(Em<erange[0] | Em>erange[1]) continue;
                            // calc overlap
                            lapack_complex_double px_lm=velocity[          ikpt*kpt_width+bandl*norbs+bandm];
                            lapack_complex_double py_lm=velocity[dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                            lapack_complex_double px_mn=velocity[          ikpt*kpt_width+bandm*norbs+bandn];
                            lapack_complex_double py_mn=velocity[dir_width+ikpt*kpt_width+bandm*norbs+bandn];

                            for(int iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                                int writing_ph_index=writing_k_index+iphoton*8;
                                double omega=omegaList[iphoton];
                            
                                lapack_complex_double factor_global=0;
                                factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                                conductance_on_kmesh_on_omega[writing_ph_index  ]+=px_nl*px_lm*px_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+1]+=px_nl*px_lm*py_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+2]+=py_nl*py_lm*px_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+3]+=py_nl*py_lm*py_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+4]+=px_nl*py_lm*px_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+5]+=px_nl*py_lm*py_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+6]+=py_nl*px_lm*px_mn*factor_global;
                                conductance_on_kmesh_on_omega[writing_ph_index+7]+=py_nl*px_lm*py_mn*factor_global;
                            }
                        } 
                    }
                }
            }
        }
        printf("BPVE part%d done\n",ipart+1);
        fp=fopen("./c_log","a");
        fprintf(fp,"BPVE part%d done\n",ipart+1);
        fclose(fp);
        free(velocity);free(energy);
    }

    printf("All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);

    PyObject * conductance_real=PyTuple_New(nphoton*nkpt*8);
    PyObject * conductance_imag=PyTuple_New(nphoton*nkpt*8);

    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            for(int icomp=0;icomp<8;icomp++){
                PyTuple_SetItem(conductance_real, icomp*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+icomp])));
                PyTuple_SetItem(conductance_imag, icomp*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+icomp])));
            }
        }
    }
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

PyObject * calcShiftCond(int nkpt, double kmesh[nkpt][3], double dk[3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       int _nthreads){
    printf("entering shift BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, dk: %.2f %.2f %.2f, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,dk[0],dk[1],dk[2],norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.4f MB\n",(double)(nphoton*nkpt*16+(2*norbs*norbs+4*norbs*norbs+norbs)*nkpt)/1024/1024*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, dk: %.2f %.2f %.2f, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,dk[0],dk[1],dk[2],norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(double)(nphoton*nkpt*16+(2*norbs*norbs+4*norbs*norbs+norbs)*nkpt)/1024/1024*sizeof(lapack_complex_double));
    
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(2*nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *bi_velocity=(lapack_complex_double*)malloc(4*nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *energy=(lapack_complex_double*)malloc(nkpt*norbs*sizeof(lapack_complex_double));

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(nphoton*nkpt*8*sizeof(lapack_complex_double));
    for(int i=0;i<nphoton*nkpt*8;i++)conductance_on_kmesh_on_omega[i]=0;

    int kpt_width=norbs*norbs;
    int dir_width=nkpt*kpt_width;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"calc velocity and bi-velocity using %d cores in ham.dll\n",nthreads);
    fclose(fp);

    printf("calc velocities using %d cores in ham.dll\n",nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double DhkDkxy[norbs][norbs][2]; // init DHk/Dkx,Dky
            lapack_complex_double DhkDk1xyDk2xy[norbs][norbs][2][2]; // init D^2Hk/Dkx,Dky
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i][j]=0;
                    for(int u=0;u<2;u++)DhkDkxy[i][j][u]=0;
                    for(int u=0;u<2;u++)for(int v=0;v<2;v++)DhkDk1xyDk2xy[i][j][u][v]=0;
                }
            }
            // construct Ham_k, DHam, and DDHam
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                lapack_complex_double hopping_R[3];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                
                lapack_complex_double hopping_R_c[3];
                for(int i=0;i<3;i++)hopping_R_c[i]=0;
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)hopping_R_c[i]+=hopping_R[j]*lattice[j][i];

                for(int i=0;i<2;i++) DhkDkxy[sourceID[ihopping]][targetID[ihopping]][i]+=tk*hopping_R_c[i];
                for(int i=0;i<2;i++)for(int j=0;j<2;j++)DhkDk1xyDk2xy[sourceID[ihopping]][targetID[ihopping]][i][j]+=tk*hopping_R_c[i]*hopping_R_c[j];
            }

            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);

            int ipos;
            for(int bandm=0;bandm<norbs;bandm++){
                energy[ikpt*norbs+bandm]=eigs[bandm];
                for(int bandn=0;bandn<norbs;bandn++){
                    for(int idir=0;idir<2;idir++){
                        ipos=idir*dir_width+ikpt*kpt_width+bandm*norbs+bandn;
                        velocity[ipos]=0;
                        for(int icontract=0;icontract<norbs;icontract++){
                            for(int jcontract=0;jcontract<norbs;jcontract++){
                                velocity[ipos]+=conj(hk[icontract][bandm])*DhkDkxy[icontract][jcontract][idir]*hk[jcontract][bandn];
                            }
                        }
                        for(int jdir=0;jdir<2;jdir++){
                            ipos=(idir*2+jdir)*dir_width+ikpt*kpt_width+bandm*norbs+bandn;
                            bi_velocity[ipos]=0;
                            for(int icontract=0;icontract<norbs;icontract++){
                                for(int jcontract=0;jcontract<norbs;jcontract++){
                                    bi_velocity[ipos]+=conj(hk[icontract][bandm])*DhkDk1xyDk2xy[icontract][jcontract][idir][jdir]*hk[jcontract][bandn];
                                }
                            }
                        }
                    }
                }
            }  
        }
    }
    fp=fopen("./c_log","a");
    fprintf(fp,"velocity completed\ncalc BPVE...\n");
    fclose(fp);
    printf("velocity completed\nnow calc shift BPVE using %d cores in ham.dll\n",nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int writing_k_index=ikpt*nphoton*8;
            for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                double El=energy[ikpt*norbs+bandl];
                if(El<erange[0] | El>erange[1]) continue;
                int fl=0;
                if(El<=fermi)fl=1;
                for(int bandn=0;bandn<norbs;bandn++){ // loop over band_n
                    double En=energy[ikpt*norbs+bandn];
                    if(En<erange[0] | En>erange[1]) continue;
                    int fn=0;
                    if(En<=fermi)fn=1;
                    int fln=fl-fn;
                    if (fln==0) continue;
                    double Enl=En-El;
                    double Eln=-Enl;
                    // calc overlap
                    lapack_complex_double vx_nl=velocity[          ikpt*kpt_width+bandn*norbs+bandl];
                    lapack_complex_double vy_nl=velocity[dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                    lapack_complex_double vx_nn=velocity[          ikpt*kpt_width+bandn*norbs+bandn];
                    lapack_complex_double vy_nn=velocity[dir_width+ikpt*kpt_width+bandn*norbs+bandn];
                    lapack_complex_double vx_ll=velocity[          ikpt*kpt_width+bandl*norbs+bandl];
                    lapack_complex_double vy_ll=velocity[dir_width+ikpt*kpt_width+bandl*norbs+bandl];
                    
                    lapack_complex_double vx_ln=conj(vx_nl);
                    lapack_complex_double vy_ln=conj(vy_nl);

                    lapack_complex_double rx_nl=_Complex_I*vx_nl/Enl;
                    lapack_complex_double ry_nl=_Complex_I*vy_nl/Enl;

                    lapack_complex_double wxx_ln=bi_velocity[            ikpt*kpt_width+bandl*norbs+bandn];
                    lapack_complex_double wxy_ln=bi_velocity[  dir_width+ikpt*kpt_width+bandl*norbs+bandn];
                    lapack_complex_double wyx_ln=bi_velocity[2*dir_width+ikpt*kpt_width+bandl*norbs+bandn];
                    lapack_complex_double wyy_ln=bi_velocity[3*dir_width+ikpt*kpt_width+bandl*norbs+bandn];
                    //wxx_ln=0;wxy_ln=0;wyx_ln=0;wyy_ln=0;

                    lapack_complex_double SIGMA_xx_ln=0;
                    lapack_complex_double SIGMA_xy_ln=0;
                    lapack_complex_double SIGMA_yx_ln=0;
                    lapack_complex_double SIGMA_yy_ln=0;
                    for(int bandm=0;bandm<norbs;bandm++){
                        if(bandm==bandn|bandm==bandl)continue;
                        double Em=energy[ikpt*norbs+bandm];
                        if(En<erange[0] | En>erange[1]) continue;
                        lapack_complex_double vx_lm=velocity[          ikpt*kpt_width+bandl*norbs+bandm];
                        lapack_complex_double vy_lm=velocity[dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                        lapack_complex_double vx_mn=velocity[          ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double vy_mn=velocity[dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                        
                        double Eml=Em-El;//-itau;
                        double Enm=En-Em;//-itau;
                        double Eml_reverse, Enm_reverse;
                        if(fabs(Eml)<1e-7){
                            Eml_reverse=0;
                        }else{
                            Eml_reverse=1/Eml;
                        }
                        if(fabs(Enm)<1e-7){
                            Enm_reverse=0;
                        }else{
                            Enm_reverse=1/Enm;
                        }
                        
                        SIGMA_xx_ln+=vx_lm*vx_mn*(Eml_reverse+Enm_reverse);
                        SIGMA_xy_ln+=vx_lm*vy_mn*Eml_reverse+vy_lm*vx_mn*Enm_reverse;
                        SIGMA_yx_ln+=vy_lm*vx_mn*Eml_reverse+vx_lm*vy_mn*Enm_reverse;
                        SIGMA_yy_ln+=vy_lm*vy_mn*(Eml_reverse+Enm_reverse);

                    }
                    lapack_complex_double rxx_ln=_Complex_I/Eln*(
                        (vx_ln*(vx_ll-vx_nn)+vx_ln*(vx_ll-vx_nn))/Eln-wxx_ln+SIGMA_xx_ln);
                    lapack_complex_double rxy_ln=_Complex_I/Eln*(
                        (vx_ln*(vy_ll-vy_nn)+vy_ln*(vx_ll-vx_nn))/Eln-wxy_ln+SIGMA_xy_ln);
                    lapack_complex_double ryx_ln=_Complex_I/Eln*(
                        (vy_ln*(vx_ll-vx_nn)+vx_ln*(vy_ll-vy_nn))/Eln-wyx_ln+SIGMA_yx_ln);
                    lapack_complex_double ryy_ln=_Complex_I/Eln*(
                        (vy_ln*(vy_ll-vy_nn)+vy_ln*(vy_ll-vy_nn))/Eln-wyy_ln+SIGMA_yy_ln);

                    for(int iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                        int writing_ph_index=writing_k_index+iphoton*8;
                        double omega=omegaList[iphoton];
                            
                        lapack_complex_double factor_global=0;
                        //if(fabs(En-El)>1e-6)
                        //factor_global=-fln*(creal(1/(Enl-omega-itau))+_Complex_I*cimag(1/(Enl-omega-itau)))/2;
                        //factor_global=-fln*(_Complex_I*cimag(1/(Enl-omega-itau)))/2;
                        lapack_complex_double factor_global1=fln*(1/(Enl-omega-itau))/2;
                        lapack_complex_double factor_global2=fln*(1/(Enl+omega+itau))/2;
                        //printf("ik%d omega %.4f El:%.4f Em:%.4f En:%.4f fln:%d dEmn:%.4f dEnl:%.4f factor: %.4f,%.4f\n",ikpt,omega,El,Em,En,fln,Em-En,En-El,creal(factor_global),cimag(factor_global));
                        //printf("   l,m,n: %d %d %d -- px_nl: %.4f, %.4f px_lm: %.4f, %.4f px_mn: %.4f, %.4f\n",bandl,bandm,bandn,creal(px_nl),cimag(px_nl),creal(px_lm),cimag(px_lm),creal(px_mn),cimag(px_mn));
                        //printf("                   py_nl: %.4f, %.4f py_lm: %.4f, %.4f py_mn: %.4f, %.4f\n",creal(py_nl),cimag(py_nl),creal(py_lm),cimag(py_lm),creal(py_mn),cimag(py_mn));
                        //printf("                   xxx: %.4f, %.4f yyx: %.4f, %.4f\n",creal(px_nl*px_lm*px_mn),cimag(px_nl*px_lm*px_mn),creal(py_nl*py_lm*px_mn),cimag(py_nl*py_lm*px_mn));
                        
                        conductance_on_kmesh_on_omega[writing_ph_index  ]+=(rx_nl*rxx_ln*factor_global1+rx_nl*rxx_ln*factor_global2);//*factor_global; //x_xx
                        conductance_on_kmesh_on_omega[writing_ph_index+1]+=(rx_nl*rxy_ln*factor_global1+rx_nl*rxy_ln*factor_global2);//*factor_global; //y_xx
                        conductance_on_kmesh_on_omega[writing_ph_index+2]+=(ry_nl*ryx_ln*factor_global1+ry_nl*ryx_ln*factor_global2);//*factor_global; //x_yy
                        conductance_on_kmesh_on_omega[writing_ph_index+3]+=(ry_nl*ryy_ln*factor_global1+ry_nl*ryy_ln*factor_global2);//*factor_global; //y_yy
                        conductance_on_kmesh_on_omega[writing_ph_index+4]+=(rx_nl*ryx_ln*factor_global1+ry_nl*rxx_ln*factor_global2);//*factor_global; //x_xy
                        conductance_on_kmesh_on_omega[writing_ph_index+5]+=(rx_nl*ryy_ln*factor_global1+ry_nl*rxy_ln*factor_global2);//*factor_global; //y_xy
                        conductance_on_kmesh_on_omega[writing_ph_index+6]+=(ry_nl*rxx_ln*factor_global1+rx_nl*ryx_ln*factor_global2);//*factor_global; //x_yx
                        conductance_on_kmesh_on_omega[writing_ph_index+7]+=(ry_nl*rxy_ln*factor_global1+rx_nl*ryy_ln*factor_global2);//*factor_global; //y_yx
                    }
                    
                }
            }
        }
    }
    printf("shift BPVE done\n");
    printf("clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fprintf(fp,"shift BPVE done\n");
    fclose(fp);
    free(bi_velocity);free(velocity);free(energy);

    PyObject * conductance_real=PyTuple_New(nphoton*nkpt*8);
    PyObject * conductance_imag=PyTuple_New(nphoton*nkpt*8);

    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            PyTuple_SetItem(conductance_real, 0*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8  ])));
            PyTuple_SetItem(conductance_real, 1*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+1])));
            PyTuple_SetItem(conductance_real, 2*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+2])));
            PyTuple_SetItem(conductance_real, 3*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+3])));
            PyTuple_SetItem(conductance_real, 4*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+4])));
            PyTuple_SetItem(conductance_real, 5*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+5])));
            PyTuple_SetItem(conductance_real, 6*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+6])));
            PyTuple_SetItem(conductance_real, 7*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+7])));

            PyTuple_SetItem(conductance_imag, 0*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8  ])));
            PyTuple_SetItem(conductance_imag, 1*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+1])));
            PyTuple_SetItem(conductance_imag, 2*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+2])));
            PyTuple_SetItem(conductance_imag, 3*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+3])));
            PyTuple_SetItem(conductance_imag, 4*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+4])));
            PyTuple_SetItem(conductance_imag, 5*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+5])));
            PyTuple_SetItem(conductance_imag, 6*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+6])));
            PyTuple_SetItem(conductance_imag, 7*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+7])));
        }
    }
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

PyObject * calcInjectCond(int nkpt, double kmesh[nkpt][3], double dk[3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       int _nthreads){
    printf("now in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, dk: %.2f %.2f %.2f, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,dk[0],dk[1],dk[2],norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.4f MB\n",nphoton*nkpt/1024*8*sizeof(lapack_complex_double)*2/1024);
    fclose(fp);

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(nphoton*nkpt*8*sizeof(lapack_complex_double));
    for(int i=0;i<nphoton*nkpt*8;i++)conductance_on_kmesh_on_omega[i]=0;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"use %d cores in ham.dll\n",nthreads);
    fclose(fp);

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int writing_k_index=ikpt*nphoton*8;
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double Dhk[norbs][norbs][3]; // init DHk/Dk1,Dk2
            lapack_complex_double DhkDkxy[norbs][norbs][2]; // init DHk/Dkx,Dky
            //lapack_complex_double SDhk_x[norbs][norbs]; // init S \dot DHk/Dkx
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i][j]=0;
                    for(int u=0;u<3;u++)Dhk[i][j][u]=0;
                    for(int u=0;u<2;u++)DhkDkxy[i][j][u]=0;
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                lapack_complex_double hopping_R[3];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
                
                lapack_complex_double hopping_R_c[3];
                for(int i=0;i<3;i++)hopping_R_c[i]=0;
                for(int i=0;i<3;i++)for(int j=0;j<3;j++)hopping_R_c[i]+=hopping_R[j]*lattice[j][i];

                //for(int i=0;i<3;i++) Dhk[sourceID[ihopping]][targetID[ihopping]][0]+=tk*hopping_R[i];
                for(int i=0;i<2;i++) DhkDkxy[sourceID[ihopping]][targetID[ihopping]][i]+=tk*hopping_R_c[i];
            }
            // unitary transform into real space partial, note each lattice[k] represent one crystal vector with three xyz components
            /*
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    for(int k=0;k<3;k++){
                        DhkDkxy[i][j][0]+=Dhk[i][j][k]*lattice[k][0];
                        DhkDkxy[i][j][1]+=Dhk[i][j][k]*lattice[k][1];
                        for(int p=0;p<3;p++){
                            DhkDk1xyDk2xy[i][j][0][0]+=Dhk1k2[i][j][k][p]*lattice[k][0]*lattice[k][0];
                            DhkDk1xyDk2xy[i][j][0][1]+=Dhk1k2[i][j][k][p]*lattice[k][0]*lattice[k][1];
                            DhkDk1xyDk2xy[i][j][1][0]+=Dhk1k2[i][j][k][p]*lattice[k][1]*lattice[k][0];
                            DhkDk1xyDk2xy[i][j][1][1]+=Dhk1k2[i][j][k][p]*lattice[k][1]*lattice[k][1];
                        }
                    }
                }
            }*/

            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);

            for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                double El=eigs[bandl];
                if(El<erange[0] | El>erange[1]) continue;
                int fl=0;
                if(El<=fermi)fl=1;
                for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                    double En=eigs[bandn];
                    if(En<erange[0] | En>erange[1]) continue;
                    int fn=0;
                    if(En<=fermi)fn=1;
                    int fln=fl-fn;
                    if (fln==0) continue;
                    double Enl=En-El;
                    double Eln=-Enl;
                    // calc overlap
                    lapack_complex_double vx_ln=0;
                    lapack_complex_double vy_ln=0;
                    lapack_complex_double vx_ll=0;
                    lapack_complex_double vy_ll=0;
                    lapack_complex_double vx_nn=0;
                    lapack_complex_double vy_nn=0;
                    for(int i=0;i<norbs;i++){
                        for(int j=0;j<norbs;j++){
                            vx_ln+=conj(hk[j][bandl])*DhkDkxy[j][i][0]*hk[i][bandn];
                            vy_ln+=conj(hk[j][bandl])*DhkDkxy[j][i][1]*hk[i][bandn];
                            vx_ll+=conj(hk[j][bandl])*DhkDkxy[j][i][0]*hk[i][bandl];
                            vy_ll+=conj(hk[j][bandl])*DhkDkxy[j][i][1]*hk[i][bandl];
                            vx_nn+=conj(hk[j][bandn])*DhkDkxy[j][i][0]*hk[i][bandn];
                            vy_nn+=conj(hk[j][bandn])*DhkDkxy[j][i][1]*hk[i][bandn];
                        }
                    }
                    lapack_complex_double dvx_ln=vx_ll-vx_nn;
                    lapack_complex_double dvy_ln=vy_ll-vy_nn;

                    lapack_complex_double rx_ln=-_Complex_I*vx_ln/Enl;
                    lapack_complex_double ry_ln=-_Complex_I*vy_ln/Enl;
                    lapack_complex_double rx_nl=conj(rx_ln);
                    lapack_complex_double ry_nl=conj(ry_ln);

                    

                    for(int iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                        int writing_ph_index=writing_k_index+iphoton*8;
                        double omega=omegaList[iphoton];
                            
                        //lapack_complex_double factor_global=0;
                        //if(fabs(En-El)>1e-6)
                        lapack_complex_double factor_global=-_Complex_I*fln*cimag(1/(Enl-omega-itau));
                        conductance_on_kmesh_on_omega[writing_ph_index  ]+=dvx_ln*rx_ln*rx_nl*factor_global; //x_xx
                        conductance_on_kmesh_on_omega[writing_ph_index+1]+=dvy_ln*rx_ln*rx_nl*factor_global; //y_xx
                        conductance_on_kmesh_on_omega[writing_ph_index+2]+=dvx_ln*ry_ln*ry_nl*factor_global; //x_yy
                        conductance_on_kmesh_on_omega[writing_ph_index+3]+=dvy_ln*ry_ln*ry_nl*factor_global; //y_yy
                        conductance_on_kmesh_on_omega[writing_ph_index+4]+=dvx_ln*rx_ln*ry_nl*factor_global; //x_xy
                        conductance_on_kmesh_on_omega[writing_ph_index+5]+=dvy_ln*rx_ln*ry_nl*factor_global; //y_xy
                        conductance_on_kmesh_on_omega[writing_ph_index+6]+=dvx_ln*ry_ln*rx_nl*factor_global; //x_yx
                        conductance_on_kmesh_on_omega[writing_ph_index+7]+=dvy_ln*ry_ln*rx_nl*factor_global; //y_yx
                    }
                    
                }
            }
        }
    }
    printf(" done\n");
    printf("clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fprintf(fp," done\n");
    fclose(fp);
    

    PyObject * conductance_real=PyTuple_New(nphoton*nkpt*8);
    PyObject * conductance_imag=PyTuple_New(nphoton*nkpt*8);

    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            PyTuple_SetItem(conductance_real, 0*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8  ])));
            PyTuple_SetItem(conductance_real, 1*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+1])));
            PyTuple_SetItem(conductance_real, 2*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+2])));
            PyTuple_SetItem(conductance_real, 3*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+3])));
            PyTuple_SetItem(conductance_real, 4*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+4])));
            PyTuple_SetItem(conductance_real, 5*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+5])));
            PyTuple_SetItem(conductance_real, 6*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+6])));
            PyTuple_SetItem(conductance_real, 7*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+7])));

            PyTuple_SetItem(conductance_imag, 0*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8  ])));
            PyTuple_SetItem(conductance_imag, 1*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+1])));
            PyTuple_SetItem(conductance_imag, 2*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+2])));
            PyTuple_SetItem(conductance_imag, 3*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+3])));
            PyTuple_SetItem(conductance_imag, 4*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+4])));
            PyTuple_SetItem(conductance_imag, 5*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+5])));
            PyTuple_SetItem(conductance_imag, 6*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+6])));
            PyTuple_SetItem(conductance_imag, 7*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ikpt*nphoton*8+iphoton*8+7])));
        }
    }
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

//********** Blocks for projected currents *****************
// 1. the velocities and projected velocities
/*
PyObject * calcFullBandsAndVecsAndSpinProjectedVelocityInMemory(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int overLat[nhoppings][3],
                       double lattice[3][3],
                       double orbCorr[norbs][3], // fractional coor. of each orb
                       int orbProj1Weight[norbs], int orbProj2Weight[norbs],  // weight of each orb, either 1 or 0, for projection calc.
                       int _nthreads){
    printf("entering spin-polarized projective velocity calc. in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d\n",nkpt,norbs,nhoppings);
    fprintf(fp,"memory estimation: %.4f MB\n",(8.0*norbs*norbs+norbs)/1024/1024*nkpt*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d\n",nkpt,norbs,nhoppings);
    printf("memory estimation: %.4f MB\n",(8.0*norbs*norbs+norbs)/1024/1024*nkpt*sizeof(lapack_complex_double));
    
    double *energy=(double*)malloc(nkpt*norbs*sizeof(double));
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(2*nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_proj_sx=(lapack_complex_double*)malloc(2*nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_proj_sy=(lapack_complex_double*)malloc(2*nkpt*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_proj_sz=(lapack_complex_double*)malloc(2*nkpt*norbs*norbs*sizeof(lapack_complex_double));

    double sx[norbs][norbs];
    lapack_complex_double sy[norbs][norbs];
    double sz[norbs][norbs];
    for(int i=0;i<norbs;i++){
        for(int j=0;j<norbs;j++){
            sx[i][j]=0;
            sy[i][j]=0;
            sz[i][j]=0;
        }
    }
    for(int i=0;i<norbs/2;i++){
        sx[i][i+norbs/2]=0.5;
        sx[i+norbs/2][i]=0.5;
        sy[i][i+norbs/2]=-0.5*_Complex_I;
        sy[i+norbs/2][i]=0.5*_Complex_I;
        sz[i][i]=0.5;
        sz[i+norbs/2][i+norbs/2]=-0.5;
    }

    int kpt_width=norbs*norbs;
    int dir_width=nkpt*kpt_width;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("use %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"calc spin-polarized velocity using %d cores in ham.dll\n",nthreads);
    fclose(fp);

    printf("calc spin-polarized projective velocities using %d cores in ham.dll\n",nthreads);
    //for(int i=0;i<norbs;i++)printf("%f ",orbWeight[i]);
    //exit(1);
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            lapack_complex_double hk[norbs][norbs]; // init Hk
            lapack_complex_double DhkDkxy[norbs][norbs][2]; // init DHk/Dkx,Dky
            lapack_complex_double DhkDkxy_proj[norbs][norbs][2]; // init projective DHk/Dkx,Dky
            lapack_complex_double DhkDkxy_proj_sx[norbs][norbs][2]; // init projective DHk/Dkx,Dky
            lapack_complex_double DhkDkxy_proj_sy[norbs][norbs][2]; // init projective DHk/Dkx,Dky
            lapack_complex_double DhkDkxy_proj_sz[norbs][norbs][2]; // init projective DHk/Dkx,Dky
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i][j]=0;
                    for(int k=0;k<2;k++){
                        DhkDkxy[i][j][k]=0;
                        DhkDkxy_proj[i][j][k]=0;
                        DhkDkxy_proj_sx[i][j][k]=0;
                        DhkDkxy_proj_sy[i][j][k]=0;
                        DhkDkxy_proj_sz[i][j][k]=0;
                    }
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){ // setting Hk, derivated Hk, and projectively derivated Hk
                lapack_complex_double hopping_R[3];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[targetID[ihopping]][i]-orbCorr[sourceID[ihopping]][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;

                lapack_complex_double hopping_R_c[2];
                for(int i=0;i<2;i++)hopping_R_c[i]=0;
                for(int i=0;i<2;i++)for(int j=0;j<3;j++)hopping_R_c[i]+=hopping_R[j]*lattice[j][i];

                for(int i=0;i<2;i++) DhkDkxy[sourceID[ihopping]][targetID[ihopping]][i]+=tk*hopping_R_c[i];
                for(int i=0;i<2;i++) {
                    // project to hopping path
                    if(orbProj1Weight[sourceID[ihopping]]*orbProj2Weight[targetID[ihopping]]+orbProj2Weight[sourceID[ihopping]]*orbProj1Weight[targetID[ihopping]]>0)DhkDkxy_proj[sourceID[ihopping]][targetID[ihopping]][i]+=tk*hopping_R_c[i];
                }
            }
            // project to spin
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    for(int k=0;k<2;k++){
                        for(int icontract=0;icontract<norbs;icontract++){
                            DhkDkxy_proj_sx[i][j][k]+=DhkDkxy_proj[i][icontract][k]*sx[icontract][j]+sx[i][icontract]*DhkDkxy_proj[icontract][j][k];
                            DhkDkxy_proj_sy[i][j][k]+=DhkDkxy_proj[i][icontract][k]*sy[icontract][j]+sy[i][icontract]*DhkDkxy_proj[icontract][j][k];
                            DhkDkxy_proj_sz[i][j][k]+=DhkDkxy_proj[i][icontract][k]*sz[icontract][j]+sz[i][icontract]*DhkDkxy_proj[icontract][j][k];
                        }
                    }
                }
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0){
                printf("LAPACKE failed, info: %d\n",info);
                exit(1);
            }
            for(int bandm=0;bandm<norbs;bandm++){
                energy[ikpt*norbs+bandm]=eigs[bandm];
                for(int bandn=0;bandn<norbs;bandn++){
                    for(int idir=0;idir<2;idir++){
                        int ipos=idir*dir_width+ikpt*kpt_width+bandm*norbs+bandn;
                        velocity[ipos]=0;
                        velocity_proj_sx[ipos]=0;
                        velocity_proj_sy[ipos]=0;
                        velocity_proj_sz[ipos]=0;
                        for(int icontract=0;icontract<norbs;icontract++){
                            for(int jcontract=0;jcontract<norbs;jcontract++){
                                velocity[ipos]+=conj(hk[icontract][bandm])*DhkDkxy[icontract][jcontract][idir]*hk[jcontract][bandn];
                                velocity_proj_sx[ipos]+=conj(hk[icontract][bandm])*DhkDkxy_proj_sx[icontract][jcontract][idir]*hk[jcontract][bandn];
                                velocity_proj_sy[ipos]+=conj(hk[icontract][bandm])*DhkDkxy_proj_sy[icontract][jcontract][idir]*hk[jcontract][bandn];
                                velocity_proj_sz[ipos]+=conj(hk[icontract][bandm])*DhkDkxy_proj_sz[icontract][jcontract][idir]*hk[jcontract][bandn];
                            }
                        }
                    }
                }
            }  
        }
    }
    fp=fopen("./c_log","a");
    fprintf(fp,"spin projective velocity completed\ncalc BPVE...\n");
    fclose(fp);
    
    long long eigs_addr=(long long)energy;
    long long vel_addr=(long long)velocity;
    long long vel_proj_sx_addr=(long long)velocity_proj_sx;
    long long vel_proj_sy_addr=(long long)velocity_proj_sy;
    long long vel_proj_sz_addr=(long long)velocity_proj_sz;
    printf("in the current system, size of long long: %d, size of pointer %d\n",sizeof(long long),sizeof(lapack_complex_double *));
    printf("pls. see if they are identical. Otherwise the calc. results with cdll are seriously wrong!\n");
    printf("in ham.dll, eig.:%p, vel.:%p, proj. vel.:%p %p %p\n",energy,velocity,velocity_proj_sx,velocity_proj_sy,velocity_proj_sz);
    return Py_BuildValue("(LLLLL)",eigs_addr,vel_addr,vel_proj_sx_addr,vel_proj_sy_addr,vel_proj_sz_addr);
}

PyObject * calcSpinProjectiveBulkPhotovoltaicCond(int nkpt,
                       long long eigs_addr, long long vel_addr, long long vel_proj_sx_addr, long long vel_proj_sy_addr, long long vel_proj_sz_addr,
                       int norbs,
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       int _nthreads){
    printf("entering spinful projective BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.4f MB\n",((double)nphoton*nkpt/1024/1024*32+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",((double)nphoton*nkpt/1024/1024*32+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)*sizeof(lapack_complex_double));
    
    double *energy=(double*)eigs_addr;
    lapack_complex_double *velocity=(lapack_complex_double*)vel_addr;
    lapack_complex_double *velocity_proj_sx=(lapack_complex_double*)vel_proj_sx_addr;
    lapack_complex_double *velocity_proj_sy=(lapack_complex_double*)vel_proj_sy_addr;
    lapack_complex_double *velocity_proj_sz=(lapack_complex_double*)vel_proj_sz_addr;
    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(nphoton*nkpt*24*sizeof(lapack_complex_double));
    for(int i=0;i<nphoton*nkpt*24;i++)conductance_on_kmesh_on_omega[i]=0;

    int kpt_width=norbs*norbs;
    int dir_width=nkpt*kpt_width;
    int spin_width=nkpt*nphoton*8;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("calc spin projective BPVE using %d cores in ham.dll\n",nthreads);
    fp=fopen("./c_log","a");
    fprintf(fp,"calc spin projective BPVE using %d cores in ham.dll\n",nthreads);
    fclose(fp);

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int writing_k_index=ikpt*nphoton*8;
            for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                double El=energy[ikpt*norbs+bandl];
                if(El<erange[0] | El>erange[1]) continue;
                int fl=0;
                if(El<=fermi)fl=1;
                for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                    double En=energy[ikpt*norbs+bandn];
                    if(En<erange[0] | En>erange[1]) continue;
                    int fn=0;
                    if(En<=fermi)fn=1;
                    int fln=fl-fn;
                    if (fln==0) continue;

                    // calc overlap
                    lapack_complex_double px_nl=velocity[          ikpt*kpt_width+bandn*norbs+bandl];
                    lapack_complex_double py_nl=velocity[dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                    //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                    for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                        //if(bandn!=bandl)continue; // switch for bandn
                        double Em=energy[ikpt*norbs+bandm];
                        if(Em<erange[0] | Em>erange[1]) continue;
                        // calc overlap
                        lapack_complex_double px_lm=velocity[          ikpt*kpt_width+bandl*norbs+bandm];
                        lapack_complex_double py_lm=velocity[dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                        lapack_complex_double px_sx_mn=velocity_proj_sx[          ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double py_sx_mn=velocity_proj_sx[dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double px_sy_mn=velocity_proj_sy[          ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double py_sy_mn=velocity_proj_sy[dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double px_sz_mn=velocity_proj_sz[          ikpt*kpt_width+bandm*norbs+bandn];
                        lapack_complex_double py_sz_mn=velocity_proj_sz[dir_width+ikpt*kpt_width+bandm*norbs+bandn];

                        for(int iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                            int writing_ph_index=writing_k_index+iphoton*8;
                            double omega=omegaList[iphoton];
                            
                            lapack_complex_double factor_global=0;
                            factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                            
                            conductance_on_kmesh_on_omega[             writing_ph_index  ]+=px_nl*px_lm*px_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+1]+=px_nl*px_lm*py_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+2]+=py_nl*py_lm*px_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+3]+=py_nl*py_lm*py_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+4]+=px_nl*py_lm*px_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+5]+=px_nl*py_lm*py_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+6]+=py_nl*px_lm*px_sx_mn*factor_global;
                            conductance_on_kmesh_on_omega[             writing_ph_index+7]+=py_nl*px_lm*py_sx_mn*factor_global;

                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index  ]+=px_nl*px_lm*px_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+1]+=px_nl*px_lm*py_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+2]+=py_nl*py_lm*px_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+3]+=py_nl*py_lm*py_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+4]+=px_nl*py_lm*px_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+5]+=px_nl*py_lm*py_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+6]+=py_nl*px_lm*px_sy_mn*factor_global;
                            conductance_on_kmesh_on_omega[  spin_width+writing_ph_index+7]+=py_nl*px_lm*py_sy_mn*factor_global;

                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index  ]+=px_nl*px_lm*px_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+1]+=px_nl*px_lm*py_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+2]+=py_nl*py_lm*px_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+3]+=py_nl*py_lm*py_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+4]+=px_nl*py_lm*px_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+5]+=px_nl*py_lm*py_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+6]+=py_nl*px_lm*px_sz_mn*factor_global;
                            conductance_on_kmesh_on_omega[2*spin_width+writing_ph_index+7]+=py_nl*px_lm*py_sz_mn*factor_global;
                        }
                    } 
                }
            }
        }
    }
    printf("spin projective BPVE done\n");
    printf("clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"spin projective BPVE done\n");
    fprintf(fp,"clib tips: pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);
    free(velocity);free(velocity_proj_sx);free(velocity_proj_sy);free(velocity_proj_sz);free(energy);

    PyObject * conductance_real=PyTuple_New(nphoton*nkpt*24);
    PyObject * conductance_imag=PyTuple_New(nphoton*nkpt*24);

    for(int iphoton=0;iphoton<nphoton;iphoton++){
        for(int ikpt=0;ikpt<nkpt;ikpt++){
            for(int icomponent=0;icomponent<8;icomponent++){
                for(int ispin=0;ispin<3;ispin++){
                    PyTuple_SetItem(conductance_real, ispin*8*nphoton*nkpt+icomponent*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[ispin*nkpt*nphoton*8+ikpt*nphoton*8+iphoton*8+icomponent])));
                    PyTuple_SetItem(conductance_imag, ispin*8*nphoton*nkpt+icomponent*nphoton*nkpt+iphoton*nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[ispin*nkpt*nphoton*8+ikpt*nphoton*8+iphoton*8+icomponent])));
                }
            }
        }
    }
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}
*/

// New version of Proj. BPVE
int _calcBandsAndProjectedVelocities(
                       int nkpt, double kmesh[], // num. of kpts and frac. coor. of kpts
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int overLat[nhoppings][3],
                       double lattice[3][3],
                       double orbCorr[norbs][3], // fractional coor. of each orb
                       double erange[2], // energy range for calc.
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       double * energy, // output energies
                       lapack_complex_double * velocities, // output projected velocities
                       int _nthreads){
    printf("projective velocity calc. in cdll\n");

    int kpt_width_g=norbs*norbs;
    int dir_width_g=nkpt*kpt_width_g;
    int proj_width_g=2*dir_width_g;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("calc projective velocity using %d cores in ham.dll\n",nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkf=(lapack_complex_double*)malloc(nProj*norbs*norbs*3*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        int ipos;
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            //lapack_complex_double hk[norbs][norbs]; // init Hk
            //lapack_complex_double DhkDkf[nProj][norbs][norbs][2]; // init DHk/Dk1,Dk2 for each projection
            //lapack_complex_double DhkDkc[nProj][norbs][norbs][2]; // init DHk/Dkx,Dky for each projection
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i*norbs+j]=0;
                    for(int k=0;k<nProj;k++){
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+0]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+1]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+2]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                    }
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            int isource,itarget;
            for(int ihopping=0;ihopping<nhoppings;ihopping++){ // setting Hk, derivated Hk, and projectively derivated Hk
                lapack_complex_double hopping_R[3];
                isource=sourceID[ihopping];
                itarget=targetID[ihopping];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[itarget][i]-orbCorr[isource][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt*3+i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[isource*norbs+itarget]+=tk;

                for(int iproj=0;iproj<nProj;iproj++){// setting projected DhkDkf
                    if(wOrbSourceProj[iproj][isource]*wOrbTargetProj[iproj][itarget]+wOrbSourceProj[iproj][itarget]*wOrbTargetProj[iproj][isource]>0){
                        for(int i=0;i<3;i++)DhkDkf[iproj*norbs*norbs*3+isource*norbs*3+itarget*3+i]+=tk*hopping_R[i];
                    }
                }
            }
            // unitary transform into real space partial
            for(int iproj=0;iproj<nProj;iproj++){
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<3;k++){
                            //DhkDkc[iproj][i][j][0]+=DhkDkf[iproj][i][j][k]*lattice[k][0];
                            //DhkDkc[iproj][i][j][1]+=DhkDkf[iproj][i][j][k]*lattice[k][1];
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][0];
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][1];
                        }
                    }
                }
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0){
                printf("LAPACKE failed, info: %d\n",info);
                exit(1);
            }
            double Em, En;
            for(int bandm=0;bandm<norbs;bandm++){
                Em=eigs[bandm];
                energy[ikpt*norbs+bandm]=Em;
                if(Em>erange[1] | Em<erange[0])continue;
                for(int bandn=0;bandn<norbs;bandn++){
                    En=eigs[bandn];
                    if(En>erange[1] | En<erange[0])continue;
                    for(int iproj=0;iproj<nProj;iproj++){
                        for(int idir=0;idir<2;idir++){
                            ipos=iproj*proj_width_g+idir*dir_width_g+ikpt*kpt_width_g+bandm*norbs+bandn;
                            velocities[ipos]=0;
                            for(int icontract=0;icontract<norbs;icontract++){
                                for(int jcontract=0;jcontract<norbs;jcontract++){
                                    velocities[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                }
                            }
                        }
                    }
                }
            }  
        }
        free(hk);free(DhkDkf);free(DhkDkc);
    }
    //fp=fopen("./c_log","a");
    //fprintf(fp,"projective velocity completed\n");
    //fclose(fp);
    return 0;
}

PyObject * calcProjectiveBulkPhotovoltaicCond_memoryReduce(
                       int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       // projection part
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       int nScatterRoute, // num. of scattering routes
                       int scatterProjOrder[nScatterRoute][3], // proj. vel. order in each scattering route
                       // parallel part
                       int ndivide, // num. of parts
                       int _nthreads){
    printf("entering Projective BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.3f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    //exit(1);
    
    int nkpt_part=nkpt/ndivide;
    unsigned long long kpt_width=norbs*norbs;
    unsigned long long dir_width=nkpt_part*kpt_width;
    unsigned long long proj_width=2*dir_width;

    unsigned long long _nphoton=nphoton;
    unsigned long long _nkpt=nkpt;
    unsigned long long sca_width=_nkpt*_nphoton*8;

    unsigned long long _nScatterRoute=nScatterRoute;

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    for(unsigned long long i=0;i<_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8;i++)conductance_on_kmesh_on_omega[i]=0;
    
    int ikpt_head;
    if(nkpt%ndivide!=0){
        printf("Error: nkpt %d cannot be divide by %d, exiting now!\n",nkpt,ndivide);
        exit(0);
    }
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    double *energy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *sub_kmesh=(double*)malloc(nkpt_part*3*sizeof(double));
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt_part<_nthreads*10)nthreads=1;

    for(int ipart=0;ipart<ndivide;ipart++){// kmesh partition
        ikpt_head=nkpt_part*ipart;
        for(int i=0;i<nkpt_part;i++)for(int j=0;j<3;j++)sub_kmesh[i*3+j]=kmesh[i+ikpt_head][j];

        printf("use %d cores in ham.dll\n",nthreads);
        fp=fopen("./c_log","a");
        fprintf(fp,"calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        fclose(fp);
        printf("calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        
        int status=_calcBandsAndProjectedVelocities(
                       nkpt_part, sub_kmesh, // num. of kpts and frac. coor. of kpts
                       norbs, nhoppings,
                       sourceID, targetID, 
                       hopping_real,
                       hopping_imag,
                       overLat,
                       lattice,
                       orbCorr, // fractional coor. of each orb
                       erange, // energy range for calc.
                       nProj, // num. of projections
                       wOrbSourceProj, // index of source orb in each projection
                       wOrbTargetProj, // index of target orb in each projection
                       energy, // output energies
                       velocity, // output projected velocities
                       nthreads);
        if(status!=0){
            printf("Error: velocity calc. failed\n");
            exit(1);
        }
        fp=fopen("./c_log","a");
        fprintf(fp,"velocities part%d completed\ncalc BPVE...\n",ipart+1);
        fclose(fp);
        printf("velocities part%d completed\nnow calc BPVE use %d cores in ham.dll\n",ipart+1,nthreads);

        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                unsigned long long ikpt_true=ikpt_head+ikpt;
                unsigned long long writing_k_index=ikpt_true*nphoton*8;
                // printf("ikpt_true: %d\n",ikpt_true);
                for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                    double El=energy[ikpt*norbs+bandl];
                    if(El<erange[0] | El>erange[1]) continue;
                    int fl=0;
                    if(El<=fermi)fl=1;
                    for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                        double En=energy[ikpt*norbs+bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        int fn=0;
                        if(En<=fermi)fn=1;
                        int fln=fl-fn;
                        if (fln==0) continue;

                        unsigned long long iv0,iv1,iv2;
                        for(int iscatter=0;iscatter<nScatterRoute;iscatter++){
                            iv0=scatterProjOrder[iscatter][0];
                            iv1=scatterProjOrder[iscatter][1];
                            iv2=scatterProjOrder[iscatter][2];
                            //printf("iv0 %d iv1 %d iv2 %d\n",iv0,iv1,iv2);
                            unsigned long long writing_scatter_index=iscatter*sca_width;
                            //if(writing_scatter_index<0){
                            //    printf("wri sca %d, isc %d, sca_width %d\n",writing_scatter_index,iscatter,sca_width);
                            //    exit(1);
                            //}
                            // calc overlap
                            lapack_complex_double px_nl=velocity[iv0*proj_width+          ikpt*kpt_width+bandn*norbs+bandl];
                            lapack_complex_double py_nl=velocity[iv0*proj_width+dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                            //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                            for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                                //if(bandn!=bandl)continue; // switch for bandn
                                double Em=energy[ikpt*norbs+bandm];
                                if(Em<erange[0] | Em>erange[1]) continue;
                                // calc overlap
                                lapack_complex_double px_lm=velocity[iv1*proj_width+          ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double py_lm=velocity[iv1*proj_width+dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double px_mn=velocity[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn=velocity[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                //printf("band%d-%d, px_lm %f %f, py_lm %f %f band%d-%d px_mn %f %f py_mn %f %f\n",bandl,bandm,creal(px_lm),cimag(px_lm),creal(py_lm),cimag(py_lm),bandm,bandn,creal(px_mn),cimag(px_mn),creal(py_mn),cimag(py_mn));
                                for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                                    unsigned long long writing_ph_index=writing_scatter_index+writing_k_index+iphoton*8;
                                    double omega=omegaList[iphoton];
                                    lapack_complex_double factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                                    //printf("iphoton %d start, pos=%d (wri sca %d, wri k %d), omega=%f factor: %f %f\n",iphoton,writing_ph_index,writing_scatter_index,writing_k_index,omega,creal(factor_global),cimag(factor_global));
                                    conductance_on_kmesh_on_omega[writing_ph_index  ]+=px_nl*px_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+1]+=px_nl*px_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+2]+=py_nl*py_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+3]+=py_nl*py_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+4]+=px_nl*py_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+5]+=px_nl*py_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+6]+=py_nl*px_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+7]+=py_nl*px_lm*py_mn*factor_global;
                                    //printf("iphoton %d end\n",iphoton);
                                }
                            } 
                        }
                    }
                }
            }
        }
        printf("BPVE part%d done\n",ipart+1);
        fp=fopen("./c_log","a");
        fprintf(fp,"BPVE part%d done\n",ipart+1);
        fclose(fp);
    }
    free(velocity);free(energy);
    printf("All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);
    
    
    /*
    PyObject * conductance_real=PyTuple_New((unsigned long long)nScatterRoute*_nphoton*8*_nkpt);
    PyObject * conductance_imag=PyTuple_New((unsigned long long)nScatterRoute*_nphoton*8*_nkpt);
    printf("memory allocated\n");
     
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                for(unsigned long long icomp=0;icomp<8;icomp++){
                    PyTuple_SetItem(conductance_real, iscatter*sca_width+icomp*_nphoton*_nkpt+iphoton*_nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp])));
                    PyTuple_SetItem(conductance_imag, iscatter*sca_width+icomp*_nphoton*_nkpt+iphoton*_nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp])));
                }
            }
        }
    }
    */
    PyObject * conductance_real=PyTuple_New(_nScatterRoute*_nphoton*8);
    PyObject * conductance_imag=PyTuple_New(_nScatterRoute*_nphoton*8);
    printf("memory allocated\n");
    lapack_complex_double bzIntegration;
    unsigned long long pos;
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long icomp=0;icomp<8;icomp++){
                bzIntegration=0;
                for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                    pos=iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp;
                    bzIntegration+=conductance_on_kmesh_on_omega[pos];
                }
                PyTuple_SetItem(conductance_real, iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration/nkpt)));
                PyTuple_SetItem(conductance_imag, iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration/nkpt)));
            }
        }
    }

    printf("py data setted\n");
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

PyObject * calcModernOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce(
                       int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       // projection part
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       int nScatterRoute, // num. of scattering routes
                       int scatterProjOrder[nScatterRoute][3], // proj. vel. order in each scattering route
                       // parallel part
                       int ndivide, // num. of parts
                       int _nthreads){
    printf("entering Projective modern orbital BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.3f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(2*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    //exit(1);
    
    int nkpt_part=nkpt/ndivide;
    unsigned long long kpt_width=norbs*norbs;
    unsigned long long dir_width=nkpt_part*kpt_width;
    unsigned long long proj_width=2*dir_width;

    unsigned long long _nphoton=nphoton;
    unsigned long long _nkpt=nkpt;
    unsigned long long sca_width=_nkpt*_nphoton*8;

    unsigned long long _nScatterRoute=nScatterRoute;

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega_LC=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    lapack_complex_double * conductance_on_kmesh_on_omega_IC=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    for(unsigned long long i=0;i<_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8;i++){
        conductance_on_kmesh_on_omega_LC[i]=0;
        conductance_on_kmesh_on_omega_IC[i]=0;
    }
    
    int ikpt_head;
    if(nkpt%ndivide!=0){
        printf("Error: nkpt %d cannot be divide by %d, exiting now!\n",nkpt,ndivide);
        exit(0);
    }
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    
    double *energy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *sub_kmesh=(double*)malloc(nkpt_part*3*sizeof(double));
    // for orbital angular moments
    double *G_xy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *H_xy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *F_xy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *IC_xy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *LC_xy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt_part<_nthreads*10)nthreads=1;

    for(int ipart=0;ipart<ndivide;ipart++){// kmesh partition
        ikpt_head=nkpt_part*ipart;
        for(int i=0;i<nkpt_part;i++)for(int j=0;j<3;j++)sub_kmesh[i*3+j]=kmesh[i+ikpt_head][j];

        printf("use %d cores in ham.dll\n",nthreads);
        fp=fopen("./c_log","a");
        fprintf(fp,"calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        fclose(fp);
        printf("calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        
        int status=_calcBandsAndProjectedVelocities(
                       nkpt_part, sub_kmesh, // num. of kpts and frac. coor. of kpts
                       norbs, nhoppings,
                       sourceID, targetID, 
                       hopping_real,
                       hopping_imag,
                       overLat,
                       lattice,
                       orbCorr, // fractional coor. of each orb
                       erange, // energy range for calc.
                       nProj, // num. of projections
                       wOrbSourceProj, // index of source orb in each projection
                       wOrbTargetProj, // index of target orb in each projection
                       energy, // output energies
                       velocity, // output projected velocities
                       nthreads);
        if(status!=0){
            printf("Error: velocity calc. failed\n");
            exit(1);
        }
        fp=fopen("./c_log","a");
        fprintf(fp,"velocities part%d completed\ncalc orbital angular moments...\n",ipart+1);
        fclose(fp);
        printf("velocities part%d completed\nnow calc orbital angular moments use %d cores in ham.dll\n",ipart+1,nthreads);

        // calc. orbital angular moments
        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){
                for(int bandl=0;bandl<norbs;bandl++){
                    double El=energy[ikpt*norbs+bandl];
                    //G_xy[ikpt*norbs+bandl]=0;
                    //H_xy[ikpt*norbs+bandl]=0;
                    //F_xy[ikpt*norbs+bandl]=0;
                    double _g_xy=0;
                    double _h_xy=0;
                    double _f_xy=0;
                    double Em,dE2;
                    lapack_complex_double px_lm,py_ml;
                    for(int bandm=0;bandm<norbs;bandm++){
                        Em=energy[ikpt*norbs+bandm];
                        dE2=(Em-El)*(Em-El);
                        if(fabs(dE2)<1e-8) continue;
                        px_lm=velocity[          ikpt*kpt_width+bandl*norbs+bandm];
                        py_ml=velocity[dir_width+ikpt*kpt_width+bandm*norbs+bandl];
                        _g_xy+=-2*Em/dE2*cimag(px_lm*py_ml);
                        _h_xy+=-2*El/dE2*cimag(px_lm*py_ml);
                        _f_xy+=-2/dE2*cimag(px_lm*py_ml);
                    }
                    G_xy[ikpt*norbs+bandl]=_g_xy;
                    H_xy[ikpt*norbs+bandl]=_h_xy;
                    F_xy[ikpt*norbs+bandl]=_f_xy;
                    LC_xy[ikpt*norbs+bandl]=2*(_g_xy-fermi*_f_xy);
                    IC_xy[ikpt*norbs+bandl]=2*(_h_xy-fermi*_f_xy);
                }
            }
        }
        
        fp=fopen("./c_log","a");
        fprintf(fp,"Orbital part%d completed\ncalc BPVE...\n",ipart+1);
        fclose(fp);
        printf("Orbital part%d completed\nnow calc BPVE use %d cores in ham.dll\n",ipart+1,nthreads);
        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                unsigned long long ikpt_true=ikpt_head+ikpt;
                unsigned long long writing_k_index=ikpt_true*nphoton*8;
                // printf("ikpt_true: %d\n",ikpt_true);
                for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                    double El=energy[ikpt*norbs+bandl];
                    if(El<erange[0] | El>erange[1]) continue;
                    int fl=0;
                    if(El<=fermi)fl=1;
                    for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                        double En=energy[ikpt*norbs+bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        int fn=0;
                        if(En<=fermi)fn=1;
                        int fln=fl-fn;
                        if (fln==0) continue;

                        unsigned long long iv0,iv1,iv2;
                        for(int iscatter=0;iscatter<nScatterRoute;iscatter++){
                            iv0=scatterProjOrder[iscatter][0];
                            iv1=scatterProjOrder[iscatter][1];
                            iv2=scatterProjOrder[iscatter][2];
                            //printf("iv0 %d iv1 %d iv2 %d\n",iv0,iv1,iv2);
                            unsigned long long writing_scatter_index=iscatter*sca_width;
                            //if(writing_scatter_index<0){
                            //    printf("wri sca %d, isc %d, sca_width %d\n",writing_scatter_index,iscatter,sca_width);
                            //    exit(1);
                            //}
                            // calc overlap
                            lapack_complex_double px_nl=velocity[iv0*proj_width+          ikpt*kpt_width+bandn*norbs+bandl];
                            lapack_complex_double py_nl=velocity[iv0*proj_width+dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                            //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                            for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                                //if(bandn!=bandl)continue; // switch for bandn
                                double Em=energy[ikpt*norbs+bandm];
                                if(Em<erange[0] | Em>erange[1]) continue;
                                // calc overlap
                                lapack_complex_double px_lm=velocity[iv1*proj_width+          ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double py_lm=velocity[iv1*proj_width+dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double px_mn=velocity[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn=velocity[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                //printf("band%d-%d, px_lm %f %f, py_lm %f %f band%d-%d px_mn %f %f py_mn %f %f\n",bandl,bandm,creal(px_lm),cimag(px_lm),creal(py_lm),cimag(py_lm),bandm,bandn,creal(px_mn),cimag(px_mn),creal(py_mn),cimag(py_mn));
                                // calc orbital angular moments
                                double _lc_avg=(LC_xy[ikpt*norbs+bandm]+LC_xy[ikpt*norbs+bandn])/2;
                                double _ic_avg=(IC_xy[ikpt*norbs+bandm]+IC_xy[ikpt*norbs+bandn])/2;
                                for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                                    unsigned long long writing_ph_index=writing_scatter_index+writing_k_index+iphoton*8;
                                    double omega=omegaList[iphoton];
                                    lapack_complex_double factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                                    lapack_complex_double factor_lc=factor_global*_lc_avg;
                                    //printf("iphoton %d start, pos=%d (wri sca %d, wri k %d), omega=%f factor: %f %f\n",iphoton,writing_ph_index,writing_scatter_index,writing_k_index,omega,creal(factor_global),cimag(factor_global));
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index  ]+=px_nl*px_lm*px_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+1]+=px_nl*px_lm*py_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+2]+=py_nl*py_lm*px_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+3]+=py_nl*py_lm*py_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+4]+=px_nl*py_lm*px_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+5]+=px_nl*py_lm*py_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+6]+=py_nl*px_lm*px_mn*factor_lc;
                                    conductance_on_kmesh_on_omega_LC[writing_ph_index+7]+=py_nl*px_lm*py_mn*factor_lc;
                                    //printf("iphoton %d end\n",iphoton);
                                    lapack_complex_double factor_ic=factor_global*_ic_avg;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index  ]+=px_nl*px_lm*px_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+1]+=px_nl*px_lm*py_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+2]+=py_nl*py_lm*px_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+3]+=py_nl*py_lm*py_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+4]+=px_nl*py_lm*px_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+5]+=px_nl*py_lm*py_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+6]+=py_nl*px_lm*px_mn*factor_ic;
                                    conductance_on_kmesh_on_omega_IC[writing_ph_index+7]+=py_nl*px_lm*py_mn*factor_ic;
                                }
                            } 
                        }
                    }
                }
            }
        }
        printf("BPVE part%d done\n",ipart+1);
        fp=fopen("./c_log","a");
        fprintf(fp,"BPVE part%d done\n",ipart+1);
        fclose(fp);
    }
    free(velocity);free(energy);
    printf("All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);
    
    
    /*
    PyObject * conductance_real=PyTuple_New((unsigned long long)nScatterRoute*_nphoton*8*_nkpt);
    PyObject * conductance_imag=PyTuple_New((unsigned long long)nScatterRoute*_nphoton*8*_nkpt);
    printf("memory allocated\n");
     
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                for(unsigned long long icomp=0;icomp<8;icomp++){
                    PyTuple_SetItem(conductance_real, iscatter*sca_width+icomp*_nphoton*_nkpt+iphoton*_nkpt+ikpt, PyFloat_FromDouble(creal(conductance_on_kmesh_on_omega[iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp])));
                    PyTuple_SetItem(conductance_imag, iscatter*sca_width+icomp*_nphoton*_nkpt+iphoton*_nkpt+ikpt, PyFloat_FromDouble(cimag(conductance_on_kmesh_on_omega[iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp])));
                }
            }
        }
    }
    */
    PyObject * conductance_real=PyTuple_New(2*_nScatterRoute*_nphoton*8);
    PyObject * conductance_imag=PyTuple_New(2*_nScatterRoute*_nphoton*8);
    printf("memory allocated\n");
    lapack_complex_double bzIntegration_ic,bzIntegration_lc;
    unsigned long long pos;
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long icomp=0;icomp<8;icomp++){
                bzIntegration_ic=0;bzIntegration_lc=0;
                for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                    pos=iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp;
                    bzIntegration_lc+=conductance_on_kmesh_on_omega_LC[pos];
                    bzIntegration_ic+=conductance_on_kmesh_on_omega_IC[pos];
                }
                PyTuple_SetItem(conductance_real, 0*_nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration_lc/nkpt)));
                PyTuple_SetItem(conductance_imag, 0*_nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration_lc/nkpt)));
                PyTuple_SetItem(conductance_real, 1*_nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration_ic/nkpt)));
                PyTuple_SetItem(conductance_imag, 1*_nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration_ic/nkpt)));
            }
        }
    }

    printf("py data setted\n");
    free(conductance_on_kmesh_on_omega_LC);free(conductance_on_kmesh_on_omega_IC);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

int _calcSpinBandsAndProjectedVelocities(
                       int nkpt, double kmesh[], // num. of kpts and frac. coor. of kpts
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int overLat[nhoppings][3],
                       double lattice[3][3],
                       double orbCorr[norbs][3], // fractional coor. of each orb
                       double erange[2], // energy range for calc.
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       double sx[norbs][norbs], lapack_complex_double sy[norbs][norbs], double sz[norbs][norbs], // spin operators
                       double * energy, // output energies
                       lapack_complex_double * velocities, // output projected velocities
                       lapack_complex_double * velocities_sx, lapack_complex_double * velocities_sy, lapack_complex_double * velocities_sz, // output projected spin velocities
                       int _nthreads){
    printf("projective spinful velocity calc. in cdll\n");

    int kpt_width_g=norbs*norbs;
    int dir_width_g=nkpt*kpt_width_g;
    int proj_width_g=2*dir_width_g;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("calc projective spinful velocity using %d cores in ham.dll\n",nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkf=(lapack_complex_double*)malloc(nProj*norbs*norbs*3*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_sx=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_sy=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_sz=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        int ipos;
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i*norbs+j]=0;
                    for(int k=0;k<nProj;k++){
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+0]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+1]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+2]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                        DhkDkc_sx[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc_sx[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                        DhkDkc_sy[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc_sy[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                        DhkDkc_sz[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc_sz[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                    }
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            int isource,itarget;
            for(int ihopping=0;ihopping<nhoppings;ihopping++){ // setting Hk, derivated Hk, and projectively derivated Hk
                lapack_complex_double hopping_R[3];
                isource=sourceID[ihopping];
                itarget=targetID[ihopping];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[itarget][i]-orbCorr[isource][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt*3+i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[isource*norbs+itarget]+=tk;

                for(int iproj=0;iproj<nProj;iproj++){// setting projected DhkDkf
                    if(wOrbSourceProj[iproj][isource]*wOrbTargetProj[iproj][itarget]+wOrbSourceProj[iproj][itarget]*wOrbTargetProj[iproj][isource]>0){
                        for(int i=0;i<3;i++)DhkDkf[iproj*norbs*norbs*3+isource*norbs*3+itarget*3+i]+=tk*hopping_R[i];
                    }
                }
            }
            // unitary transform into real space partial
            for(int iproj=0;iproj<nProj;iproj++){
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<3;k++){
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][0];
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][1];
                        }
                    }
                }
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int icontraction=0;icontraction<norbs;icontraction++){
                            DhkDkc_sx[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+0]*sx[icontraction][j]+sx[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+0];
                            DhkDkc_sx[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+1]*sx[icontraction][j]+sx[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+1];
                            DhkDkc_sy[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+0]*sy[icontraction][j]+sy[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+0];
                            DhkDkc_sy[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+1]*sy[icontraction][j]+sy[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+1];
                            DhkDkc_sz[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+0]*sz[icontraction][j]+sz[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+0];
                            DhkDkc_sz[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkc[iproj*norbs*norbs*2+i*norbs*2+icontraction*2+1]*sz[icontraction][j]+sz[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+j*2+1];
                        }
                    }
                }
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0){
                printf("LAPACKE failed, info: %d\n",info);
                exit(1);
            }
            double Em, En;
            for(int bandm=0;bandm<norbs;bandm++){
                Em=eigs[bandm];
                energy[ikpt*norbs+bandm]=Em;
                if(Em>erange[1] | Em<erange[0])continue;
                for(int bandn=0;bandn<norbs;bandn++){
                    En=eigs[bandn];
                    if(En>erange[1] | En<erange[0])continue;
                    for(int iproj=0;iproj<nProj;iproj++){
                        for(int idir=0;idir<2;idir++){
                            ipos=iproj*proj_width_g+idir*dir_width_g+ikpt*kpt_width_g+bandm*norbs+bandn;
                            velocities[ipos]=0;velocities_sx[ipos]=0;velocities_sy[ipos]=0;velocities_sz[ipos]=0;
                            for(int icontract=0;icontract<norbs;icontract++){
                                for(int jcontract=0;jcontract<norbs;jcontract++){
                                    velocities[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                    velocities_sx[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc_sx[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                    velocities_sy[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc_sy[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                    velocities_sz[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc_sz[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                }
                            }
                        }
                    }
                }
            }  
        }
        free(hk);free(DhkDkf);free(DhkDkc);free(DhkDkc_sx);free(DhkDkc_sy);free(DhkDkc_sz);
    }
    //fp=fopen("./c_log","a");
    //fprintf(fp,"projective velocity completed\n");
    //fclose(fp);
    return 0;
}

PyObject * calcSpinProjectiveBulkPhotovoltaicCond_memoryReduce(
                       int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       // projection part
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       int nScatterRoute, // num. of scattering routes
                       int scatterProjOrder[nScatterRoute][3], // proj. vel. order in each scattering route
                       // parallel part
                       int ndivide, // num. of parts
                       int _nthreads){
    printf("entering spinful projective BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.3f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(8*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(8*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    //exit(1);
    
    double sx[norbs][norbs];
    lapack_complex_double sy[norbs][norbs];
    double sz[norbs][norbs];
    for(int i=0;i<norbs;i++){
        for(int j=0;j<norbs;j++){
            sx[i][j]=0;
            sy[i][j]=0;
            sz[i][j]=0;
        }
    }
    for(int i=0;i<norbs/2;i++){
        sx[i][i+norbs/2]=0.5;
        sx[i+norbs/2][i]=0.5;
        sy[i][i+norbs/2]=-0.5*_Complex_I;
        sy[i+norbs/2][i]=0.5*_Complex_I;
        sz[i][i]=0.5;
        sz[i+norbs/2][i+norbs/2]=-0.5;
    }

    int nkpt_part=nkpt/ndivide;
    unsigned long long kpt_width=norbs*norbs;
    unsigned long long dir_width=nkpt_part*kpt_width;
    unsigned long long proj_width=2*dir_width;

    unsigned long long _nphoton=nphoton;
    unsigned long long _nkpt=nkpt;
    unsigned long long sca_width=_nkpt*_nphoton*8;

    unsigned long long _nScatterRoute=nScatterRoute;

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega_sx=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    lapack_complex_double * conductance_on_kmesh_on_omega_sy=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    lapack_complex_double * conductance_on_kmesh_on_omega_sz=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    for(unsigned long long i=0;i<_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8;i++){
        conductance_on_kmesh_on_omega_sx[i]=0;
        conductance_on_kmesh_on_omega_sy[i]=0;
        conductance_on_kmesh_on_omega_sz[i]=0;
    }
    
    int ikpt_head;
    if(nkpt%ndivide!=0){
        printf("Error: nkpt %d cannot be divide by %d, exiting now!\n",nkpt,ndivide);
        exit(0);
    }
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_sx=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_sy=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_sz=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    double *energy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *sub_kmesh=(double*)malloc(nkpt_part*3*sizeof(double));
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt_part<_nthreads*10)nthreads=1;

    for(int ipart=0;ipart<ndivide;ipart++){// kmesh partition
        ikpt_head=nkpt_part*ipart;
        for(int i=0;i<nkpt_part;i++)for(int j=0;j<3;j++)sub_kmesh[i*3+j]=kmesh[i+ikpt_head][j];

        printf("use %d cores in ham.dll\n",nthreads);
        fp=fopen("./c_log","a");
        fprintf(fp,"calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        fclose(fp);
        printf("calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        
        int status=_calcSpinBandsAndProjectedVelocities(
                       nkpt_part, sub_kmesh, // num. of kpts and frac. coor. of kpts
                       norbs, nhoppings,
                       sourceID, targetID, 
                       hopping_real,
                       hopping_imag,
                       overLat,
                       lattice,
                       orbCorr, // fractional coor. of each orb
                       erange, // energy range for calc.
                       nProj, // num. of projections
                       wOrbSourceProj, // index of source orb in each projection
                       wOrbTargetProj, // index of target orb in each projection
                       sx,sy,sz,// spin operators
                       energy, // output energies
                       velocity, // output projected velocities
                       velocity_sx, velocity_sy, velocity_sz, // output projected spin velocities
                       nthreads);
        if(status!=0){
            printf("Error: velocity calc. failed\n");
            exit(1);
        }
        fp=fopen("./c_log","a");
        fprintf(fp,"velocities part%d completed\ncalc BPVE...\n",ipart+1);
        fclose(fp);
        printf("velocities part%d completed\nnow calc BPVE use %d cores in ham.dll\n",ipart+1,nthreads);

        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                unsigned long long ikpt_true=ikpt_head+ikpt;
                unsigned long long writing_k_index=ikpt_true*nphoton*8;
                // printf("ikpt_true: %d\n",ikpt_true);
                for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                    double El=energy[ikpt*norbs+bandl];
                    if(El<erange[0] | El>erange[1]) continue;
                    int fl=0;
                    if(El<=fermi)fl=1;
                    for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                        double En=energy[ikpt*norbs+bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        int fn=0;
                        if(En<=fermi)fn=1;
                        int fln=fl-fn;
                        if (fln==0) continue;

                        unsigned long long iv0,iv1,iv2;
                        for(int iscatter=0;iscatter<nScatterRoute;iscatter++){
                            iv0=scatterProjOrder[iscatter][0];
                            iv1=scatterProjOrder[iscatter][1];
                            iv2=scatterProjOrder[iscatter][2];
                            //printf("iv0 %d iv1 %d iv2 %d\n",iv0,iv1,iv2);
                            unsigned long long writing_scatter_index=iscatter*sca_width;
                            //if(writing_scatter_index<0){
                            //    printf("wri sca %d, isc %d, sca_width %d\n",writing_scatter_index,iscatter,sca_width);
                            //    exit(1);
                            //}
                            // calc overlap
                            lapack_complex_double px_nl=velocity[iv0*proj_width+          ikpt*kpt_width+bandn*norbs+bandl];
                            lapack_complex_double py_nl=velocity[iv0*proj_width+dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                            //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                            for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                                //if(bandn!=bandl)continue; // switch for bandn
                                double Em=energy[ikpt*norbs+bandm];
                                if(Em<erange[0] | Em>erange[1]) continue;
                                // calc overlap
                                lapack_complex_double px_lm=velocity[iv1*proj_width+          ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double py_lm=velocity[iv1*proj_width+dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double px_mn_sx=velocity_sx[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn_sx=velocity_sx[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double px_mn_sy=velocity_sy[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn_sy=velocity_sy[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double px_mn_sz=velocity_sz[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn_sz=velocity_sz[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                //printf("band%d-%d, px_lm %f %f, py_lm %f %f band%d-%d px_mn %f %f py_mn %f %f\n",bandl,bandm,creal(px_lm),cimag(px_lm),creal(py_lm),cimag(py_lm),bandm,bandn,creal(px_mn),cimag(px_mn),creal(py_mn),cimag(py_mn));
                                for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                                    unsigned long long writing_ph_index=writing_scatter_index+writing_k_index+iphoton*8;
                                    double omega=omegaList[iphoton];
                                    lapack_complex_double factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                                    //printf("iphoton %d start, pos=%d (wri sca %d, wri k %d), omega=%f factor: %f %f\n",iphoton,writing_ph_index,writing_scatter_index,writing_k_index,omega,creal(factor_global),cimag(factor_global));
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index  ]+=px_nl*px_lm*px_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+1]+=px_nl*px_lm*py_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+2]+=py_nl*py_lm*px_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+3]+=py_nl*py_lm*py_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+4]+=px_nl*py_lm*px_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+5]+=px_nl*py_lm*py_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+6]+=py_nl*px_lm*px_mn_sx*factor_global;
                                    conductance_on_kmesh_on_omega_sx[writing_ph_index+7]+=py_nl*px_lm*py_mn_sx*factor_global;

                                    conductance_on_kmesh_on_omega_sy[writing_ph_index  ]+=px_nl*px_lm*px_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+1]+=px_nl*px_lm*py_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+2]+=py_nl*py_lm*px_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+3]+=py_nl*py_lm*py_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+4]+=px_nl*py_lm*px_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+5]+=px_nl*py_lm*py_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+6]+=py_nl*px_lm*px_mn_sy*factor_global;
                                    conductance_on_kmesh_on_omega_sy[writing_ph_index+7]+=py_nl*px_lm*py_mn_sy*factor_global;

                                    conductance_on_kmesh_on_omega_sz[writing_ph_index  ]+=px_nl*px_lm*px_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+1]+=px_nl*px_lm*py_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+2]+=py_nl*py_lm*px_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+3]+=py_nl*py_lm*py_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+4]+=px_nl*py_lm*px_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+5]+=px_nl*py_lm*py_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+6]+=py_nl*px_lm*px_mn_sz*factor_global;
                                    conductance_on_kmesh_on_omega_sz[writing_ph_index+7]+=py_nl*px_lm*py_mn_sz*factor_global;
                                    //printf("iphoton %d end\n",iphoton);
                                }
                            } 
                        }
                    }
                }
            }
        }
        printf("BPVE part%d done\n",ipart+1);
        fp=fopen("./c_log","a");
        fprintf(fp,"BPVE part%d done\n",ipart+1);
        fclose(fp);
    }
    free(velocity);free(velocity_sx);free(velocity_sy);free(velocity_sz);free(energy);
    printf("All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);
    
    PyObject * conductance_real=PyTuple_New(3*_nScatterRoute*_nphoton*8);
    PyObject * conductance_imag=PyTuple_New(3*_nScatterRoute*_nphoton*8);
    printf("memory allocated\n");
    lapack_complex_double bzIntegration_sx,bzIntegration_sy,bzIntegration_sz;
    unsigned long long pos;
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long icomp=0;icomp<8;icomp++){
                bzIntegration_sx=0;bzIntegration_sy=0;bzIntegration_sz=0;
                for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                    pos=iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp;
                    bzIntegration_sx+=conductance_on_kmesh_on_omega_sx[pos];
                    bzIntegration_sy+=conductance_on_kmesh_on_omega_sy[pos];
                    bzIntegration_sz+=conductance_on_kmesh_on_omega_sz[pos];
                }
                PyTuple_SetItem(conductance_real, 0*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration_sx/nkpt)));
                PyTuple_SetItem(conductance_imag, 0*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration_sx/nkpt)));
                PyTuple_SetItem(conductance_real, 1*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration_sy/nkpt)));
                PyTuple_SetItem(conductance_imag, 1*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration_sy/nkpt)));
                PyTuple_SetItem(conductance_real, 2*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration_sz/nkpt)));
                PyTuple_SetItem(conductance_imag, 2*nScatterRoute*_nphoton*8+iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration_sz/nkpt)));
            }
        }
    }

    printf("py data setted\n");
    free(conductance_on_kmesh_on_omega_sx);free(conductance_on_kmesh_on_omega_sy);free(conductance_on_kmesh_on_omega_sz);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}

int _calcOrbitalBandsAndProjectedVelocities(
                       int nkpt, double kmesh[], // num. of kpts and frac. coor. of kpts
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int overLat[nhoppings][3],
                       double lattice[3][3],
                       double orbCorr[norbs][3], // fractional coor. of each orb
                       double erange[2], // energy range for calc.
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       lapack_complex_double S_T[norbs][norbs], lapack_complex_double S_star[norbs][norbs], // unitary transformations
                       int Lz[norbs][norbs], // eigenvalues of Lz
                       double * energy, // output energies
                       lapack_complex_double * velocities, // output projected velocities
                       lapack_complex_double * velocities_Lz, // output projected orbital velocities
                       int _nthreads){
    printf("projective orbital velocity calc. in cdll\n");

    int kpt_width_g=norbs*norbs;
    int dir_width_g=nkpt*kpt_width_g;
    int proj_width_g=2*dir_width_g;

    int nthreads;
    nthreads=_nthreads;
    if(nkpt<_nthreads*10)nthreads=1;
    printf("calc projective orbital velocity using %d cores in ham.dll\n",nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        lapack_complex_double * hk=(lapack_complex_double*)malloc(norbs*norbs*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkf=(lapack_complex_double*)malloc(nProj*norbs*norbs*3*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_Lz=(lapack_complex_double*)malloc(nProj*norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_tmp1=(lapack_complex_double*)malloc(norbs*norbs*2*sizeof(lapack_complex_double));
        lapack_complex_double * DhkDkc_tmp2=(lapack_complex_double*)malloc(norbs*norbs*2*sizeof(lapack_complex_double));
        int ipos;
        #pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            for(int i=0;i<norbs;i++){
                for(int j=0;j<norbs;j++){
                    hk[i*norbs+j]=0;
                    for(int k=0;k<nProj;k++){
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+0]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+1]=0;
                        DhkDkf[k*norbs*norbs*3+i*norbs*3+j*3+2]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                        DhkDkc_Lz[k*norbs*norbs*2+i*norbs*2+j*2+0]=0;
                        DhkDkc_Lz[k*norbs*norbs*2+i*norbs*2+j*2+1]=0;
                    }
                }
            }
            double eigs[norbs]; // output eigenvalues of Hk
            int isource,itarget;
            for(int ihopping=0;ihopping<nhoppings;ihopping++){ // setting Hk, derivated Hk, and projectively derivated Hk
                lapack_complex_double hopping_R[3];
                isource=sourceID[ihopping];
                itarget=targetID[ihopping];
                for(int i=0;i<3;i++)hopping_R[i]=(-_Complex_I)*(overLat[ihopping][i]+orbCorr[itarget][i]-orbCorr[isource][i]);

                lapack_complex_double phase=0;
                for (int i=0;i<3;i++) phase+=hopping_R[i]*kmesh[ikpt*3+i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(TwoPI*phase);
                hk[isource*norbs+itarget]+=tk;

                for(int iproj=0;iproj<nProj;iproj++){// setting projected DhkDkf
                    if(wOrbSourceProj[iproj][isource]*wOrbTargetProj[iproj][itarget]+wOrbSourceProj[iproj][itarget]*wOrbTargetProj[iproj][isource]>0){
                        for(int i=0;i<3;i++)DhkDkf[iproj*norbs*norbs*3+isource*norbs*3+itarget*3+i]+=tk*hopping_R[i];
                    }
                }
            }
            // unitary transform into real space partial
            for(int iproj=0;iproj<nProj;iproj++){
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<3;k++){
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+0]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][0];
                            DhkDkc[iproj*norbs*norbs*2+i*norbs*2+j*2+1]+=DhkDkf[iproj*norbs*norbs*3+i*norbs*3+j*3+k]*lattice[k][1];
                        }
                    }
                }
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<2;k++){
                            DhkDkc_tmp1[i*norbs*2+j*2+k]=0;
                            DhkDkc_tmp2[i*norbs*2+j*2+k]=0;
                        }
                    }
                }
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<2;k++){
                            for(int icontraction=0;icontraction<norbs;icontraction++){
                                for(int jcontraction=0;jcontraction<norbs;jcontraction++){
                                    DhkDkc_tmp1[i*norbs*2+j*2+k]+=S_star[i][icontraction]*DhkDkc[iproj*norbs*norbs*2+icontraction*norbs*2+jcontraction*2+k]*S_T[jcontraction][j];
                                }
                            }
                        }
                    }
                }
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<2;k++){
                            for(int icontraction=0;icontraction<norbs;icontraction++){
                                DhkDkc_tmp2[i*norbs*2+j*2+k]+=DhkDkc_tmp1[i*norbs*2+icontraction*2+k]*Lz[icontraction][j] + Lz[i][icontraction]*DhkDkc_tmp1[icontraction*norbs*2+j*2+k];
                            }
                        }
                    }
                }
                for(int i=0;i<norbs;i++){
                    for(int j=0;j<norbs;j++){
                        for(int k=0;k<2;k++){
                            for(int icontraction=0;icontraction<norbs;icontraction++){
                                for(int jcontraction=0;jcontraction<norbs;jcontraction++){
                                    DhkDkc_Lz[iproj*norbs*norbs*2+i*norbs*2+j*2+k]+=S_T[i][icontraction]*DhkDkc_tmp2[icontraction*norbs*2+jcontraction*2+k]*S_star[jcontraction][j]/2;
                                }
                            }
                        }
                    }
                }
            }

            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,hk,norbs,eigs);
            // vec[i]=hk[:][i]
            if(info!=0){
                printf("LAPACKE failed, info: %d\n",info);
                exit(1);
            }
            double Em, En;
            for(int bandm=0;bandm<norbs;bandm++){
                Em=eigs[bandm];
                energy[ikpt*norbs+bandm]=Em;
                if(Em>erange[1] | Em<erange[0])continue;
                for(int bandn=0;bandn<norbs;bandn++){
                    En=eigs[bandn];
                    if(En>erange[1] | En<erange[0])continue;
                    for(int iproj=0;iproj<nProj;iproj++){
                        for(int idir=0;idir<2;idir++){
                            ipos=iproj*proj_width_g+idir*dir_width_g+ikpt*kpt_width_g+bandm*norbs+bandn;
                            velocities[ipos]=0;velocities_Lz[ipos]=0;
                            for(int icontract=0;icontract<norbs;icontract++){
                                for(int jcontract=0;jcontract<norbs;jcontract++){
                                    velocities[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                    velocities_Lz[ipos]+=conj(hk[icontract*norbs+bandm])*DhkDkc_Lz[iproj*norbs*norbs*2+icontract*norbs*2+jcontract*2+idir]*hk[jcontract*norbs+bandn];
                                }
                            }
                        }
                    }
                }
            }  
        }
        free(hk);free(DhkDkf);free(DhkDkc);free(DhkDkc_Lz);free(DhkDkc_tmp1);free(DhkDkc_tmp2);
    }
    //fp=fopen("./c_log","a");
    //fprintf(fp,"projective velocity completed\n");
    //fclose(fp);
    return 0;
}

PyObject * calcOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce(
                       int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int nphoton,
                       double omegaList[nphoton],
                       double fermi,
                       double lattice[3][3], // lattice vectors: [a_1,a_2,a_3]
                       double orbCorr[norbs][3], // fractional corrdinates of orbs
                       double tau,
                       double erange[2],
                       // projection part
                       int nProj, // num. of projections
                       int wOrbSourceProj[nProj][norbs], // index of source orb in each projection
                       int wOrbTargetProj[nProj][norbs], // index of target orb in each projection
                       int nScatterRoute, // num. of scattering routes
                       int scatterProjOrder[nScatterRoute][3], // proj. vel. order in each scattering route
                       // transformation from atomic rep. to Lz rep.
                       double tTBLz_real[norbs][norbs], double tTBLz_imag[norbs][norbs],
                       int Lz[norbs][norbs],
                       // parallel part
                       int ndivide, // num. of parts
                       int _nthreads){
    printf("entering Orbital projective BPVC module in cdll\n");
    FILE*fp;
    fp=fopen("./c_log","w");
    fprintf(fp,"#Log file of ham.dll\n");
    // verification
    fprintf(fp,"nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    fprintf(fp,"memory estimation: %.3f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(4*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    fclose(fp);
    printf("nkpt: %d, norbs: %d, nhoppings: %d, nphotons: %d, erange: %.4f ~ %.4f\n",nkpt,norbs,nhoppings,nphoton,erange[0],erange[1]);
    printf("memory estimation: %.4f MB\n",(nphoton*(double)nkpt/1024/1024*16+(double)(4*norbs*norbs+norbs)/1024/1024*nkpt)/ndivide*nProj*sizeof(lapack_complex_double));
    //exit(1);
    lapack_complex_double S_T[norbs][norbs];
    lapack_complex_double S_star[norbs][norbs];
    for(int i=0;i<norbs;i++){
        for(int j=0;j<norbs;j++){
            S_T[i][j]    =tTBLz_real[j][i]+_Complex_I*tTBLz_imag[j][i];
            S_star[i][j] =tTBLz_real[i][j]-_Complex_I*tTBLz_imag[i][j];
        }
    }
    /*lapack_complex_double tmp;
    for(int i=0;i<norbs;i++){
        tmp=0;
        for(int j=0;j<norbs;j++) tmp+=S_T[i][j]*S_star[j][i];
        printf("%f %f\n",creal(tmp),cimag(tmp));
    }
    exit(1);*/

    int nkpt_part=nkpt/ndivide;
    unsigned long long kpt_width=norbs*norbs;
    unsigned long long dir_width=nkpt_part*kpt_width;
    unsigned long long proj_width=2*dir_width;

    unsigned long long _nphoton=nphoton;
    unsigned long long _nkpt=nkpt;
    unsigned long long sca_width=_nkpt*_nphoton*8;

    unsigned long long _nScatterRoute=nScatterRoute;

    lapack_complex_double itau=tau*_Complex_I;
    lapack_complex_double * conductance_on_kmesh_on_omega=(lapack_complex_double*)malloc(_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8*sizeof(lapack_complex_double));
    for(unsigned long long i=0;i<_nScatterRoute*_nphoton*_nkpt*(unsigned long long)8;i++)conductance_on_kmesh_on_omega[i]=0;
    
    int ikpt_head;
    if(nkpt%ndivide!=0){
        printf("Error: nkpt %d cannot be divide by %d, exiting now!\n",nkpt,ndivide);
        exit(0);
    }
    lapack_complex_double *velocity=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    lapack_complex_double *velocity_Lz=(lapack_complex_double*)malloc(nProj*2*nkpt_part*norbs*norbs*sizeof(lapack_complex_double));
    double *energy=(double*)malloc(nkpt_part*norbs*sizeof(double));
    double *sub_kmesh=(double*)malloc(nkpt_part*3*sizeof(double));
    
    int nthreads;
    nthreads=_nthreads;
    if(nkpt_part<_nthreads*10)nthreads=1;

    for(int ipart=0;ipart<ndivide;ipart++){// kmesh partition
        ikpt_head=nkpt_part*ipart;
        for(int i=0;i<nkpt_part;i++)for(int j=0;j<3;j++)sub_kmesh[i*3+j]=kmesh[i+ikpt_head][j];

        printf("use %d cores in ham.dll\n",nthreads);
        fp=fopen("./c_log","a");
        fprintf(fp,"calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        fclose(fp);
        printf("calc velocities (part%d/%d) using %d cores in ham.dll\n",ipart+1,ndivide,nthreads);
        
        int status=_calcOrbitalBandsAndProjectedVelocities(
                       nkpt_part, sub_kmesh, // num. of kpts and frac. coor. of kpts
                       norbs, nhoppings,
                       sourceID, targetID, 
                       hopping_real,
                       hopping_imag,
                       overLat,
                       lattice,
                       orbCorr, // fractional coor. of each orb
                       erange, // energy range for calc.
                       nProj, // num. of projections
                       wOrbSourceProj, // index of source orb in each projection
                       wOrbTargetProj, // index of target orb in each projection
                       S_T, S_star, Lz,
                       energy, // output energies
                       velocity, // output projected velocities
                       velocity_Lz, // output projected oribital velocities
                       nthreads);
        if(status!=0){
            printf("Error: velocity calc. failed\n");
            exit(1);
        }
        fp=fopen("./c_log","a");
        fprintf(fp,"velocities part%d completed\ncalc BPVE...\n",ipart+1);
        fclose(fp);
        printf("velocities part%d completed\nnow calc BPVE use %d cores in ham.dll\n",ipart+1,nthreads);

        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for
            for(int ikpt=0;ikpt<nkpt_part;ikpt++){// loop over kmesh
                unsigned long long ikpt_true=ikpt_head+ikpt;
                unsigned long long writing_k_index=ikpt_true*nphoton*8;
                // printf("ikpt_true: %d\n",ikpt_true);
                for(int bandl=0;bandl<norbs;bandl++){ // loop over band_l
                    double El=energy[ikpt*norbs+bandl];
                    if(El<erange[0] | El>erange[1]) continue;
                    int fl=0;
                    if(El<=fermi)fl=1;
                    for(int bandn=0;bandn<norbs;bandn++){ // loop over band_m
                        double En=energy[ikpt*norbs+bandn];
                        if(En<erange[0] | En>erange[1]) continue;
                        int fn=0;
                        if(En<=fermi)fn=1;
                        int fln=fl-fn;
                        if (fln==0) continue;

                        unsigned long long iv0,iv1,iv2;
                        for(int iscatter=0;iscatter<nScatterRoute;iscatter++){
                            iv0=scatterProjOrder[iscatter][0];
                            iv1=scatterProjOrder[iscatter][1];
                            iv2=scatterProjOrder[iscatter][2];
                            //printf("iv0 %d iv1 %d iv2 %d\n",iv0,iv1,iv2);
                            unsigned long long writing_scatter_index=iscatter*sca_width;
                            //if(writing_scatter_index<0){
                            //    printf("wri sca %d, isc %d, sca_width %d\n",writing_scatter_index,iscatter,sca_width);
                            //    exit(1);
                            //}
                            // calc overlap
                            lapack_complex_double px_nl=velocity[iv0*proj_width+          ikpt*kpt_width+bandn*norbs+bandl];
                            lapack_complex_double py_nl=velocity[iv0*proj_width+dir_width+ikpt*kpt_width+bandn*norbs+bandl];
                            //printf("band%d-%d, px_nl %f %f, py_nl %f %f\n",bandn,bandl,creal(px_nl),cimag(px_nl),creal(py_nl),cimag(py_nl));
                            for(int bandm=0;bandm<norbs;bandm++){ // loop over band_n
                                //if(bandn!=bandl)continue; // switch for bandn
                                double Em=energy[ikpt*norbs+bandm];
                                if(Em<erange[0] | Em>erange[1]) continue;
                                // calc overlap
                                lapack_complex_double px_lm=velocity[iv1*proj_width+          ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double py_lm=velocity[iv1*proj_width+dir_width+ikpt*kpt_width+bandl*norbs+bandm];
                                lapack_complex_double px_mn=velocity_Lz[iv2*proj_width+          ikpt*kpt_width+bandm*norbs+bandn];
                                lapack_complex_double py_mn=velocity_Lz[iv2*proj_width+dir_width+ikpt*kpt_width+bandm*norbs+bandn];
                                //printf("band%d-%d, px_lm %f %f, py_lm %f %f band%d-%d px_mn %f %f py_mn %f %f\n",bandl,bandm,creal(px_lm),cimag(px_lm),creal(py_lm),cimag(py_lm),bandm,bandn,creal(px_mn),cimag(px_mn),creal(py_mn),cimag(py_mn));
                                for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){ // loop over photon
                                    unsigned long long writing_ph_index=writing_scatter_index+writing_k_index+iphoton*8;
                                    double omega=omegaList[iphoton];
                                    lapack_complex_double factor_global=fln/(omega*omega)/(En-Em-itau)*(1/(En-El+omega+itau)+1/(En-El-omega-itau));
                                    //printf("iphoton %d start, pos=%d (wri sca %d, wri k %d), omega=%f factor: %f %f\n",iphoton,writing_ph_index,writing_scatter_index,writing_k_index,omega,creal(factor_global),cimag(factor_global));
                                    conductance_on_kmesh_on_omega[writing_ph_index  ]+=px_nl*px_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+1]+=px_nl*px_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+2]+=py_nl*py_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+3]+=py_nl*py_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+4]+=px_nl*py_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+5]+=px_nl*py_lm*py_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+6]+=py_nl*px_lm*px_mn*factor_global;
                                    conductance_on_kmesh_on_omega[writing_ph_index+7]+=py_nl*px_lm*py_mn*factor_global;
                                    //printf("iphoton %d end\n",iphoton);
                                }
                            } 
                        }
                    }
                }
            }
        }
        printf("BPVE part%d done\n",ipart+1);
        fp=fopen("./c_log","a");
        fprintf(fp,"BPVE part%d done\n",ipart+1);
        fclose(fp);
    }
    free(velocity);free(velocity_Lz);free(energy);
    printf("All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fp=fopen("./c_log","a");
    fprintf(fp,"All calc. done! pls. times 3.6981286E-19 (C m/s) (m/V)^2 to get the current (Ampere) per 1V/m field per one cell\n");
    fclose(fp);
    
    
    PyObject * conductance_real=PyTuple_New(_nScatterRoute*_nphoton*8);
    PyObject * conductance_imag=PyTuple_New(_nScatterRoute*_nphoton*8);
    printf("memory allocated\n");
    lapack_complex_double bzIntegration;
    unsigned long long pos;
    for(unsigned long long iscatter=0;iscatter<nScatterRoute;iscatter++){
        for(unsigned long long iphoton=0;iphoton<nphoton;iphoton++){
            for(unsigned long long icomp=0;icomp<8;icomp++){
                bzIntegration=0;
                for(unsigned long long ikpt=0;ikpt<nkpt;ikpt++){
                    pos=iscatter*sca_width+ikpt*_nphoton*8+iphoton*8+icomp;
                    bzIntegration+=conductance_on_kmesh_on_omega[pos];
                }
                PyTuple_SetItem(conductance_real, iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(creal(bzIntegration/nkpt)));
                PyTuple_SetItem(conductance_imag, iscatter*_nphoton*8+icomp*_nphoton+iphoton, PyFloat_FromDouble(cimag(bzIntegration/nkpt)));
            }
        }
    }

    printf("py data setted\n");
    free(conductance_on_kmesh_on_omega);
    return Py_BuildValue("(OO)",conductance_real,conductance_imag);
}
