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

PyObject * calcFullBands(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int _nthreads){

    PyObject * Spectra=PyTuple_New(nkpt);
    //int nthreads;
    //nthreads=_nthreads;
    //if(nkpt<_nthreads*10)nthreads=1;
    // calculate matrix elements
    //#pragma omp parallel num_threads(nthreads)
    {
        //#pragma omp for
        for(int ikpt=0;ikpt<nkpt;ikpt++){// loop over kmesh
            int kBaseIndex=ikpt*norbs*norbs;
            lapack_complex_double hk[norbs][norbs]; // init Hk
            for(int i=0;i<norbs;i++)for(int j=0;j<norbs;j++)hk[i][j]=0;
            double eigs[norbs]; // output eigenvalues of Hk
            for(int ihopping=0;ihopping<nhoppings;ihopping++){
                double phase=0;
                for (int i=0;i<3;i++) phase+=overLat[ihopping][i]*kmesh[ikpt][i];
                lapack_complex_double tk=(hopping_real[ihopping]+hopping_imag[ihopping]*_Complex_I)*cexp(-TwoPI*_Complex_I*phase);
                hk[sourceID[ihopping]][targetID[ihopping]]+=tk;
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'N','U',norbs,*hk,norbs,eigs);
            PyObject * eig_k=PyTuple_New(norbs);
            for(int iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(eig_k, iorb, PyFloat_FromDouble(eigs[iorb]));
            PyTuple_SetItem(Spectra, ikpt, eig_k);
        }
    }
    return Spectra;
}

PyObject * calcFullBandsAndVecs(int nkpt, double kmesh[nkpt][3],
                       int norbs, int nhoppings,
                       int sourceID[nhoppings], int targetID[nhoppings], 
                       int overLat[nhoppings][3], 
                       double hopping_real[nhoppings],
                       double hopping_imag[nhoppings],
                       int _nthreads){

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
    //int nthreads;
    //nthreads=_nthreads;
    //if(nkpt<_nthreads*10)nthreads=1;
    // calculate matrix elements
    //#pragma omp parallel num_threads(nthreads)
    {
        //#pragma omp for
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
                    PyTuple_SetItem(Hk_r, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(creal(hk[iorb][jorb])));
                    PyTuple_SetItem(Hk_i, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(cimag(hk[iorb][jorb])));
                    PyTuple_SetItem(Hk_dk1_r, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(creal(Dhk[iorb][jorb][0])));
                    PyTuple_SetItem(Hk_dk1_i, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(cimag(Dhk[iorb][jorb][0])));
                    PyTuple_SetItem(Hk_dk2_r, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(creal(Dhk[iorb][jorb][1])));
                    PyTuple_SetItem(Hk_dk2_i, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(cimag(Dhk[iorb][jorb][1])));
                    PyTuple_SetItem(Hk_dk3_r, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(creal(Dhk[iorb][jorb][2])));
                    PyTuple_SetItem(Hk_dk3_i, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(cimag(Dhk[iorb][jorb][2])));
                    //if(ikpt==0)printf("%.6e %.6e, ",creal(Dhk[iorb][jorb][0]),cimag(Dhk[iorb][jorb][0]));
                }
                //if(ikpt==0)printf("\n");
            }
            int info=LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','L',norbs,*hk,norbs,eigs);
            if(info!=0)printf("LAPACKE failed, info: %d\n",info);
            //PyObject * eig_k=PyTuple_New(norbs);
            for(int iorb=0;iorb<norbs;iorb++) PyTuple_SetItem(Spectra, ikpt*norbs+iorb, PyFloat_FromDouble(eigs[iorb]));
            for(int iorb=0;iorb<norbs;iorb++){
                for(int jorb=0;jorb<norbs;jorb++){
                    PyTuple_SetItem(Vector_r, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(creal(hk[iorb][jorb])));
                    PyTuple_SetItem(Vector_i, ikpt*norbs*norbs+iorb*norbs+jorb, PyFloat_FromDouble(cimag(hk[iorb][jorb])));
                }
            }
            //PyTuple_SetItem(Spectra, ikpt, eig_k);
        }
    }
    return Py_BuildValue("(OOOOOOOOOOO)",Spectra,Vector_r,Vector_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i);
}