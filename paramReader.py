#
# use to read GlobalConst file and store global variables
#
import re,os

global NPAR,KPAR,PREC,ALGO,EDIFF,CDIFF,CTYPE,ISMEAR,SIGMA,NELM
global DFTSTD,DFTNCL,DFTGAM,DFTOPT,MINMAG,MAXSTR,MAXRANGE,SYMEXPAND
global MINCELL,MAXCELL,MAXTRISTR,MAXDFTTIME,MAXINV,ENMAX_aug
global CALRANGE,BONDTOL,MAGTOL,initMagEnhance,METAGGA,KDEN,POTROOT,EV2K,MAXCOMB
global NCORE,OVERIDE,MCSIZE,THERMALIZATION,SWEEPS,MINTEMP,MAXTEMP
global SWNKPT,SWCVG,SWTMIN,SWTMAX,MCCORE,TMIN,TSTEP,MCALGO,CVGRANGE
global doneList,BONDBIAS,BONDINPT,ADDGRID,LREAL,NINFO4STR,METALSOLFT
global SCFGRID,SYMPREC,PGHOLD,RSVSYM,TIME,FORCEZ,SYMAFM,MAXMCSTEP
global AMIX,OPTAMIX,BMIX,BMIX_MAG,GPUMODE,NODES,PPN,waitBeforeDFT
global U0,DU,USTEP,TOVERLAP,PRECHECKAFM,CHECKEAZYAXIS,CPERCENTAGE
global EDIFFG,NSW,CALCANI,MCCALC,PROPOSAL,CELLRLX,DSN,PBC,TOTMAGTOL
global IVDW,COUPLINGTOL,BANDCALC,KINTERVAL,BANDLOW,BANDHIGH,DOSLEGEND
global DOSCALC,PDOS,LBOUND,DOSKFACTOR,NBANDFACTOR,NEDOS,PERTURBANI
global PBSMODE,AMIX0,AMIX_step,nAMIX,CHECKINTEMAG,SPINFOLD,MAGTYPE
global ANIPOLAR,NTHETA,LOCALFRAMERANGE,EXHCALC,RWIGSOVERLAP,ZAXIS,LAMBDA0,DLAMBDA
global HFSCREEN,RLXONLY,PHONON,PHONON_SC,MINFREQUENCY,IONRLX,FIXTOTMAG

global color_pool
color_pool=['red','green','blue','yellow','orange','violet']

def readGlobalConst():
    global rpath,prescript,enmax,NPAR,KPAR,PREC,ALGO,TIME,EDIFF,CDIFF,ENMAX_aug
    global ISMEAR,SIGMA,NELM,MINMAG,MAXSTR,MAXRANGE,NCORE,SYMEXPAND,OVERIDE
    global MINCELL,DFTSTD,DFTNCL,DFTGAM,DFTOPT,MAXCELL,MAXTRISTR,CTYPE,POTROOT,SYMAFM
    global MAXDFTTIME,MAXINV,CALRANGE,BONDTOL,MAGTOL,METAGGA,KDEN,EV2K,NINFO4STR
    global MCSIZE,THERMALIZATION,SWEEPS,MINTEMP,MAXTEMP,SWNKPT,SWCVG,SWTMIN,SWTMAX
    global MCCORE,TMIN,TSTEP,MCALGO,CVGRANGE,BONDBIAS,BONDINPT,ADDGRID,LREAL,OPTAMIX
    global SCFGRID,SYMPREC,PGHOLD,RSVSYM,FORCEZ,AMIX,BMIX,BMIX_MAG,U0,DU,USTEP
    global EDIFFG,NSW,CALCANI,MCCALC,MAXCOMB,GPUMODE,NODES,PPN,DSN,waitBeforeDFT
    global TOVERLAP,MAXMCSTEP,PROPOSAL,METALSOLFT,PRECHECKAFM,CHECKEAZYAXIS,DOSLEGEND
    global CPERCENTAGE,CELLRLX,IVDW,COUPLINGTOL,BANDCALC,KINTERVAL,BANDLOW,BANDHIGH
    global DOSCALC,PDOS,LBOUND,DOSKFACTOR,NBANDFACTOR,PBC,NEDOS,PERTURBANI,SPINFOLD
    global initMagEnhance,TOTMAGTOL,PBSMODE,AMIX0,AMIX_step,nAMIX,CHECKINTEMAG
    global ANIPOLAR,NTHETA,LOCALFRAMERANGE,BONDSEARCHRANGE,VERBOSE,MAGTYPE,EXHCALC
    global RWIGSOVERLAP,ZAXIS,LAMBDA0,DLAMBDA,HFSCREEN,RLXONLY,PHONON,PHONON_SC,MINFREQUENCY
    global IONRLX,FIXTOTMAG

    # read global constants
    try:
        fgConst=open('./GlobalConst','r')
        gConst=[line for line in fgConst.read().split('\n') if line]
        fgConst.close()
    except:
        print('ERR: cannot open GlobalConst. make sure it is here')
        exit()
    POTROOT='/home/bright/Document/LL/POT/'
    PBSMODE=0
    VERBOSE=True
    OVERIDE=1
    PBC=2
    ENMAX_aug=50.
    METAGGA='F'
    GPUMODE='F'
    PROPOSAL=False
    RLXONLY=False
    IONRLX=True
    PHONON=False
    PHONON_SC=[2, 2, 2]
    MINFREQUENCY=-0.2
    BANDCLAC=False
    KINTERVAL=30
    BANDLOW=-3
    BANDHIGH=2
    DOSCALC=False
    RWIGSOVERLAP=0.2
    NEDOS=1000
    PERTURBANI=True
    DOSKFACTOR=4
    NBANDFACTOR=2
    PDOS='none'
    DOSLEGEND=True
    LBOUND=True
    ANIPOLAR=True
    NTHETA=10
    LOCALFRAMERANGE=1.5
    BONDSEARCHRANGE=3
    CELLRLX=2
    CHECKEAZYAXIS=True
    CHECKINTEMAG=False
    IVDW=-1
    EXHCALC=3
    CALCANI='z'
    PRECHECKAFM=True
    EDIFFG =-1e-3
    NSW=300
    U0,DU,USTEP=-1.,0.,1
    HFSCREEN=-1
    NPAR=2
    KPAR=2
    PREC='M'
    ADDGRID='F'
    LREAL='A'
    ALGO='F'
    AMIX=0.02
    OPTAMIX=False
    AMIX0=0.02
    AMIX_step=0.02
    nAMIX=15
    BMIX=0.0001
    BMIX_MAG=0.0001
    TIME=0.4
    EDIFF=1E-5
    BONDTOL=0.01
    MAGTOL=0.3
    TOTMAGTOL=0.1
    FIXTOTMAG=False
    initMagEnhance=1.0
    MAGTYPE='ISING'
    METALSOLFT=2
    SYMPREC=0.01
    CDIFF=0.05
    CPERCENTAGE=0.1
    CTYPE='COUPLING'
    COUPLINGTOL=0.001
    ISMEAR='-5'
    SIGMA='0.1'
    NELM='200'
    MINMAG=0.5
    MAXSTR=5
    MAXCOMB=100000
    NINFO4STR=50
    MAXTRISTR=1
    MAXRANGE=3
    CALRANGE=2
    MINCELL=2
    MAXCELL=4
    KDEN=30
    MAXINV=2
    SYMEXPAND=True
    SYMAFM=1
    RSVSYM=2
    PGHOLD=0.5
    NCORE=1
    NODES=2
    PPN=16
    DFTSTD='vasp_std'
    DFTNCL='vasp_ncl'
    DFTGAM='vasp_gam'
    DFTOPT='vasp_opt'
    waitBeforeDFT=2
    MAXDFTTIME=7200
    BONDBIAS=0.1
    BONDINPT=20
    ZAXIS=[0,0,1]
    LAMBDA=1.0
    DLAMBDA=1.0

    SWNKPT=32
    DSN=0.5
    SWCVG=0.1
    SWTMIN=0.1
    SWTMAX=0.5
    SCFGRID=200
    TSTEP=1.0
    FORCEZ=-1
 
    MCCALC='T'
    MCSIZE=16
    THERMALIZATION=20000
    SWEEPS=100000
    MCALGO='loop'
    TMIN=10
    MINTEMP=5
    MAXTEMP=200 
    TOVERLAP=0.3
    CVGRANGE=5
    MCCORE=20
    MAXMCSTEP=128
    SPINFOLD=1

    EV2K=11588.75
    for line in gConst:
        datapack=line.split('=')
        if len(datapack)<2:
            continue
        lhs, rhs= datapack[0], datapack[1]
        tag=lhs.strip() # delete space
        value=rhs.split('#')[0].strip() # ignore info. after # and ignore space 
        if(tag=='POTROOT'):
            POTROOT=value
        if(tag=='OVERIDE'):
            OVERIDE=bool(int(value))
        if(tag=='PBSMODE'):
            PBSMODE=int(value)
        if(tag=='VERBOSE'):
            VERBOSE=True if value=='T' else False
        if(tag=='PBC'):
            PBC=int(value)
        if(tag=='ENMAX_aug'):
            ENMAX_aug=float(value)
        if(tag=='METAGGA'):
            METAGGA=value
        if(tag=='PROPOSAL'):
            PROPOSAL=True if value=='T' else False
        if(tag=='RLXONLY'):
            RLXONLY=True if value=='T' else False
        if(tag=='PHONON'):
            PHONON=True if value=='T' else False
        if(tag=='PHONON_SC'):
            PHONON_SC=[int(x) for x in value.split()]
        if(tag=='MINFREQUENCY'):
            MINFREQUENCY=float(value)
        if(tag=='CELLRLX'):
            CELLRLX=int(value)
        if(tag=='IONRLX'):
            IONRLX=True if value=='T' else False
        if(tag=='CHECKEAZYAXIS'):
            CHECKEAZYAXIS=True if value=='T' else False
        if(tag=='CHECKINTEMAG'):
            CHECKINTEMAG=True if value=='T' else False
        if(tag=='BANDCALC'):
            BANDCALC=True if value=='T' else False
        if(tag=='KINTERVAL'):
            KINTERVAL=int(value)
        if(tag=='BANDLOW'):
            BANDLOW=int(value)
        if(tag=='BANDHIGH'):
            BANDHIGH=int(value)
        if(tag=='DOSCALC'):
            DOSCALC=True if value=='T' else False
        if(tag=='RWIGSOVERLAP'):
            RWIGSOVERLAP=float(value)
        if(tag=='NEDOS'):
            NEDOS=int(value)
        if(tag=='PERTURBANI'):
            PERTURBANI=True if value=='T' else False
        if(tag=='DOSKFACTOR'):
            DOSKFACTOR=int(value)
        if(tag=='NBANDFACTOR'):
            NBANDFACTOR=int(value)
        if(tag=='PDOS'):
            PDOS=value
        if(tag=='DOSLEGEND'):
            DOSLEGEND=True if value=='T' else False
        if(tag=='LBOUND'):
            LBOUND=True if value=='T' else False
        if(tag=='ANIPOLAR'):
            ANIPOLAR=True if value=='T' else False
        if(tag=='NTHETA'):
            NTHETA=int(value)
        if(tag=='LOCALFRAMERANGE'):
            LOCALFRAMERANGE=float(value)
        if(tag=='BONDSEARCHRANGE'):
            BONDSEARCHRANGE=int(value)
        if(tag=='IVDW'):
            IVDW=int(value)
        if(tag=='GPUMODE'):
            GPUMODE=value
        if(tag=='EXHCALC'):
            EXHCALC=int(value)
        if(tag=='CALCANI'):
            CALCANI=value
        if(tag=='PRECHECKAFM'):
            PRECHECKAFM=True if value=='T' else False
        if(tag=='EDIFFG'):
            EDIFFG=float(value)
        if(tag=='NSW'):
            NSW=int(value)
        if(tag=='U0'):
            U0=float(value)
        if(tag=='DU'):
            DU=float(value)
        if(tag=='USTEP'):
            USTEP=int(value)
        if(tag=='HFSCREEN'):
            HFSCREEN=float(value)
        if(tag=='NPAR'):
            NPAR=int(value)
        if(tag=='KPAR'):
            KPAR=int(value)
        if(tag=='PREC'):
            PREC=value
        if(tag=='ADDGRID'):
            ADDGRID=value
        if(tag=='LREAL'):
            LREAL=value
        if(tag=='ALGO'):
            ALGO=value
        if(tag=='AMIX'):
            AMIX=float(value)
        if(tag=='OPTAMIX'):
            OPTAMIX=True if value=='T' else False
        if(tag=='AMIX0'):
            AMIX0=float(value)
        if(tag=='AMIX_step'):
            AMIX_step=float(value)
        if(tag=='nAMIX'):
            nAMIX=int(value)
        if(tag=='BMIX'):
            BMIX=float(value)
        if(tag=='BMIX_MAG'):
            BMIX_MAG=float(value)
        if(tag=='TIME'):
            TIME=float(value)
        if(tag=='EDIFF'):
            EDIFF=float(value)
        if(tag=='CDIFF'):
            CDIFF=float(value)
        if(tag=='CPERCENTAGE'):
            CPERCENTAGE=float(value)
        if(tag=='CTYPE'):
            CTYPE=value
        if(tag=='COUPLINGTOL'):
            COUPLINGTOL=float(value)
        if(tag=='KDEN'):
            KDEN=int(value)
        if(tag=='BONDTOL'):
            BONDTOL=float(value)
        if(tag=='MAGTOL'):
            MAGTOL=float(value)
        if(tag=='TOTMAGTOL'):
            TOTMAGTOL=float(value)
        if(tag=='FIXTOTMAG'):
            FIXTOTMAG=True if value =='T' else False
        if(tag=='initMagEnhance'):
            initMagEnhance=float(value)
        if(tag=='METALSOLFT'):
            METALSOLFT=float(value)
        if(tag=='SYMPREC'):
            SYMPREC=float(value)
        if(tag=='PGHOLD'):
            PGHOLD=float(value)
        if(tag=='RSVSYM'):
            RSVSYM=int(value)
        if(tag=='ISMEAR'):
            ISMEAR=value
        if(tag=='SIGMA'):
            SIGMA=value
        if(tag=='NELM'):
            NELM=value
        if(tag=='MAGTYPE'):
            MAGTYPE=value
        if(tag=='MINMAG'):
            MINMAG=float(value)
        if(tag=='MAXSTR'):
            MAXSTR=int(value)
        if(tag=='MAXTRISTR'):
            MAXTRISTR=int(value)
        if(tag=='MAXCOMB'):
            MAXCOMB=int(value)
        if(tag=='NINFO4STR'):
            NINFO4STR=int(value)
        if(tag=='MAXRANGE'):
            MAXRANGE=int(value)
        if(tag=='CALRANGE'):
            CALRANGE=int(value)
        if(tag=='MINCELL'):
            MINCELL=int(value)
        if(tag=='MAXCELL'):
            MAXCELL=int(value)
        if(tag=='MAXINV'):
            MAXINV=int(value)
        if(tag=='SYMEXPAND'):
            SYMEXPAND=True if value=='T' else False
        if(tag=='SYMAFM'):
            SYMAFM=int(value)
        if(tag=='NCORE'):
            NCORE=int(value)
        if(tag=='NODES'):
            NODES=int(value)
        if(tag=='PPN'):
            PPN=int(value)
        if(tag=='DFTSTD'):
            DFTSTD=value
        if(tag=='DFTNCL'):
            DFTNCL=value
        if(tag=='DFTGAM'):
            DFTGAM=value
        if(tag=='DFTOPT'):
            DFTOPT=value
        if(tag=='BONDBIAS'):
            BONDBIAS=float(value)
        if(tag=='BONDINPT'):
            BONDINPT=int(value)
        if(tag=='ZAXIS'):
            ZAXIS=[float(x) for x in value.split()]
            assert len(ZAXIS)==3
        if(tag=='LAMBDA0'):
            LAMBDA0=float(value)
        if(tag=='DLAMBDA'):
            DLAMBDA=float(value)
        if(tag=='waitBeforeDFT'):
            waitBeforeDFT=int(value)
        if(tag=='MAXDFTTIME'):
            MAXDFTTIME=int(value)
        if(tag=='FORCEZ'):
            FORCEZ=int(value)
        if(tag=='SWNKPT'):
            SWNKPT=int(value)
        if(tag=='DSN'):
            DSN=float(value)
        if(tag=='SWCVG'):
            SWCVG=float(value)
        if(tag=='SWTMIN'):
            SWTMIN=float(value)
        if(tag=='SWTMAX'):
            SWTMAX=float(value)
        if(tag=='SCFGRID'):
            SCFGRID=int(value)
        if(tag=='MCCALC'):
            MCCALC=value
        if(tag=='MCSIZE'):
            MCSIZE=int(value)
        if(tag=='THERMALIZATION'):
            THERMALIZATION=int(value)
        if(tag=='SWEEPS'):
            SWEEPS=int(value)
        if(tag=='MCALGO'):
            MCALGO=value
        if(tag=='TMIN'):
            TMIN=float(value)
        if(tag=='MINTEMP'):
            MINTEMP=int(value)
        if(tag=='MAXTEMP'):
            MAXTEMP=int(value)
        if(tag=='TSTEP'):
            TSTEP=float(value)
        if(tag=='TOVERLAP'):
            TOVERLAP=float(value)
        if(tag=='CVGRANGE'):
            CVGRANGE=int(value)
        if(tag=='MCCORE'):
            MCCORE=int(value)
        if(tag=='MAXMCSTEP'):
            MAXMCSTEP=int(value)
        if(tag=='SPINFOLD'):
            SPINFOLD=int(value)
        if(tag=='EV2K'):
            EV2K=float(value)

def readxxyOUT():
    global doneList
    doneList=[]
    try:
        fxxyOUT=open('xxyOUT','r')
    except:
        return
    xxyOUT=[line for line in fxxyOUT.read().split('\n') if line]
    fxxyOUT.close()
    for line in xxyOUT:
        pack=re.findall('[A-Z\_a-z0-9\.\-]+',line)
        if(len(pack)>0 and pack[0]=='title'):
            doneList.append(pack[1])
