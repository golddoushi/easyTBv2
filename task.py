# Module to implement the parallel tasks
# 
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from auxiliary import showElapsedTime, SYMPREC
from system import System
from calculator import Velocity, Velocity_diag, Energy, SpinBerryCurv, PDOS, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath
from calculator import Spin, Orbital, OrbitalModern_Full, SpinOrbital
from calculator import AnormalousHallConductance, AnormalousHallConductance_diag, AnormalousHallConductance_LG_DM, BulkPhotovoltaicConductance_VG, BulkPhotovoltaicConductance_VG_DM, BulkPhotovoltaicConductance_LG_Shift, BulkPhotovoltaicConductance_LG_Injection, BulkPhotovoltaicConductance_LG_DM, BulkPhotovoltaicConductance_TDSE, BulkPhotovoltaicConductance_TDSE_LG_Naive
from calculator import InverseFaradayEffect_TDSE, InverseFaradayEffect_BlochAcceleration_TDSE, InverseFaradayEffect_VG, InverseFaradayEffect_VG_DM, InverseFaradayEffect_LG, InverseFaradayEffect_LG_DM
from calculator import InverseFaradayTorque_TDSE, InverseFaradayTorque_VG_DM, InverseFaradayTorque_LG_DM
from calculator import SpinHallConductance, SpinHallConductance_diag, OrbitalHallConductance
from calculator import SpinGalvanicConductance, SpinGalvanicConductance_diag, OrbitalGalvanicConductance, OrbitalGalvanicConductance_diag
from calculator import SpinTorque, SpintorqueGalvanicConductance
from calculator import IntrinsicNPHE_BCP, SecondOrderAHC_BCP_FermiSea, SecondOrderAHC_BCP_FermiSurf
from calculator import EnergyOnPath, SpinExpectationOnPath, OrbitalExpectationOnPath, OrbitalModernExpectationOnPath, SurfaceBulkSpectraOnPath

class BZGridTask:

    def __init__(self,model:System,
                 taskFunc,
                 args:dict,
                 BZGrid:np.ndarray|tuple|list,
                 BZSymm=False,
                 mode="run",
                 ncores=1) -> None:
        '''
        Conduct tasks that calculate quantity on the regular grids of Brillouin zone, using mixed FFT algorithm
        
        Parameters
        ----------
        - model : System
            System object that have mp_grid
        - taskFunc : function
            the function used to do the task (at FFT grid)
        - args : dict
            extra args passing to the task Func
        - BZGrid : np.ndarray(3) | tuple(3) | list(3), int
            size of BZ sampling
        - mode : "run" | "show"
            "run" to run the tasks directly, ortherwise, only show the grid info.
        - ncores : int
            parallel number
        '''
        FastGrid=np.zeros(3,dtype=int)
        for icomp, comp in enumerate(model.mp_grid):
            if comp>1: 
                FastGrid[icomp]=comp+1
            elif comp==1:
                FastGrid[icomp]=1
            else:
                raise RuntimeError(f"FFT grid is zero {FastGrid=}")

        for icomp in range(3):
            for extra_size in range(BZGrid[icomp]-FastGrid[icomp]+1):
                if BZGrid[icomp]%FastGrid[icomp]==0: break
                FastGrid[icomp]+=1

        self.QGrid=np.array([max([x,1]) for x in np.array(BZGrid)//FastGrid])

        print(f"Real-space grid: {model.mp_grid}")
        print(f"Intepolation BZ grid: {BZGrid}")
        print(f"FFT grid: {FastGrid}")
        print(f"Phase grid: {self.QGrid}")
        showElapsedTime()

        if mode!="run": 
            print(f"{mode=} the tasks are stoped. set mode = 'run' to run the tasks")
            exit()
    
        tasks=[]
        for iq1 in range(self.QGrid[0]):
            for iq2 in range(self.QGrid[1]):
                for iq3 in range(self.QGrid[2]):
                    q=np.array([iq1,iq2,iq3])
                    qlen=self.distGamma(model,q/BZGrid)
                    newTask={
                    'iq':q,
                    'q':q/BZGrid,
                    'qlen':qlen,
                    'phase':np.array([np.exp(2j*np.pi*((q/BZGrid) @ Rvec)) for Rvec in model.iRvec]),
                    'model':model,
                    'FFT grid':tuple(FastGrid),
                    'verbose':False, # quiet by default
                    }
                    newTask.update(args)
                    if not BZSymm: 
                        tasks.append(newTask)
                        continue
                    for itask, oldTask in enumerate(tasks):
                        if abs(oldTask['qlen'] - qlen) < SYMPREC:
                            if self.equiv(model,oldTask['q'],newTask['q']):
                                break
                            else:
                                tasks.insert(itask,newTask)
                                break
                        elif oldTask['qlen']>qlen+SYMPREC:
                            tasks.insert(itask,newTask)
                            break
                    else:
                        tasks.append(newTask)

        showElapsedTime(extraInfo=f"find {len(tasks)} irreducible q, {len(tasks)/np.prod(self.QGrid)*100:.3f}% the whole BZ")
    
        #pool=Pool(processes=ncores)
        self.results=[]
        if ncores==1: # serial run
            for task in tasks:
                result=taskFunc(task)
                self.results.append(result)
                showElapsedTime(offset=False,extraInfo=f"{len(self.results):3d}/{len(tasks):3d} finished")
        else: # parallel run
            for itask in range(0,len(tasks),ncores): tasks[itask]["verbose"]=2 # output one information per batch
            with ThreadPoolExecutor(max_workers=ncores) as executor:
                futures = [executor.submit(taskFunc, task) for task in tasks]
                for future in as_completed(futures):
                    self.results.append(future.result())
        showElapsedTime(extraInfo=f"integration finished")

    def distGamma(self, model:System, q:np.ndarray):
        '''
        Find the shortest distance to the Gamma point

        Parameters
        ----------
        - model : System
            System object that have reciLattice
        - q : np.ndarray(3)
            fractional coordinates of k-point
        '''
        shift_corners = np.arange(-2, 3)
        corners = np.array([[x, y, z] for x in shift_corners for y in shift_corners for z in shift_corners])
        return np.linalg.norm(model.reciLattice @ ((q % 1)[:, None] - corners[:,None,:]), axis=1).min()
    
    def equiv(self, model:System, q1:np.ndarray, q2:np.ndarray):
        '''
        Check the equivalance of to k-points

        Parameters
        ----------
        - model : System
            System object that have reciLattice
        - q1 : np.ndarray(3)
            fractional coordinates of the 1st k-point
        - q2 : np.ndarray(3)
            fractional coordinates of the other k-point
        '''
        q_diff = model.pointGroup.star(q1)[:,None,:] - model.pointGroup.star(q2)[None,:,:]
        if np.linalg.norm((q_diff - np.round(q_diff)), axis=2).min() < SYMPREC: return True
        return False

class BZLineTask:
    def __init__(self, model: System,
                 taskFunc,
                 args: dict=dict(),
                 KPOINTS="./KPOINTS",
                 mode="run",
                 ncores=1,
                 ) -> None:
        '''
        Conduct parallel tasks along a 1D k-path defined in a KPOINTS file.
        Splits the k-path into chunks and distributes them across cores to minimize IPC overhead.
        
        Parameters
        ----------
        - model : System
            System object containing the Hamiltonian data.
        - taskFunc : function
            The function used to do the task on a chunk of the k-path.
        - args : dict
            Extra args passing to the taskFunc.
        - KPOINTS : str
            Path to the KPOINTS file.
        - mode : "run" | "show"
            "run" to run the tasks directly, otherwise, only show the partition info.
        - ncores : int
            Number of parallel processes.
        '''
        self.KPOINTS = KPOINTS
        self.ncores = ncores
        
        # 1. Read the KPOINTS file
        with open(self.KPOINTS, "r") as fin:
            data = fin.read().strip().split("\n")
        self.nkpt = int(data[1].split()[0])
        self.kpath = np.loadtxt(data[3:3+self.nkpt])[:, :3]
        
        print(f"Read {self.nkpt} k-points from {self.KPOINTS}")
        showElapsedTime()
            
        # 2. Partition the k-path based on available cores
        # If nkpt is not perfectly divisible, array_split handles it cleanly
        n_chunks = min(self.ncores, self.nkpt)
        indices = np.arange(self.nkpt)
        chunks = np.array_split(indices, n_chunks)
        
        tasks = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0: continue
            newTask = {
                'chunk_id': i,
                'indices': chunk,
                'kpath': self.kpath[chunk],
                'model': model,
                'verbose': False, # quiet by default
            }
            newTask.update(args)
            tasks.append(newTask)
            
        print(f"Split k-path into {len(tasks)} chunks for {self.ncores} cores using thread backend.")
        if mode != "run":
            print(f"{mode=} the tasks are stopped. Set mode = 'run' to execute.")
            exit()
        
        # 3. Execute parallel mapping
        self.results = []
        
        if ncores == 1:
            for task in tasks:
                task['verbose']=2 # output all information when serial run
                result = taskFunc(task)
                self.results.append(result)
                showElapsedTime(offset=False, extraInfo=f"{len(self.results):3d}/{len(tasks):3d} finished")
        else:
            for itask in range(0,len(tasks),ncores): tasks[itask]["verbose"]=2 # output one information per batch
            with ThreadPoolExecutor(max_workers=ncores) as executor:
                futures = [executor.submit(taskFunc, task) for task in tasks]
                for future in as_completed(futures):
                    self.results.append(future.result())
        
        showElapsedTime(extraInfo="1D K-Path integration finished")

def solveEnergy(param):
    calc=Energy(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],
                skip_low=param.get('skip low',False),
                #verbose=param['verbose']
                )
    return {"energies":calc.energies,"iq":param["iq"]}

def solvePDOS(param):
    calc=PDOS(energyAxis=param['fermi'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,model=param['model'],)
    return {"energies":calc.energies,"PDOS":calc.pdos,"iq":param["iq"]}

def solveAHC(param):
    calc=AnormalousHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   currentProjectionList    =param['projected shells for applying current'],
                                   fermiList=param['fermi'],finiteTemperatureOccupations=param.get('finite temperature occupations', False),kBT=param.get('kBT', 0.026),
                                   verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            "projected Berry curv":calc.projected_BerryCurv,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveAHC_diag(param):
    calc=AnormalousHallConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected Berry curv":calc.projected_BerryCurv_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSHC(param):
    calc=SpinHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             spincurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "spin Berry curv":calc.spinBerryCurv,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSHC_diag(param):
    calc=SpinHallConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             spincurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "spin Berry curv":calc.spinvelocityVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOHC(param):
    calc=OrbitalHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             orbitalcurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "orbital Berry curv":calc.orbitalBerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveStGC(param):
    calc=SpintorqueGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinTorqueProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList  =param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveAHC_LG_DM(param):
    calc=AnormalousHallConductance_LG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   match_resonator=param.get('match resonator', False),
                                   )
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def _solve2ndAHC(param, calculatorClass):
    calc = calculatorClass(model=param['model'], kgrid=param['FFT grid'], phaseshift=param['phase'], fermiList=param['fermi'], verbose=param.get('verbose', 0),
            eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026), degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),)
    result = {"energies": calc.energies,
              "BCP": calc.G_K, "response": calc.response,
              "sigma": calc.sigma, "iq": param["iq"]}
    if hasattr(calc, "G_Dipole_K"):
        result["BCP dipole"] = calc.G_Dipole_K
    return result

def solve2ndAHC_BCP_FermiSea(param):
    return _solve2ndAHC(param, SecondOrderAHC_BCP_FermiSea)


def solve2ndAHC_BCP_FermiSurf(param):
    return _solve2ndAHC(param, SecondOrderAHC_BCP_FermiSurf)

def solveINPHE_BCP(param):
    calc = IntrinsicNPHE_BCP(model=param['model'], kgrid=param['FFT grid'], phaseshift=param['phase'], fermiList=param['fermi'],
            eta=param.get('eta', 0.0), kBT=param.get('kBT', 0.026), degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
            mag_type=param.get('mag type', 'total'), gS=param.get('gS', 2.0), verbose=param.get('verbose', 0))
    return{"energies": calc.energies,
           "response": calc.response,
           "sigma": calc.Upsilon,
           "iq": param["iq"],
    }

def solveBPVE_VG(param):
    calc=BulkPhotovoltaicConductance_VG(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_VG_DM(param):
    calc=BulkPhotovoltaicConductance_VG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'], verbose=param['verbose'],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_Shift(param):
    calc=BulkPhotovoltaicConductance_LG_Shift(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_Inject(param):
    calc=BulkPhotovoltaicConductance_LG_Injection(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_DM(param):
    calc=BulkPhotovoltaicConductance_LG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_TDSE(param):
    calc = BulkPhotovoltaicConductance_TDSE(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                    fermiList=param['fermi'],omegaList=param['omega'],
                                    include_offdiag=param.get('include offdiag', True),verbose=param["verbose"],
                                    )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.sigma_TDSE, 
            "iq": param["iq"]}

def solveBPVE_TDSE_LG(param):
    calc = BulkPhotovoltaicConductance_TDSE_LG_Naive(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],verbose=param["verbose"],
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.sigma_TDSE, 
            "iq": param["iq"]}

def solveIFE_VG(param):
    calc = InverseFaradayEffect_VG(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_VG_DM(param):
    calc = InverseFaradayEffect_VG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_LG(param):
    calc = InverseFaradayEffect_LG(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_LG_DM(param):
    calc = InverseFaradayEffect_LG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_TDSE(param):
    calc = InverseFaradayEffect_TDSE(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   include_offdiag=param.get('include offdiag', True),
                                   E0=param.get('E0', 0.001),
                                   n_steps_per_cycle=param.get('n steps per cycle', 100),verbose=param["verbose"],
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.chi_TDSE, "energies":calc.energies,
            "iq": param["iq"]}

def solveIFE_BlochAcceleration_TDSE(param):
    calc = InverseFaradayEffect_BlochAcceleration_TDSE(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   include_offdiag=param.get('include offdiag', True),
                                   E0=param.get('E0', 0.001),
                                   n_steps_per_cycle=param.get('n steps per cycle', 100),verbose=param["verbose"],
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.chi_TDSE,"energies":calc.energies,
            "iq": param["iq"]}

def solveIFTorque_TDSE(param):
    torqueProjectionList = param.get('projected shells for detecting torque',
                                     param.get('projected shells for detecting spin accumulation', []))
    calc = InverseFaradayTorque_TDSE(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     include_offdiag=param.get('include offdiag', True),
                                     E0=param.get('E0', 0.001),
                                     n_steps_per_cycle=param.get('n steps per cycle', 100),verbose=param["verbose"],
                                    )
    return {"projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,"energies":calc.energies,
            "iq": param["iq"]}

def solveIFTorque_VG_DM(param):
    torqueProjectionList = param.get('projected shells for detecting torque',
                                     param.get('projected shells for detecting spin accumulation', []))
    calc = InverseFaradayTorque_VG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     responseOmegaIndexList=param.get('record frequency ids', []),
                                     responseFermiIndexList=param.get('record fermi ids', None),
                                     finiteTemperatureOccupations=param.get('finite temperature occupations', False),
                                     kBT=param.get('kBT', 0.026),verbose=param["verbose"],
                                    )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,
            "iq": param["iq"]}

def solveIFTorque_LG_DM(param):
    torqueProjectionList = param.get('projected shells for detecting torque',
                                     param.get('projected shells for detecting spin accumulation', []))
    calc = InverseFaradayTorque_LG_DM(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     responseOmegaIndexList=param.get('record frequency ids', []),
                                     responseFermiIndexList=param.get('record fermi ids', None),
                                     finiteTemperatureOccupations=param.get('finite temperature occupations', False),
                                     kBT=param.get('kBT', 0.026),verbose=param["verbose"],
                                    )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,
            "iq": param["iq"]}

def solveSGC(param):
    calc=SpinGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSGC_diag(param):
    calc=SpinGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC(param):
    calc=OrbitalGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC_diag(param):
    calc=OrbitalGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveVelocity(param):
    calc=Velocity(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_K,"iq":param["iq"]}

def solveVelocity_diag(param):
    calc=Velocity_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_diag_K,"iq":param["iq"]}

def solveMagneticMoment(param):
    spin_eig, orbital_LC_eig, orbital_IC_eig=None, None, None
    skip_low=param.get('skip low',False)
    if param['spin proj'] and param['orbital proj']:
        calc=SpinOrbital(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=skip_low,verbose=param['verbose'])
        spin_eig       = np.diagonal(calc.spin_K,axis1=-2,axis2=-1)
        orbital_LC_eig = np.diagonal(calc.orbital_K,axis1=-2,axis2=-1)
        if hasattr(calc,"orbital_IC_K"): orbital_IC_eig = calc.orbital_IC_K
    elif param['spin proj']:
        calc=Spin(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=skip_low,verbose=param['verbose'])
        spin_eig       = np.diagonal(calc.spin_K,axis1=-2,axis2=-1)
    elif param['orbital proj']:
        calc=OrbitalModern_Full(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=skip_low,verbose=param['verbose'])
        orbital_LC_eig = np.diagonal(calc.orbital_K,axis1=-2,axis2=-1)
        if hasattr(calc,"orbital_IC_K"): orbital_IC_eig = calc.orbital_IC_K
    else:
        calc=Energy(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=skip_low,verbose=param['verbose'])
    return {
        'energies': calc.energies,
        'vectors':calc.vectors,
        'spin': spin_eig, 'orbital LC': orbital_LC_eig, 'orbital IC': orbital_IC_eig,
        "iq":param["iq"]
    }

#+--------------------------+
#| factory for BZLine tasks |
#+--------------------------+
def solveSurfaceSpectraTask(param):
    calc=SurfaceBulkSpectraOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'], 
                                  surfaceDir=param['surfaceDir'], fermiAxis=param['fermiAxis'], eta=param['eta'],
                                  n_iter=param['n_iter'],)
    return {
        'indices': param['indices'],
        'A_bulk': calc.A_bulk,
        'A_surf_L': calc.A_surf_L,
        'A_surf_R': calc.A_surf_R,
    }

def solveBandStructure(param):
    spin_eig, orbital_eig=None, None
    degen=param.get('degen', 1e-6)
    if param['spin proj'] and param['orbital proj']:
        calc=SpinExpectationOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'], degen=degen)
        spin_eig = calc.spin_eig.real
        calc=OrbitalExpectationOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'], degen=degen)
        orbital_eig = calc.orbital_eig.real
    elif param['spin proj']:
        calc=SpinExpectationOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'], degen=degen)
        spin_eig = calc.spin_eig.real
    elif param['orbital proj']:
        calc=OrbitalModernExpectationOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'], degen=degen)
        orbital_eig = calc.orbital_eig.real
    else:
        calc=EnergyOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'])
    return {
        'indices': param['indices'],
        'energies': calc.energies,
        'spin_eig': spin_eig,
        'orbital_eig': orbital_eig,
    }
