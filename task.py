# Module to implement the parallel tasks
# 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import contextmanager
import multiprocessing as mp
import os
import sys
import threading
from time import perf_counter
import numpy as np
from auxiliary import showElapsedTime, SYMPREC
from system import System
from calculator.collision import CollisionSpecification, PreparedCollision, collision_mode
from calculator import Velocity, Velocity_diag, Energy, SpinBerryCurv, PDOS, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath
from calculator import Spin, Orbital, OrbitalModern_Full, SpinOrbital
from calculator import AnormalousHallConductance, AnormalousHallConductance_diag, AnormalousHallConductance_LG_DM, BulkPhotovoltaicEffect_VG, BulkPhotovoltaicEffect_VG_DM, BulkPhotovoltaicEffect_LG_Shift, BulkPhotovoltaicEffect_LG_Injection, BulkPhotovoltaicEffect_LG_DM, BulkPhotovoltaicEffect_TDSE, BulkPhotovoltaicEffect_TDSE_LG_Naive
from calculator import InverseFaradayEffect_TDSE, InverseFaradayEffect_BlochAcceleration_TDSE, InverseFaradayEffect_VG, InverseFaradayEffect_VG_DM, InverseFaradayEffect_LG, InverseFaradayEffect_LG_DM
from calculator import InverseFaradayTorque_TDSE, InverseFaradayTorque_VG_DM, InverseFaradayTorque_LG_DM
from calculator import SpinHallConductance, SpinHallConductance_LG_DM, SpinHallConductance_diag, OrbitalHallConductance
from calculator import SpinGalvanicConductance, SpinGalvanicConductance_diag, OrbitalGalvanicConductance, OrbitalGalvanicConductance_diag
from calculator import (SpinGalvanicConductivity_LG_DM, SpinGalvanicConductivity_VG_DM,
                        OrbitalGalvanicConductivity_LG_DM, OrbitalGalvanicConductivity_VG_DM)
from calculator import (SpinTorque, SpintorqueGalvanicConductance,
                        SpintorqueGalvanicConductivity_LG_DM, SpintorqueGalvanicConductivity_VG_DM)
from calculator import IntrinsicNPHE_BCP, SecondOrderAHC_BCP_FermiSea, SecondOrderAHC_BCP_FermiSurf
from calculator import EnergyOnPath, SpinExpectationOnPath, OrbitalExpectationOnPath, OrbitalModernExpectationOnPath, SurfaceBulkSpectraOnPath
from calculator import AnormalousHallConductance_VG_DM, Opticalconductivity_LG_DM, Opticalconductivity_VG_DM, Opticalconductivity_TDSE
from calculator import Opticalconductivity_VG_DM_Phonon

_bz_worker_state = None
_bz_worker_thread_limit = None

def _initialize_bz_worker(model, task_func, common_args, worker_threads):
    """Install the model once per process, inherited copy-on-write with fork."""
    from threadpoolctl import threadpool_limits
    global _bz_worker_state, _bz_worker_thread_limit
    _bz_worker_state = (model, task_func, common_args)
    _bz_worker_thread_limit = threadpool_limits(limits=worker_threads)

def _run_bz_task(job, model, task_func, common_args):
    params = dict(common_args)
    params.update(job)
    params["model"] = model
    return task_func(params)

def _run_bz_worker(job):
    return _run_bz_task(job, *_bz_worker_state)

@contextmanager
def _bz_numerical_threads(number):
    """Limit loaded libraries and libraries imported later by spawned workers."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ImportError("Parallel BZGridTask requires threadpoolctl: python -m pip install threadpoolctl") from exc
    names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(number)
        with threadpool_limits(limits=number):
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

class BZGridTask:

    def __init__(self,model:System,
                 taskFunc,
                 args:dict,
                 BZGrid:np.ndarray|tuple|list,
                 BZSymm=False,
                 mode="run",
                 ncores:int=1,
                 parallelBackend="process",
                 workerThreads:int=1,
                 processStartMethod=None) -> None:
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
            size of BZ sampling.
            Every generated task passes its own ``q / BZGrid`` as ``kOffset`` to
            the task function; this is the sole representation of the FFT-mesh
            offset.
        - mode : "run" | "show"
            "run" to run the tasks directly, ortherwise, only show the grid info.
        - ncores : int
            maximum number of FFT-submesh workers (capped by the number of tasks)
            Mesh-coupled EPC uses one complete BZGrid task instead of independent
            submeshes. Local CRTA/SERTA retain the ordinary worker decomposition.
        - parallelBackend : "process" | "thread"
            processes avoid the Python GIL; threads retain the shared-memory legacy backend
        - workerThreads : int
            BLAS/OpenMP threads per parallel worker; serial execution is unchanged
        - processStartMethod : None | "fork" | "spawn" | "forkserver"
            None chooses fork on single-threaded Linux callers, spawn otherwise.
            Spawn requires an importable taskFunc and a guarded __main__ entry point.
            Fork shares model arrays copy-on-write; tasks must treat the model as read-only.
        '''
        if parallelBackend not in ("process", "thread"):
            raise ValueError("parallelBackend must be 'process' or 'thread'")
        if processStartMethod is not None and processStartMethod not in mp.get_all_start_methods():
            raise ValueError(f"Unavailable processStartMethod: {processStartMethod}")
        ncores, workerThreads = int(ncores), int(workerThreads)
        BZGrid = np.asarray(BZGrid).astype(int)
        # Markov populations and covariant mesh derivatives are coupled over
        # the complete electronic mesh. Independent offset solves are invalid.
        has_collision = "collisionSpecification" in args or "preparedCollision" in args
        if has_collision and collision_mode(args.get('collisionSpecification'), args.get('preparedCollision')) == "epc":
            FastGrid = BZGrid.copy()
            if ncores > 1:
                print("Mesh-coupled EPC: using one complete BZ mesh; independent offset workers are disabled")
        else:
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
        print(f"Offset grid: {self.QGrid}")
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
                    'kOffset':q/BZGrid,
                    'qlen':qlen,
                    'FFT grid':tuple(FastGrid),
                    'verbose':False, # quiet by default
                    }
                    if not BZSymm: 
                        tasks.append(newTask)
                        continue
                    for itask, oldTask in enumerate(tasks):
                        if abs(oldTask['qlen'] - qlen) < SYMPREC:
                            if self.equiv(model,oldTask['kOffset'],newTask['kOffset']):
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

        # k-point parallel should build the collision operator during the parallel,
        # rather than prepared before the parallel, because the kOffset should be
        # matched between the electronic parts and EPC parts.
        if args.get('preparedCollision') is not None:
            if len(tasks) != 1:
                raise ValueError("preparedCollision belongs to one exact mesh; use a direct calculator or one BZ task")
            # Reuse is valid for any mode when the exact prepared mesh is used.
            prepared = args['preparedCollision']
            if (model is not prepared.inputs.model
                    or tuple(FastGrid) != prepared.inputs.kgrid
                    or not np.array_equal(tasks[0]['kOffset'], prepared.inputs.kOffset)):
                raise ValueError("preparedCollision does not match the BZ task model, mesh or offset")

        electronic = args.get('electronic')
        states = electronic.states if electronic is not None else args.get('states')
        if states is not None:
            if (len(tasks) != 1 or states.model is not model
                    or tuple(FastGrid) != states.kgrid
                    or not np.array_equal(tasks[0]['kOffset'], states.kOffset)):
                raise ValueError("Prepared electronic data requires one BZ task with the same model, mesh and offset")

        self.nworkers = min(ncores, len(tasks))
        self.parallelBackend = "serial" if self.nworkers == 1 else parallelBackend
        self.processStartMethod = None
        self.workerThreads = workerThreads
        self.results = [None] * len(tasks)
        start = perf_counter()
        if self.nworkers == 1:
            for index, task in enumerate(tasks):
                self.results[index] = _run_bz_task(task, model, taskFunc, args)
                showElapsedTime(offset=False, extraInfo=f"{index+1:3d}/{len(tasks):3d} finished")
        else:
            if parallelBackend == "process":
                start_method = processStartMethod
                if start_method is None:
                    start_method = "fork" if sys.platform.startswith("linux") and threading.active_count() == 1 else "spawn"
                if start_method == "fork" and threading.active_count() != 1:
                    raise RuntimeError("Cannot fork BZ workers from a multithreaded caller; use processStartMethod='spawn'.")
                self.processStartMethod = start_method
                # Windows ProcessPoolExecutor supports at most 61 worker processes.
                if sys.platform == "win32":
                    self.nworkers = min(self.nworkers, 61)
            for itask in range(0, len(tasks), self.nworkers):
                tasks[itask]["verbose"] = 2
            print(f"BZ parallel backend: {parallelBackend}, workers: {self.nworkers}/{ncores} requested, "
                  f"numerical threads/worker: {workerThreads}, start method: {self.processStartMethod}", flush=True)
            with _bz_numerical_threads(workerThreads):
                if parallelBackend == "process":
                    executor = ProcessPoolExecutor(max_workers=self.nworkers, mp_context=mp.get_context(start_method),
                        initializer=_initialize_bz_worker, initargs=(model, taskFunc, args, workerThreads))
                    submit = lambda job: executor.submit(_run_bz_worker, job)
                else:
                    executor = ThreadPoolExecutor(max_workers=self.nworkers)
                    submit = lambda job: executor.submit(_run_bz_task, job, model, taskFunc, args)
                pending = {}
                next_index, completed, last_report = 0, 0, 0
                report_every = max(1, len(tasks)//10)
                try:
                    # Bound queued work and retain task order for reproducible reductions.
                    while next_index < len(tasks) or pending:
                        while next_index < len(tasks) and len(pending) < 2*self.nworkers:
                            pending[submit(tasks[next_index])] = next_index
                            next_index += 1
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            try:
                                self.results[index] = future.result()
                            except Exception as exc:
                                raise RuntimeError(f"BZ task iq={tuple(tasks[index]['iq'])} failed") from exc
                            completed += 1
                        if completed == len(tasks) or completed-last_report >= report_every:
                            print(f"BZ tasks: {completed}/{len(tasks)} finished in {perf_counter()-start:.2f} s", flush=True)
                            last_report = completed
                finally:
                    for future in pending:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
        self.elapsed_seconds = perf_counter() - start
        print(f"BZ integration finished in {self.elapsed_seconds:.3f} s ({self.parallelBackend})", flush=True)

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
    calc=Energy(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                skip_low=param.get('skip low',False),
                #verbose=param['verbose']
                )
    return {"energies":calc.energies,"iq":param["iq"]}

def solvePDOS(param):
    calc=PDOS(energyAxis=param['fermi'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.026,model=param['model'],)
    return {"energies":calc.energies,"PDOS":calc.pdos,"iq":param["iq"]}

def solveAHC(param):
    calc=AnormalousHallConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=False,
                                   currentProjectionList    =param['projected shells for applying current'],
                                   fermiList=param['fermi'],finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),kBT=param.get('kBT', 0.026),
                                   verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            "projected Berry curv":calc.projected_BerryCurv,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveAHC_diag(param):
    calc=AnormalousHallConductance_diag(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected Berry curv":calc.projected_BerryCurv_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSHC(param):
    calc=SpinHallConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=True,
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
    calc=SpinHallConductance_diag(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
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
    calc=OrbitalHallConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             orbitalcurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "orbital Berry curv":calc.orbitalBerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveStGC(param):
    calc=SpintorqueGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinTorqueProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList  =param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveAHC_LG_DM(param):
    calc=AnormalousHallConductance_LG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
                                fermiList=param['fermi'],
                                currentProjectionList    =param['projected shells for applying current'],
                                responseFermiIndexList=param.get('record fermi ids', None),
                                finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                                fullCovariantDerivative=param.get('full covariant derivative', False),
                                verbose=param.get('verbose', 0),
                                   )
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "projected response":calc.projected_response,
            "projected sigma channels":calc.projected_sigma_channels,
            "projected response channels":calc.projected_response_channels,
            "response fermi ids":calc.responseFermiIndexList,
            "iq":param["iq"]}

def _solveLinearConductivity(param, calculatorClass):
    """BZGridTask adapter preserving complex optical tensors and record axes."""
    options = dict(model=param['model'], kgrid=param['FFT grid'], kOffset=param['kOffset'],
                   eta=param.get('eta', .06582), kBT=param.get('kBT', .026),
                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                   degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
                   fermiList=param['fermi'], omegaList=param.get('omega', np.array([0.])),
                   currentProjectionList=param.get('projected shells for applying current', []),
                   fullCovariantDerivative=param.get('full covariant derivative', True),
                   finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', True),
                   verbose=param.get('verbose', 0))
    if calculatorClass is Opticalconductivity_TDSE:
        for key in ('E0', 'n_steps_per_cycle', 'n_cycles_measure', 'transient_lifetimes', 'max_phase_step', 'recordCurrent',
                    'phononDynamics', 'phononTemperature', 'phononStatistics',
                    'nPhononTrajectories', 'phononSeed', 'phononCycles',
                    'phononDamping', 'phononBathCycles', 'phononPhasePoints'):
            if key in param:
                options[key] = param[key]
    elif calculatorClass is Opticalconductivity_VG_DM_Phonon:
        for key in ('phononTemperature', 'phononStatistics', 'nPhononTrajectories',
                    'phononSeed', 'phononPhasePoints', 'phononDamping', 'phononBathCycles',
                    'solverTolerance', 'solverMaxiter', 'solverRestart'):
            if key in param:
                options[key] = param[key]
    else:
        options.update(responseFermiIndexList=param.get('record fermi ids'),
                       responseOmegaIndexList=param.get('record frequency ids'))
    calc = calculatorClass(**options)
    result = {'energies': calc.energies, 'projected sigma': calc.projected_sigma,
              'sigma': calc.sigma, 'iq': param['iq']}
    for key, attr in (('projected response', 'projected_response'),
                      ('projected sigma channels', 'projected_sigma_channels'),
                      ('projected response channels', 'projected_response_channels'),
                      ('sigma channels', 'sigma_channels'),
                      ('response fermi ids', 'responseFermiIndexList'),
                      ('response frequency ids', 'responseOmegaIndexList'),
                      ('sum rule residual', 'sum_rule_residual'),
                      ('time traces', 'time_traces'), ('integration info', 'integration_info'),
                      ('sigma samples', 'sigma_samples'), ('sigma stderr real', 'sigma_stderr_real'),
                      ('sigma stderr imag', 'sigma_stderr_imag'), ('phonon dynamics info', 'phonon_dynamics_info'),
                      ('phonon initial positions', 'phonon_initial_positions'),
                      ('phonon initial velocities', 'phonon_initial_velocities'),
                      ('phonon trajectory cosine', 'phonon_trajectory_cosine'),
                      ('phonon trajectory sine', 'phonon_trajectory_sine'),
                      ('phonon energies', 'phonon_energies'), ('solver info', 'solver_info'),
                      ('conductivity harmonics', 'conductivity_harmonics'),
                      ('phonon harmonics', 'phonon_harmonics'), ('phonon frequency', 'phonon_frequency')):
        if hasattr(calc, attr):
            result[key] = getattr(calc, attr)
    return result

def solveAHC_VG_DM(param):
    return _solveLinearConductivity(param, AnormalousHallConductance_VG_DM)

def solveOpticalconductivity_LG_DM(param):
    return _solveLinearConductivity(param, Opticalconductivity_LG_DM)

def solveOpticalconductivity_VG_DM(param):
    return _solveLinearConductivity(param, Opticalconductivity_VG_DM)

def solveOpticalconductivity_TDSE(param):
    return _solveLinearConductivity(param, Opticalconductivity_TDSE)

def solveOpticalconductivity_VG_DM_Phonon(param):
    return _solveLinearConductivity(param, Opticalconductivity_VG_DM_Phonon)

def solveSHC_LG_DM(param):
    calc=SpinHallConductance_LG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                              eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                              **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                              degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
                              fermiList=param['fermi'],
                              currentProjectionList=param['projected shells for applying current'],
                              spincurrentProjectionList=param['projected shells for detecting spincurrent'],
                              responseFermiIndexList=param.get('record fermi ids', None),
                              finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                              fullCovariantDerivative=param.get('full covariant derivative', False),
                              verbose=param.get('verbose', 0),)
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "projected response":calc.projected_response,
            "projected sigma channels":calc.projected_sigma_channels,
            "projected response channels":calc.projected_response_channels,
            "response fermi ids":calc.responseFermiIndexList,
            "iq":param["iq"]}

def _solve2ndAHC(param, calculatorClass):
    calc = calculatorClass(model=param['model'], kgrid=param['FFT grid'], kOffset=param['kOffset'], fermiList=param['fermi'], verbose=param.get('verbose', 0),
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
    calc = IntrinsicNPHE_BCP(model=param['model'], kgrid=param['FFT grid'], kOffset=param['kOffset'], fermiList=param['fermi'],
            eta=param.get('eta', 0.0), kBT=param.get('kBT', 0.026), degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
            mag_type=param.get('mag type', 'total'), gS=param.get('gS', 2.0), verbose=param.get('verbose', 0))
    return{"energies": calc.energies,
           "response": calc.response,
           "sigma": calc.Upsilon,
           "iq": param["iq"],
    }

def solveBPVE_VG(param):
    calc=BulkPhotovoltaicEffect_VG(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_VG_DM(param):
    calc=BulkPhotovoltaicEffect_VG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                   eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                   degen=param.get('degen', 1e-6), skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),
                                   finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                                   fullCovariantDerivative=param.get('full covariant derivative', False),
                                   currentProjectionList=param['projected shells for applying current'], verbose=param['verbose'],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "projected response channels":calc.projected_response_channels,
            "projected sigma channels":calc.projected_sigma_channels,
            "sigma channels":calc.sigma_VG_DM_channels,
            "iq":param["iq"]}

def solveBPVE_LG_Shift(param):
    calc=BulkPhotovoltaicEffect_LG_Shift(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_Inject(param):
    calc=BulkPhotovoltaicEffect_LG_Injection(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_DM(param):
    calc=BulkPhotovoltaicEffect_LG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                   eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                   degen=param.get('degen', 1e-6), skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),
                                   finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                                   fullCovariantDerivative=param.get('full covariant derivative', False),
                                   currentProjectionList=param['projected shells for applying current'],verbose=param["verbose"],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "projected response channels":calc.projected_response_channels,
            "projected sigma channels":calc.projected_sigma_channels,
            "sigma channels":calc.sigma_LG_DM_channels,
            "iq":param["iq"]}

def solveBPVE_TDSE(param):
    calc = BulkPhotovoltaicEffect_TDSE(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,skip_low=False,
                                    fermiList=param['fermi'],omegaList=param['omega'],
                                    include_offdiag=param.get('include offdiag', True),verbose=param["verbose"],
                                    )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.sigma_TDSE, 
            "iq": param["iq"]}

def solveBPVE_TDSE_LG(param):
    calc = BulkPhotovoltaicEffect_TDSE_LG_Naive(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],verbose=param["verbose"],
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.sigma_TDSE, 
            "iq": param["iq"]}

def solveIFE_VG(param):
    calc = InverseFaradayEffect_VG(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_VG_DM(param):
    calc = InverseFaradayEffect_VG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                   eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                   degen=param.get('degen', 1e-6), skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   fullCovariantDerivative=param.get('full covariant derivative', False),
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "projected response channels": calc.projected_response_channels,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFE_channels,
            "iq": param["iq"]}

def solveIFE_LG(param):
    calc = InverseFaradayEffect_LG(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "iq": param["iq"]}

def solveIFE_LG_DM(param):
    calc = InverseFaradayEffect_LG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                   eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                   degen=param.get('degen', 1e-6), skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   fullCovariantDerivative=param.get('full covariant derivative', False),
                                   responseOmegaIndexList=param.get('record frequency ids', []),
                                   responseFermiIndexList=param.get('record fermi ids', None),verbose=param["verbose"],
                                   )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_sigma, "sigma": calc.chi_IFE,
            "projected response channels": calc.projected_response_channels,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFE_channels,
            "iq": param["iq"]}

def solveIFE_TDSE(param):
    calc = InverseFaradayEffect_TDSE(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],
                                   eta=param.get('eta', 0.06582), kBT=param.get('kBT', 0.026),
                                   degen=param.get('degen', 1e-6), skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   ife_type=param['mag type'],
                                   fullCovariantDerivative=param.get('full covariant derivative', False),
                                   include_offdiag=param.get('include offdiag', True),
                                   E0=param.get('E0', 0.001),
                                   n_steps_per_cycle=param.get('n steps per cycle', 100),verbose=param["verbose"],
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.chi_TDSE, "energies":calc.energies,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFE_channels,
            "iq": param["iq"]}

def solveIFE_BlochAcceleration_TDSE(param):
    calc = InverseFaradayEffect_BlochAcceleration_TDSE(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,skip_low=False,
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
    calc = InverseFaradayTorque_TDSE(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     kBT=param.get('kBT', 0.026),
                                     degen=param.get('degen', 1e-6),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     fullCovariantDerivative=param.get('full covariant derivative', False),
                                     exchangeVelocity=param.get('exchange velocity', True),
                                     include_offdiag=param.get('include offdiag', True),
                                     E0=param.get('E0', 0.001),
                                     n_steps_per_cycle=param.get('n steps per cycle', 100),verbose=param["verbose"],
                                    )
    return {"projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,"energies":calc.energies,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFT_channels,
            "iq": param["iq"]}

def solveIFTorque_VG_DM(param):
    torqueProjectionList = param.get('projected shells for detecting torque',
                                     param.get('projected shells for detecting spin accumulation', []))
    calc = InverseFaradayTorque_VG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                     degen=param.get('degen', 1e-6),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     responseOmegaIndexList=param.get('record frequency ids', []),
                                     responseFermiIndexList=param.get('record fermi ids', None),
                                     finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                                     fullCovariantDerivative=param.get('full covariant derivative', False),
                                     exchangeVelocity=param.get('exchange velocity', True),
                                     kBT=param.get('kBT', 0.026),verbose=param["verbose"],
                                    )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,
            "projected response channels": calc.projected_response_channels,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFT_channels,
            "iq": param["iq"]}

def solveIFTorque_LG_DM(param):
    torqueProjectionList = param.get('projected shells for detecting torque',
                                     param.get('projected shells for detecting spin accumulation', []))
    calc = InverseFaradayTorque_LG_DM(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=False,
                                     eta=param.get('eta', 0.06582),
                                     **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                                     degen=param.get('degen', 1e-6),
                                     fermiList=param['fermi'],omegaList=param['omega'],
                                     torqueProjectionList=torqueProjectionList,ife_type=param['mag type'],
                                     responseOmegaIndexList=param.get('record frequency ids', []),
                                     responseFermiIndexList=param.get('record fermi ids', None),
                                     finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', False),
                                     fullCovariantDerivative=param.get('full covariant derivative', False),
                                     kBT=param.get('kBT', 0.026),verbose=param["verbose"],
                                    )
    return {"projected response": calc.projected_response,"energies":calc.energies,
            "projected sigma": calc.projected_chi_IFT, "sigma": calc.chi_IFT,
            "projected response channels": calc.projected_response_channels,
            "projected sigma channels": calc.projected_sigma_channels,
            "sigma channels": calc.chi_IFT_channels,
            "iq": param["iq"]}

def solveSGC(param):
    calc=SpinGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSGC_diag(param):
    calc=SpinGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC(param):
    calc=OrbitalGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC_diag(param):
    calc=OrbitalGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def _solveGalvanicConductivity_DM(param, calculatorClass, projectionArgument,
                                  projectionKeys, extraOptions=None):
    """BZGridTask adapter for dc LG or VG density-matrix galvanic responses."""
    projectionList = []
    for key in projectionKeys:
        if key in param:
            projectionList = param[key]
            break
    options = dict(model=param['model'], kgrid=param['FFT grid'], kOffset=param['kOffset'],
                   eta=param.get('eta', .06582), kBT=param.get('kBT', .026),
                   **{key: param[key] for key in ("collisionSpecification", "preparedCollision", "epcModel", "electronic", "states") if key in param},
                   degen=param.get('degen', 1e-6), skip_low=param.get('skip low', False),
                   fermiList=param['fermi'],
                   currentProjectionList=param.get('projected shells for applying current', []),
                   responseFermiIndexList=param.get('record fermi ids'),
                   finiteTemperatureOccupations=param.get('finiteTemperatureOccupations', True),
                   fullCovariantDerivative=param.get('full covariant derivative', True),
                   verbose=param.get('verbose', 0))
    options[projectionArgument] = projectionList
    if extraOptions:
        options.update(extraOptions)
    calc = calculatorClass(**options)
    result = {"energies": calc.energies, "projected sigma": calc.projected_sigma,
              "sigma": calc.sigma, "projected response": calc.projected_response,
              "projected sigma channels": calc.projected_sigma_channels,
              "projected response channels": calc.projected_response_channels,
              "sigma channels": calc.sigma_channels,
              "response fermi ids": calc.responseFermiIndexList,
              "units": calc.galvanic_units, "iq": param["iq"]}
    if hasattr(calc, 'sum_rule_residual'):
        result['sum rule residual'] = calc.sum_rule_residual
        result['projected sum rule residual'] = calc.projected_sum_rule_residual
    return result

def solveSGC_LG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, SpinGalvanicConductivity_LG_DM, 'spinProjectionList',
        ('projected shells for detecting spin accumulation',)
    )

def solveSGC_VG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, SpinGalvanicConductivity_VG_DM, 'spinProjectionList',
        ('projected shells for detecting spin accumulation',)
    )

def solveOGC_LG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, OrbitalGalvanicConductivity_LG_DM, 'orbitalProjectionList',
        ('projected shells for detecting orbital accumulation',
         'projected shells for detecting spin accumulation')
    )

def solveOGC_VG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, OrbitalGalvanicConductivity_VG_DM, 'orbitalProjectionList',
        ('projected shells for detecting orbital accumulation',
         'projected shells for detecting spin accumulation')
    )

def solveStGC_LG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, SpintorqueGalvanicConductivity_LG_DM, 'spinTorqueProjectionList',
        ('projected shells for detecting torque',
         'projected shells for detecting spin accumulation')
    )

def solveStGC_VG_DM(param):
    return _solveGalvanicConductivity_DM(
        param, SpintorqueGalvanicConductivity_VG_DM, 'spinTorqueProjectionList',
        ('projected shells for detecting torque',
         'projected shells for detecting spin accumulation'),
        {'exchangeVelocity': param.get('exchange velocity', True)}
    )

def solveVelocity(param):
    calc=Velocity(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_K,"iq":param["iq"]}

def solveVelocity_diag(param):
    calc=Velocity_diag(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_diag_K,"iq":param["iq"]}

def solveMagneticMoment(param):
    spin_eig, orbital_LC_eig, orbital_IC_eig=None, None, None
    skip_low=param.get('skip low',False)
    if param['spin proj'] and param['orbital proj']:
        calc=SpinOrbital(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=skip_low,verbose=param['verbose'])
        spin_eig       = np.diagonal(calc.spin_K,axis1=-2,axis2=-1)
        orbital_LC_eig = np.diagonal(calc.orbital_K,axis1=-2,axis2=-1)
        if hasattr(calc,"orbital_IC_K"): orbital_IC_eig = calc.orbital_IC_K
    elif param['spin proj']:
        calc=Spin(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=skip_low,verbose=param['verbose'])
        spin_eig       = np.diagonal(calc.spin_K,axis1=-2,axis2=-1)
    elif param['orbital proj']:
        calc=OrbitalModern_Full(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=skip_low,verbose=param['verbose'])
        orbital_LC_eig = np.diagonal(calc.orbital_K,axis1=-2,axis2=-1)
        if hasattr(calc,"orbital_IC_K"): orbital_IC_eig = calc.orbital_IC_K
    else:
        calc=Energy(model=param['model'],kgrid=param['FFT grid'],kOffset=param['kOffset'],skip_low=skip_low,verbose=param['verbose'])

    # The diagonal of an atom projector in the eigenbasis is the weight of
    # that atom in each Bloch state.  Contract only the selected rows instead
    # of materializing the full projected band matrix.  For a non-orthogonal
    # basis this follows the generalized-eigenproblem convention
    # v^dagger P O v; for an orthogonal basis it reduces to |v|**2.
    if param['model'].overlap:
        overlap_vectors = np.einsum(
            "...ab,...bn->...an", 
            calc.O_K_AO, calc.vectors, optimize=True
        )
    else:
        overlap_vectors = calc.vectors
    atomic_weights = np.zeros(calc.energies.shape[:-1] + (len(param['model'].atomicOrbInfoList), calc.energies.shape[-1]), dtype=float,)
    for iatom, atom_info in enumerate(param['model'].atomicOrbInfoList):
        orbital_indices = [iorb for shell in atom_info['orbital index'] for iorb in shell]
        if not orbital_indices: continue
        atomic_weights[..., iatom, :] = np.sum(
            calc.vectors[..., orbital_indices, :].conj()  * overlap_vectors[..., orbital_indices, :], axis=-2).real

    return {
        'energies': calc.energies,
        'vectors':calc.vectors,
        'spin': spin_eig, 'orbital LC': orbital_LC_eig, 'orbital IC': orbital_IC_eig,
        'atomic weights': atomic_weights,
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
