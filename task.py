# Module to implement the parallel tasks
# 
from multiprocessing import Pool
import numpy as np
from auxiliary import showElapsedTime, SYMPREC
from system import System

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
    
        irreducibleTask=[]
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
                    }
                    newTask.update(args)
                    if not BZSymm: 
                        irreducibleTask.append(newTask)
                        continue
                    for itask, oldTask in enumerate(irreducibleTask):
                        if abs(oldTask['qlen'] - qlen) < SYMPREC:
                            if self.equiv(model,oldTask['q'],newTask['q']):
                                break
                            else:
                                irreducibleTask.insert(itask,newTask)
                                break
                        elif oldTask['qlen']>qlen+SYMPREC:
                            irreducibleTask.insert(itask,newTask)
                            break
                    else:
                        irreducibleTask.append(newTask)

        showElapsedTime(extraInfo=f"find {len(irreducibleTask)} irreducible q, {len(irreducibleTask)/np.prod(self.QGrid)*100:.3f}% the whole BZ")
    
        pool=Pool(processes=ncores)
        self.results=[]
        if ncores==1: # serial run
            for task in irreducibleTask:
                result=taskFunc(task)
                self.results.append(result)
                showElapsedTime(offset=False,extraInfo=f"{len(self.results):3d}/{len(irreducibleTask):3d} finished")
        else: # parallel run
            for result in pool.imap_unordered(taskFunc,irreducibleTask):
                self.results.append(result)
                if len(self.results) % ncores == 1: showElapsedTime(offset=False,extraInfo=f"{len(self.results):3d}/{len(irreducibleTask):3d} finished")
            pool.close()
            pool.join()
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
                 ncores=1) -> None:
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
        
        if mode != "run":
            print(f"{mode=} the tasks are stopped. Set mode = 'run' to execute.")
            exit()
            
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
                'verbose': 0, # quiet by default
            }
            newTask.update(args)
            tasks.append(newTask)
            
        print(f"Split k-path into {len(tasks)} chunks for {self.ncores} cores.")
        
        # 3. Execute parallel mapping
        pool = Pool(processes=ncores)
        self.results = []
        
        if ncores == 1:
            for task in tasks:
                task['verbose']=2 # output all information when serial run
                result = taskFunc(task)
                self.results.append(result)
                showElapsedTime(offset=False, extraInfo=f"{len(self.results):3d}/{len(tasks):3d} finished")
        else:
            for result in pool.imap_unordered(taskFunc, tasks):
                self.results.append(result)
                if len(self.results) % ncores == 1: showElapsedTime(offset=False, extraInfo=f"{len(self.results):3d}/{len(tasks):3d} finished")
            pool.close()
            pool.join()
        
        showElapsedTime(extraInfo="1D K-Path integration finished")