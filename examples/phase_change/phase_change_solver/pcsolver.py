import numpy as np
import math

class SolidiPy:
  def __init__(self,grid,material,ics,bcs,times):  
    NI= self.NI=grid['xGrids'] 
    NJ= self.NJ=grid['yGrids']   
    NIJ= self.NIJ=max(self.NI,self.NJ)  
    NFMAX= self.NFMAX=10   
    NP= self.NP=10   
    NFX3= self.NFX3=self.NFMAX+1  

    self.ITLL=0 
    self.IPREF=1  
    self.JPREF=1   
    self.MODE=1   
    self.TIME=0  

    self.XL= grid['xLength']
    self.YL= grid['yLength']     
    L1= self.L1= grid['xGrids']       
    M1= self.M1= grid['yGrids']  
    self.NTIME= 1 
    self.NGRID= 1
    self.DARCYCONST= material['darcyConstant']
    self.RHOCON= material['density']
    self.TK= material['conductivity']
    self.CP= material['specificHeatCapacity']
    self.ALATENT= material['latentHeat']
    self.BETA= material['thermalExpansionCoeff']
    self.VISCOSITY= material['viscosity']
    self.RHOREF= material['referenceDensity']
    self.G=9.81  
    self.TINITIAL= ics['TINITIAL']
    self.TMELT=material['meltingTemp']
    self.THOT=bcs['THOT'] 
    self.ISTP= times['maxIter'] 
    self.DT=times['timeStep']
    self.TLAST=times['TimeLast']
    
    self.ERU=0.0001
    self.ERV=0.0001
    self.ERT=0.0001 
    self.ARELAX=0.2 # RELAXATION FOR ENTHALPY ONLY

    self.X=np.empty(NI, dtype=float)
    self.XU=np.empty(NI, dtype=float)
    self.XDIF=np.empty(NI, dtype=float)
    self.XCV=np.empty(NI, dtype=float)
    self.XCVS=np.empty(NI, dtype=float)
    self.FV=np.empty(NI, dtype=float)
    self.FVP=np.empty(NI, dtype=float)
    self.FX=np.empty(NI, dtype=float)
    self.FXM=np.empty(NI, dtype=float)
    self.XCVI=np.empty(NI, dtype=float)
    self.XCVIP=np.empty(NI, dtype=float)

    self.Y=np.empty(NJ, dtype=float)
    self.YV=np.empty(NJ, dtype=float)
    self.YDIF=np.empty(NJ, dtype=float)
    self.YCV=np.empty(NJ, dtype=float)
    self.YCVS=np.empty(NJ, dtype=float)
    self.YCVR=np.empty(NJ, dtype=float)
    self.YCVRS=np.empty(NJ, dtype=float)
    self.ARX=np.empty(NJ, dtype=float)
    self.ARXJ=np.empty(NJ, dtype=float)
    self.ARXJP=np.empty(NJ, dtype=float)
    self.R=np.empty(NJ, dtype=float)
    self.RMN=np.empty(NJ, dtype=float)
    self.SX=np.empty(NJ, dtype=float)
    self.SXMN=np.empty(NJ, dtype=float)
    self.FY=np.empty(NJ, dtype=float)
    self.FYM=np.empty(NJ, dtype=float)  

    self.PT=np.empty(NIJ, dtype=float) 
    self.QT=np.empty(NIJ, dtype=float)   

    self.U=np.empty([NI,NJ], dtype=float)  
    self.V=np.empty([NI,NJ], dtype=float)  
    self.PC=np.empty([NI,NJ], dtype=float)  
    self.T=np.empty([NI,NJ], dtype=float)  
    self.P=np.empty([NI,NJ], dtype=float)  
    self.RHO=np.empty([NI,NJ], dtype=float)  
    self.GAM=np.empty([NI,NJ], dtype=float)  
    self.CON=np.empty([NI,NJ], dtype=float)  
    self.Ur=np.empty([NI,NJ], dtype=float)  
    self.Vr=np.empty([NI,NJ], dtype=float)    
    self.AIP=np.empty([NI,NJ], dtype=float)  
    self.AIM=np.empty([NI,NJ], dtype=float)  
    self.AJP=np.empty([NI,NJ], dtype=float)  
    self.AJM=np.empty([NI,NJ], dtype=float)  
    self.AP=np.empty([NI,NJ], dtype=float)  
    self.AP0=np.empty([NI,NJ], dtype=float)  
    self.AP1=np.empty([NI,NJ], dtype=float)    
    self.DU=np.empty([NI,NJ], dtype=float)  
    self.DV=np.empty([NI,NJ], dtype=float)    
    self.EPSI=np.empty([NI,NJ], dtype=float)  
    self.DELH=np.empty([NI,NJ], dtype=float)  
    self.DELHO=np.empty([NI,NJ], dtype=float)    
    self.TO=np.empty([NI,NJ], dtype=float)  
    self.UO=np.empty([NI,NJ], dtype=float)  
    self.VO=np.empty([NI,NJ], dtype=float)  
    self.TOLD=np.empty([NI,NJ], dtype=float)  
    self.UOLD=np.empty([NI,NJ], dtype=float)  
    self.VOLD=np.empty([NI,NJ], dtype=float)  
    self.UHAT=np.empty([NI,NJ], dtype=float)  
    self.VHAT=np.empty([NI,NJ], dtype=float)  

    self.COFU=np.zeros([NI,NJ,6], dtype=float)  
    self.COFV=np.zeros([NI,NJ,6], dtype=float)  
    self.COFP=np.zeros([NI,NJ,6], dtype=float)  
    self.COF=np.zeros([NI,NJ,6], dtype=float)  

    self.RELAX=np.empty(NFX3, dtype=float)  
    self.NTIMES=np.empty(NFX3, dtype=int)  

    self.RELAX[:] = 0.5 
    self.RELAX[0]=0.5
    self.RELAX[1]=0.5
    self.RELAX[2]=0.6
    self.RELAX[3]=0.9

    #Logical
    self.LSOLVE=np.empty(NFX3, dtype=bool)
    self.LPRINT=np.empty(NFX3, dtype=bool)
    self.LBLK=np.empty(NFX3, dtype=bool)  

    self.LSOLVE[:] = False
    self.LSOLVE[0]=True   
    self.LSOLVE[1]=True    
    self.LSOLVE[2]=True    
    self.LSOLVE[3]=True  
    self.LSOLVE[5]=False  
    self.LSOLVE[10]=True    
    self.LSTOP=False   
        
    for i in range(NFX3):   
      self.NTIMES[i]=1   
      self.LBLK[i]=True   
 
    self.TITLE=['UVEL','VVEL','','TEMP']   
    self.ITER=0   
    self.R[0]=0.0   

    #CON,AP,U,V,RHO,PC AND P ARRAYS ARE INITIALIZED HERE    
    for J in range(M1):   
      for I in range(L1):   
        self.U[I,J]=0.0
        self.V[I,J]=0.0
        self.PC[I,J]=0.0
        self.T[I,J]=0.0
        self.P[I,J]=0.0 
        self.RHO[I,J]=self.RHOCON  
        self.DU[I,J]=0.0
        self.DV[I,J]=0.0
        self.UOLD[I,J]=0.0
        self.VOLD[I,J]=0.0  
        self.AIP[I,J]=0.0
        self.AIM[I,J]=0.0
        self.AJM[I,J]=0.0
        self.AJP[I,J]=0.0
        self.CON[I,J]=0.0
        self.AP[I,J]=0.0
        self.AP0[I,J]=0.0
        self.AP1[I,J]=0.0
        self.TOLD[I,J]=0.0  
        self.EPSI[I,J]=0.0
        self.DELH[I,J]=0.0
        self.DELHO[I,J]=0.0

  def grid_initialize(self):
    """
    THIS FUNCTION GENERATES UNIFORM GRID
    """
    self.XU[1]=0.0   
    DX=self.XL/float(self.L1-1)   
    for I in range(2,self.L1):   
      self.XU[I]=self.XU[I-1]+DX   
    self.YV[1]=0.0   
    DY=self.YL/float(self.M1-1)   
    for J in range(2,self.M1):   
      self.YV[J]=self.YV[J-1]+DY 
    #print self.XU
    #print self.YV

  def geometry_initialize(self):
    """
    THIS FUNCTION GENERATES THE CONTROL VOLUMES&VARIOUS RELATED-  
    GEOMETRICAL PARAMETERS.
    """  
                                                                     
    self.NRHO=self.NP+1   
    self.NGAM=self.NRHO+1 
    L1=self.L1
    M1=self.M1  
    L2=self.L2=self.L1-1   
    L3=self.L3=self.L2-1   
    M2=self.M2=self.M1-1   
    M3=self.M3=self.M2-1   
    self.X[0]=self.XU[1]   
    for I in range(1,L2):   
      self.X[I]=0.5*(self.XU[I+1]+self.XU[I])   
    self.X[L1-1]=self.XU[L1-1]   
    self.Y[0]=self.YV[1]   
    for J in range(1,M2):   
      self.Y[J]=0.5*(self.YV[J+1]+self.YV[J])   
    self.Y[M1-1] = self.YV[M1-1]   
    for I in range(1,L1):   
      self.XDIF[I]=self.X[I]-self.X[I-1]   
    for I in range(1,L2):   
      self.XCV[I]=self.XU[I+1]-self.XU[I]   
    for I in range(2,L2):   
      self.XCVS[I]=self.XDIF[I]   
    self.XCVS[2]= self.XCVS[2] + self.XDIF[1]   
    self.XCVS[L2-1]= self.XCVS[L2-1] + self.XDIF[L1-1]   
    for I in range(2,L3):   
      self.XCVI[I]=0.5*self.XCV[I]   
      self.XCVIP[I]=self.XCVI[I]   
    self.XCVIP[1] = self.XCV[1]   
    self.XCVI[L2-1]=self.XCV[L2-1]   
    for J in range(1,M1):   
      self.YDIF[J]=self.Y[J]-self.Y[J-1]   
    for J in range(1,M2):   
      self.YCV[J]=self.YV[J+1]-self.YV[J]   
    for J in range(2,M2):   
      self.YCVS[J]=self.YDIF[J]   
    self.YCVS[2]=self.YCVS[2]+self.YDIF[2]   
    self.YCVS[M2-1]=self.YCVS[M2-1]+self.YDIF[M1-1]   

    if self.MODE <> 1: 
      for J in range(1,M1):   
        self.R[J]=self.R[J-1]+self.YDIF[J]   
      self.RMN[1]=self.R[0]   
      for J in range(2,M2):   
        self.RMN[J]=self.RMN[J-1]+self.YCV[J-1]   
      self.RMN[M1-1]=self.R[M1-1]   
    else:
      for J in range(0,M1):   
        self.RMN[J]=1.0   
        self.R[J]=1.0   

    for J in range(0,M1):    
      self.SX[J]=1.   
      self.SXMN[J]=1.   
      if self.MODE <> 3: 
        continue  
      self.SX[J]=self.R[J]   
      if J<>1:
        self.SXMN[J]=self.RMN[J]

    for J in range(1,M2):   
      self.YCVR[J]=self.R[J]*self.YCV[J]   
      self.ARX[J]=self.YCVR[J]   
      if self.MODE <> 3: 
        continue   
      self.ARX[J]=self.YCV[J]   

    for J in range(3,M3):   
      self.YCVRS[J]=0.5*(self.R[J]+self.R[J-1])*self.YDIF[J]   
    self.YCVRS[2]=0.5*(self.R[2]+self.R[0])*self.YCVS[2]   
    self.YCVRS[M2-1]=0.5*(self.R[M1-1]+self.R[M3-1])*self.YCVS[M2-1]   
    
    if self.MODE == 2:   
      for J in range(2,M3):   
        self.ARXJ[J]=0.25*(1.0+self.RMN[J]/self.R[J])*self.ARX[J]   
        self.ARXJP[J]=self.ARX[J]-self.ARXJ[J]   

    else:  
      for J in range(2,M3):   
        self.ARXJ[J]=0.5*self.ARX[J]   
        self.ARXJP[J]=self.ARXJ[J]

    self.ARXJP[1]=self.ARX[1]   
    self.ARXJ[M2-1]=self.ARX[M2-1]   
    for J in range(2,M3):  
      self.FV[J]=self.ARXJP[J]/self.ARX[J]   
      self.FVP[J]=1.0-self.FV[J]   
    for I in range(2,L2):   
      self.FX[I]=0.5*self.XCV[I-1]/self.XDIF[I]   
      self.FXM[I]=1.0-self.FX[I]   
    self.FX[1]=0.   
    self.FXM[1]=1.   
    self.FX[L1-1]=1.   
    self.FXM[L1-1]=0.   
    for J in range(2,M2):   
      self.FY[J]=0.5*self.YCV[J-1]/self.YDIF[J]   
      self.FYM[J]=1.0-self.FY[J]   
    self.FY[1]=0.   
    self.FYM[1]=1.   
    self.FY[M1-1]=1.   
    self.FYM[M1-1]=0.   
                                                                
    self.IM4=self.M1-3   
    self.IM5=self.M1-4   

    if self.MODE == 1: 
      print('COMPUTATION  IN  CARTESIAN  COORDINATES')   
    if self.MODE == 2: 
      print('COMPUTATION FOR AXISYMMETRIC SITUATION')   
    if self.MODE == 3: 
      print('COMPUTATION   IN   POLAR   COORDINATES')      
    #print self.X
    #print self.Y    

  def start(self):  
    """
    THIS FUNCTION GIVES INITIAL CONDITIONS FOR THE PROBLEM 
    """
    for I in range(0,self.L1):   
      for J in range(0,self.M1):                                        
        self.U[I,J]=0.0  
        self.V[I,J]=0.0   
        self.T[I,J]=self.TINITIAL 
        self.TOLD[I,J]=self.T[I,J] 
        self.RHO[I,J]=self.RHOCON   

  def diflow(self):   
    """
    THIS FUNCTION CALCULATES DISCRETIZATION EQN COEFFS AS PER POWER LAW.
    """
    self.ACOF = self.DIFF   
    if self.FLOW==0:
      return   
    self.TEMP = self.DIFF - abs(self.FLOW)*0.1   
    self.ACOF=0.0   
    if self.TEMP<=0:
      return   
    self.TEMP=self.TEMP/self.DIFF   
    self.ACOF=self.DIFF*self.TEMP**5   

  def dense(self):  
    """
    THIS FUNCTION CALCULATES THE DENSITY AT A PARTICULAR TIME
    """
    for I in range(0,self.L1):   
      for J in range(0,self.M1):   
        self.RHO[I,J]=self.RHOCON  

  def oldval(self):  
    """
    THIS FUNCTION STORES INITIAL VALUES FOR THE COMING TIMESTEP 
    """
    for I in range(0,self.L1):   
      for J in range(0,self.M1): 
          self.TO[I,J]=self.T[I,J]   
          self.UO[I,J]=self.U[I,J]   
          self.VO[I,J]=self.V[I,J]
          self.DELHO[I,J]=self.DELH[I,J]   
 
  def reset(self): 
    """
    THIS FUNCTION RESETS 'ap' AND 'con' TO zero
    """  
    for J in range(1,self.M2):  
      for I in range(1,self.L2):  
        self.CON[I,J]=0.0   
        self.AP[I,J]=0.0 

  def gamsor(self): 
    """
    THIS FUNCTION GIVES DIFF. COEFF.& SOURCE TERMS FOR
    DISCRETISED EQNS.
    """ 
    if self.NF<=1:
      for J in range(0,self.M1):  
        for I in range(0,self.L1): 
          self.GAM[I,J]=self.VISCOSITY    
          if self.NF == 1:
            self.GAM[I,self.M1-1]=0.0 #top face
            self.TMEAN=self.FY[J]*self.T[I,J]+self.FYM[J]*self.T[I,J-1]
            self.CON[I,J]=self.RHOREF*self.G*self.BETA*(self.TMEAN-self.TMELT)

    if self.NF==3:
      for J in range(0,self.M1):  
        for I in range(0,self.L1): 
          self.GAM[I,J]=self.TK/self.CP 
          self.CON[I,J]=-1.0*self.RHO[I,J]*(self.DELH[I,J]-self.DELHO[I,J])/(self.CP*self.DT)

  def solve(self):
    """
    THIS FUNCTION SOLVES DISCRETISATION EQUATIONS BY 'LINE BY LINE TDMA'.
    """   
    self.F=np.empty([self.NI,self.NJ], dtype=float)    
    if self.NF == 0:
      self.F[:,:] = self.U[:,:]  
    if self.NF == 1:
      self.F[:,:] = self.V[:,:]  
    if self.NF == 2:
      self.F[:,:] = self.PC[:,:]  
    if self.NF == 3:
      self.F[:,:] = self.T[:,:]  
    if self.NF == self.NP:  
      self.F[:,:] = self.P[:,:]  
         
    self.ISTF=self.IST-1-1   
    self.JSTF=self.JST-1-1   
    self.IT1=self.L2+self.IST-2   
    self.IT2=self.L3+self.IST-2   
    self.JT1=self.M2+self.JST-2   
    self.JT2=self.M3+self.JST-2   
                                                                        
    for NT in range(0,self.NTIMES[self.NF]):   
      self.N=self.NF   
      if self.LBLK[self.NF]:   
        self.PT[self.ISTF]=0.   
        self.QT[self.ISTF]=0.   
        for I in range(self.IST-1,self.L2):   
          self.BL=0.0   
          self.BLP=0.0   
          self.BLM=0.0   
          self.BLC=0.0   
          for J in range(self.JST-1,self.M2):   
            self.BL=self.BL+self.AP[I,J]   
            if J <> self.M2-1:  
              self.BL=self.BL-self.AJP[I,J]   
            if J <> self.JST-1: 
              self.BL=self.BL-self.AJM[I,J]   
            self.BLP=self.BLP+self.AIP[I,J]   
            self.BLM=self.BLM+self.AIM[I,J]   
            self.BLC=self.BLC+self.CON[I,J]+self.AIP[I,J]*self.F[I+1,J]+self.AIM[I,J]*self.F[I-1,J]+self.AJP[I,J]*self.F[I,J+1]+self.AJM[I,J]*self.F[I,J-1]-self.AP[I,J]*self.F[I,J]        
          self.DENOM=self.BL-self.PT[I-1]*self.BLM   
          if abs(self.DENOM/self.BL)< 10.0**(-10):
            self.DENOM=10.0**30   
          self.PT[I]=self.BLP/self.DENOM   
          self.QT[I]=(self.BLC+self.BLM*self.QT[I-1])/self.DENOM   

        self.BL=0.0   
        for II in range(self.IST-1,self.L2):   
          I=self.IT1-II   
          self.BL=self.BL*self.PT[I]+self.QT[I]   
          for J in range(self.JST-1,self.M2):   
            self.F[I,J]=self.F[I,J]+self.BL   
        self.PT[self.JSTF]=0.0   
        self.QT[self.JSTF]=0.0   
        for J in range(self.JST-1,self.M2):   
          self.BL=0.0   
          self.BLP=0.0   
          self.BLM=0.0   
          self.BLC=0.0   
          for I in range(self.IST-1,self.L2):   
            self.BL=self.BL+self.AP[I,J]   
            if I <> self.L2-1: 
              self.BL=self.BL-self.AIP[I,J]   
            if I <> self.IST-1: 
              self.BL=self.BL-self.AIM[I,J]   
            self.BLP=self.BLP+self.AJP[I,J]   
            self.BLM=self.BLM+self.AJM[I,J]   
            self.BLC=self.BLC+self.CON[I,J]+self.AIP[I,J]*self.F[I+1,J]+self.AIM[I,J]*self.F[I-1,J]+self.AJP[I,J]*self.F[I,J+1]+self.AJM[I,J]*self.F[I,J-1]-self.AP[I,J]*self.F[I,J]        
          self.DENOM=self.BL-self.PT[J-1]*self.BLM   
          if abs(self.DENOM/self.BL) < 10.0**(-10): 
            self.DENOM = 10.0**30   
          self.PT[J]=self.BLP/self.DENOM   
          self.QT[J]=(self.BLC+self.BLM*self.QT[J-1])/self.DENOM   
        self.BL=0.0   
        for JJ in range(self.JST-1,self.M2):   
          J=self.JT1-JJ   
          self.BL=self.BL*self.PT[J]+self.QT[J]   
          for I in range(self.IST-1,self.L2):   
            self.F[I,J]=self.F[I,J]+self.BL   
                                                                    
      for J in range(self.JST-1,self.M2):   
        self.PT[self.ISTF]=0.0   
        self.QT[self.ISTF]=self.F[self.ISTF,J]   
        for I in range(self.IST-1,self.L2):   
          self.DENOM=self.AP[I,J]-self.PT[I-1]*self.AIM[I,J]   
          self.PT[I]=self.AIP[I,J]/self.DENOM   
          self.TEMP=self.CON[I,J]+self.AJP[I,J]*self.F[I,J+1]+self.AJM[I,J]*self.F[I,J-1]   
          self.QT[I]=(self.TEMP+self.AIM[I,J]*self.QT[I-1])/self.DENOM   
        for II in range(self.IST-1,self.L2):   
          I=self.IT1-II   
          self.F[I,J]=self.F[I+1,J]*self.PT[I]+self.QT[I]   

                                                                        
      for JJ in range(self.JST-1,self.M3):   
          J=self.JT2-JJ   
          self.PT[self.ISTF]=0.   
          self.QT[self.ISTF]=self.F[self.ISTF,J]   
          for I in range(self.IST-1,self.L2):   
            self.DENOM=self.AP[I,J]-self.PT[I-1]*self.AIM[I,J]   
            self.PT[I]=self.AIP[I,J]/self.DENOM   
            self.TEMP=self.CON[I,J]+self.AJP[I,J]*self.F[I,J+1]+self.AJM[I,J]*self.F[I,J-1]   
            self.QT[I]=(self.TEMP+self.AIM[I,J]*self.QT[I-1])/self.DENOM         
          for II in range(self.IST-1,self.L2):   
            I=self.IT1-II   
            self.F[I,J]=self.F[I+1,J]*self.PT[I]+self.QT[I]    
                                                                        
      for I in range(self.IST-1,self.L2):   
        self.PT[self.JSTF]=0.0   
        self.QT[self.JSTF]=self.F[I,self.JSTF]   
        for J in range(self.JST-1,self.M2):   
          self.DENOM=self.AP[I,J]-self.PT[J-1]*self.AJM[I,J]   
          self.PT[J]=self.AJP[I,J]/self.DENOM   
          self.TEMP=self.CON[I,J]+self.AIP[I,J]*self.F[I+1,J]+self.AIM[I,J]*self.F[I-1,J]   
          self.QT[J]=(self.TEMP+self.AJM[I,J]*self.QT[J-1])/self.DENOM   
        for JJ in range(self.JST-1,self.M2):   
          J=self.JT1-JJ   
          self.F[I,J]=self.F[I,J+1]*self.PT[J]+self.QT[J]   
                                                                      
      for II in range(self.IST-1,self.L3):   
        I=self.IT2-II   
        self.PT[self.JSTF]=0.0   
        self.QT[self.JSTF]=self.F[I,self.JSTF]   
        for J in range(self.JST-1,self.M2):   
          self.DENOM=self.AP[I,J]-self.PT[J-1]*self.AJM[I,J]   
          self.PT[J]=self.AJP[I,J]/self.DENOM   
          self.TEMP=self.CON[I,J]+self.AIP[I,J]*self.F[I+1,J]+self.AIM[I,J]*self.F[I-1,J]   
          self.QT[J]=(self.TEMP+self.AJM[I,J]*self.QT[J-1])/self.DENOM   
        for JJ in range(self.JST-1,self.M2):   
          J=self.JT1-JJ   
          self.F[I,J]=self.F[I,J+1]*self.PT[J]+self.QT[J]   
   
    if self.NF == 0:
      self.U[:,:] = self.F[:,:] 
    if self.NF == 1:
      self.V[:,:] = self.F[:,:]   
    if self.NF == 2:
      self.PC[:,:] = self.F[:,:]   
    if self.NF == 3:
      self.T[:,:] = self.F[:,:]   
    if self.NF == self.NP:  
      self.P[:,:] = self.F[:,:]    
    self.reset()
  
  def coeff(self):
    """
    THIS FUNCTION FORMS COEFFS. FOR DISCRETISATION EQNS.
    """  
    #COEFFICIENTS FOR THE U EQUATION.
    self.ITERL=1   
    self.ZERO=0.0   
    self.LCONV = False  
    while self.LCONV == False:
      self.reset()  
      self.NF=0   
      if self.LSOLVE[self.NF]:   
        self.IST=3   
        self.JST=2   
        self.gamsor()   
        self.REL=1.0-self.RELAX[self.NF]   
        for I in range(2,self.L2):   
          self.FL=self.XCVI[I]*self.V[I,1]*self.RHO[I,0]   
          self.FLM=self.XCVIP[I-1]*self.V[I-1,1]*self.RHO[I-1,0]   
          self.FLOW=self.R[0]*(self.FL+self.FLM)   
          self.DIFF=self.R[0]*(self.XCVI[I]*self.GAM[I,0]+self.XCVIP[I-1]*self.GAM[I-1,0])/self.YDIF[1]  
          self.diflow()   
          self.AJM[I,1]=self.ACOF+max(self.ZERO,self.FLOW)   
            
        for J in range(1,self.M2):   
          self.FLOW = self.ARX[J]*self.U[1,J]*self.RHO[0,J]   
          self.DIFF = self.ARX[J]*self.GAM[0,J]/(self.XCV[1]*self.SX[J])   
          self.diflow()  
          self.AIM[2,J]=self.ACOF+max(self.ZERO,self.FLOW)   
          for I in range(2,self.L2):   
            if I <> self.L2-1:  
                self.FL=self.U[I,J]*(self.FX[I]*self.RHO[I,J]+self.FXM[I]*self.RHO[I-1,J])   
                self.FLP=self.U[I+1,J]*(self.FX[I+1]*self.RHO[I+1,J]+self.FXM[I+1]*self.RHO[I,J])   
                self.FLOW=self.ARX[J]*0.5*(self.FL+self.FLP)   
                self.DIFF=self.ARX[J]*self.GAM[I,J]/(self.XCV[I]*self.SX[J])   
            else:   
                self.FLOW=self.ARX[J]*self.U[self.L1-1,J]*self.RHO[self.L1-1,J]   
                self.DIFF=self.ARX[J]*self.GAM[self.L1-1,J]/(self.XCV[self.L2-1]*self.SX[J])   
            self.diflow()   
            self.AIM[I+1,J]=self.ACOF+max(self.ZERO,self.FLOW)   
            self.AIP[I,J]=self.AIM[I+1,J]-self.FLOW   
            if J <> self.M2-1:   
              self.FL=self.XCVI[I]*self.V[I,J+1]*(self.FY[J+1]*self.RHO[I,J+1]+self.FYM[J+1]*self.RHO[I,J])   
              self.FLM=self.XCVIP[I-1]*self.V[I-1,J+1]*(self.FY[J+1]*self.RHO[I-1,J+1]+self.FYM[J+1]*self.RHO[I-1,J])                                                        
              self.GM=self.GAM[I,J]*self.GAM[I,J+1]/(self.YCV[J]*self.GAM[I,J+1]+self.YCV[J+1]*self.GAM[I,J]+10.0**(-30))*self.XCVI[I]                                                   
              self.GMM=self.GAM[I-1,J]*self.GAM[I-1,J+1]/(self.YCV[J]*self.GAM[I-1,J+1]+self.YCV[J+1]*self.GAM[I-1,J]+10.0**(-30))*self.XCVIP[I-1]                                      
              self.DIFF=self.RMN[J+1]*2.0*(self.GM+self.GMM)   
            else:
              self.FL=self.XCVI[I]*self.V[I,self.M1-1]*self.RHO[I,self.M1-1]   
              self.FLM=self.XCVIP[I-1]*self.V[I-1,self.M1-1]*self.RHO[I-1,self.M1-1]   
              self.DIFF=self.R[self.M1-1]*(self.XCVI[I]*self.GAM[I,self.M1-1]+self.XCVIP[I-1]*self.GAM[I-1,self.M1-1])/self.YDIF[self.M1-1]   
            self.FLOW=self.RMN[J+1]*(self.FL+self.FLM)   
            self.diflow()   
            self.AJM[I,J+1]=self.ACOF+max(self.ZERO,self.FLOW)   
            self.AJP[I,J]=self.AJM[I,J+1]-self.FLOW   
            self.VOL=self.YCVR[J]*self.XCVS[I]   
            self.APT=(self.RHO[I,J]*self.XCVI[I]+self.RHO[I-1,J]*self.XCVIP[I-1])/(self.XCVS[I]*self.DT)                                                       
            self.AP[I,J]=self.AP[I,J]-self.APT   
            self.CON[I,J]=self.CON[I,J]+self.APT*self.UO[I,J]   
            self.AP[I,J]=(-self.AP[I,J]*self.VOL+self.AIP[I,J]+self.AIM[I,J]+self.AJP[I,J]+self.AJM[I,J])/self.RELAX[self.NF]                                                          
            
            #CARMAN-KOZENY RELATION FOR POROUS MEDIA  EFFECTS
            self.AP0[I,J]=self.RHO[I,J]*self.VOL/self.DT     
            self.EPMEAN=self.FX[I]*self.EPSI[I,J]+self.FXM[I]*self.EPSI[I-1,J]
            self.AP[I,J]=self.AP[I,J]+1.0*self.DARCYCONST*(1.0-self.EPMEAN)**2*self.VOL/(self.EPMEAN**3+10.0**(-3))
            self.CON[I,J]=self.CON[I,J]*self.VOL+self.REL*self.AP[I,J]*self.U[I,J]   
            self.DU[I,J]=self.VOL/(self.XDIF[I]*self.SX[J])  
            self.DU[I,J]=self.DU[I,J]/self.AP[I,J]  
   
        self.COFU[:,:,0] = self.CON[:,:]  
        self.COFU[:,:,1] = self.AIP[:,:]  
        self.COFU[:,:,2] = self.AIM[:,:]  
        self.COFU[:,:,3] = self.AJP[:,:]  
        self.COFU[:,:,4] = self.AJM[:,:]  
        self.COFU[:,:,5] = self.AP[:,:]  
        
        #COEFFICIENTS FOR THE  V  EQUATION. 
        self.NF=1   
        self.reset()   
        self.IST=2   
        self.JST=3   
        self.gamsor()  
        self.REL=1.0-self.RELAX[self.NF]   
        for I in range(1,self.L2):  
          self.AREA=self.R[0]*self.XCV[I]   
          self.FLOW=self.AREA*self.V[I,1]*self.RHO[I,0]   
          self.DIFF=self.AREA*self.GAM[I,0]/self.YCV[1]   
        self.diflow()   
        self.AJM[I,2]=self.ACOF+ max(self.ZERO,self.FLOW) 

        for J in range(2,self.M2):   
          self.FL=self.ARXJ[J]*self.U[1,J]*self.RHO[0,J]   
          self.FLM=self.ARXJP[J-1]*self.U[1,J-1]*self.RHO[0,J-1]   
          self.FLOW=self.FL+self.FLM   
          self.DIFF=(self.ARXJ[J]*self.GAM[0,J]+self.ARXJP[J-1]*self.GAM[0,J-1])/(self.XDIF[1]*self.SXMN[J])   
          self.diflow()
          self.AIM[1,J]=self.ACOF+max(self.ZERO,self.FLOW)   
          for I in range(1,self.L2):   
            if I <> self.L2-1:   
              self.FL=self.ARXJ[J]*self.U[I+1,J]*(self.FX[I+1]*self.RHO[I+1,J]+self.FXM[I+1]*self.RHO[I,J])   
              self.FLM=self.ARXJP[J-1]*self.U[I+1,J-1]*(self.FX[I+1]*self.RHO[I+1,J-1]+self.FXM[I+1]*self.RHO[I,J-1])                                                       
              self.GM=self.GAM[I,J]*self.GAM[I+1,J]/(self.XCV[I]*self.GAM[I+1,J]+self.XCV[I+1]*self.GAM[I,J]+ 10.0**(-30))*self.ARXJ[J]                                                    
              self.GMM=self.GAM[I,J-1]*self.GAM[I+1,J-1]/(self.XCV[I]*self.GAM[I+1,J-1]+self.XCV[I+1]*self.GAM[I,J-1]+ 10.0**(-30))*self.ARXJP[J-1]                                    
              self.DIFF=2.0*(self.GM+self.GMM)/self.SXMN[J]   
            else:   
              self.FL=self.ARXJ[J]*self.U[self.L1-1,J]*self.RHO[self.L1-1,J]   
              self.FLM=self.ARXJP[J-1]*self.U[self.L1-1,J-1]*self.RHO[self.L1-1,J-1]   
              self.DIFF=(self.ARXJ[J]*self.GAM[self.L1-1,J]+self.ARXJP[J-1]*self.GAM[self.L1-1,J-1])/(self.XDIF[self.L1-1]*self.SXMN[J])   
            self.FLOW=self.FL+self.FLM   
            self.diflow()   
            self.AIM[I+1,J]=self.ACOF+ max(self.ZERO,self.FLOW)   
            self.AIP[I,J]=self.AIM[I+1,J]-self.FLOW   
            if J <> self.M2-1:   
              self.AREA=self.R[J]*self.XCV[I]   
              self.FL=self.V[I,J]*(self.FY[J]*self.RHO[I,J]+self.FYM[J]*self.RHO[I,J-1])*self.RMN[J]   
              self.FLP=self.V[I,J+1]*(self.FY[J+1]*self.RHO[I,J+1]+self.FYM[J+1]*self.RHO[I,J])*self.RMN[J+1]   
              self.FLOW=(self.FV[J]*self.FL+self.FVP[J]*self.FLP)*self.XCV[I]   
              self.DIFF=self.AREA*self.GAM[I,J]/self.YCV[J]    
            else:
              self.AREA=self.R[self.M1-1]*self.XCV[I]   
              self.FLOW=self.AREA*self.V[I,self.M1-1]*self.RHO[I,self.M1-1]   
              self.DIFF=self.AREA*self.GAM[I,self.M1-1]/self.YCV[self.M2-1]   
            self.diflow()   
            self.AJM[I,J+1]=self.ACOF+ max(self.ZERO,self.FLOW)   
            self.AJP[I,J]=self.AJM[I,J+1]-self.FLOW   
            self.VOL=self.YCVRS[J]*self.XCV[I]   
            self.SXT=self.SX[J]   
            if J == self.M2-1:
              self.SXT=self.SX[self.M1-1]   
            self.SXB=self.SX[J-1]   
            if J == 2:
              self.SXB=self.SX[0]   
            self.APT=(self.ARXJ[J]*self.RHO[I,J]*0.5*(self.SXT+self.SXMN[J])+self.ARXJP[J-1]*self.RHO[I,J-1]*0.5*(self.SXB+self.SXMN[J]))/(self.YCVRS[J]*self.DT)                                    
            self.AP[I,J]=self.AP[I,J]-self.APT   
            self.CON[I,J]=self.CON[I,J]+self.APT*self.VO[I,J]   
            self.AP[I,J]=(-self.AP[I,J]*self.VOL+self.AIP[I,J]+self.AIM[I,J]+self.AJP[I,J]+self.AJM[I,J])/self.RELAX[self.NF]                                                          
            
            #THIS IS CORMAN RELATION FOR POROUS MEDIA EFFECTS
            self.AP0[I,J]=self.RHO[I,J]*self.VOL/self.DT 
            self.EPMEAN=self.FY[J]*self.EPSI[I,J]+self.FYM[J]*self.EPSI[I,J-1]
            self.AP[I,J]=self.AP[I,J]+1.0*self.DARCYCONST*(1.0-self.EPMEAN)**2*self.VOL/(self.EPMEAN**3+10.0**(-3))
            self.CON[I,J]=self.CON[I,J]*self.VOL+self.REL*self.AP[I,J]*self.V[I,J]   
            self.DV[I,J]=self.VOL/self.YDIF[J]  
            self.DV[I,J]=self.DV[I,J]/self.AP[I,J]  


        self.COFV[:,:,0] = self.CON[:,:]  
        self.COFV[:,:,1] = self.AIP[:,:]  
        self.COFV[:,:,2] = self.AIM[:,:]  
        self.COFV[:,:,3] = self.AJP[:,:]  
        self.COFV[:,:,4] = self.AJM[:,:]  
        self.COFV[:,:,5] = self.AP[:,:]  
    
        #CALCULATE UHAT AND VHAT. 
        for J in range(1,self.M2):  
          for I in range(2,self.L2):  
            self.UHAT[I,J]=(self.COFU[I,J,1]*self.U[I+1,J]+self.COFU[I,J,2]*self.U[I-1,J]+self.COFU[I,J,3]*self.U[I,J+1]+self.COFU[I,J,4]*self.U[I,J-1]+self.COFU[I,J,0])/self.COFU[I,J,5]  
        for J in range(2,self.M2):  
          for I in range(1,self.L2):  
            self.VHAT[I,J]=(self.COFV[I,J,1]*self.V[I+1,J]+self.COFV[I,J,2]*self.V[I-1,J]+self.COFV[I,J,3]*self.V[I,J+1]+self.COFV[I,J,4]*self.V[I,J-1]+self.COFV[I,J,0])/self.COFV[I,J,5]  
   
        #COEFFICIENTS FOR THE PRESSURE EQUATION. 
        self.NF=2   
        self.reset()  
        self.IST=2   
        self.JST=2   
        self.gamsor()  
        for J in range(1,self.M2):  
          for I in range(1,self.L2):                                                   
            self.VOL=self.YCVR[J]*self.XCV[I]   
            self.CON[I,J]=self.CON[I,J]*self.VOL   
        
        for I in range(1,self.L2):   
          self.ARHO=self.R[0]*self.XCV[I]*self.RHO[I,0]   
          self.CON[I,1]=self.CON[I,1]+self.ARHO*self.V[I,1]   
          self.AJM[I,1]=0.0   
              
        for J in range(1,self.M2):
          self.ARHO=self.ARX[J]*self.RHO[0,J]   
          self.CON[1,J]=self.CON[1,J]+self.ARHO*self.U[1,J]   
          self.AIM[1,J]=0.0   
          for I in range(1,self.L2):  
            if I<>self.L2-1:   
              self.ARHO=self.ARX[J]*(self.FX[I+1]*self.RHO[I+1,J]+self.FXM[I+1]*self.RHO[I,J])   
              self.FLOW=self.ARHO*self.UHAT[I+1,J]   
              self.CON[I,J]=self.CON[I,J]-self.FLOW   
              self.CON[I+1,J]=self.CON[I+1,J]+self.FLOW   
              self.AIP[I,J]=self.ARHO*self.DU[I+1,J]   
              self.AIM[I+1,J]=self.AIP[I,J]   
            else:  
              self.ARHO=self.ARX[J]*self.RHO[self.L1-1,J]   
              self.CON[I,J]=self.CON[I,J]-self.ARHO*self.U[self.L1-1,J]   
              self.AIP[I,J]=0.0   
            if J<>self.M2-1:   
              self.ARHO=self.RMN[J+1]*self.XCV[I]*(self.FY[J+1]*self.RHO[I,J+1]+self.FYM[J+1]*self.RHO[I,J])   
              self.FLOW=self.ARHO*self.VHAT[I,J+1]   
              self.CON[I,J]=self.CON[I,J]-self.FLOW   
              self.CON[I,J+1]=self.CON[I,J+1]+self.FLOW   
              self.AJP[I,J]=self.ARHO*self.DV[I,J+1]   
              self.AJM[I,J+1]=self.AJP[I,J]   
            else:   
              self.ARHO=self.RMN[self.M1-1]*self.XCV[I]*self.RHO[I,self.M1-1]   
              self.CON[I,J]=self.CON[I,J]-self.ARHO*self.V[I,self.M1-1]   
              self.AJP[I,J]=0.0   
            self.AP[I,J]=self.AIP[I,J]+self.AIM[I,J]+self.AJP[I,J]+self.AJM[I,J]     

        for J in range(1,self.M2):  
          for I in range(1,self.L2):   
            self.AP[I,J]=self.AP[I,J]/self.RELAX[self.NP]   
            self.CON[I,J]=self.CON[I,J]+(1.0-self.RELAX[self.NP])*self.AP[I,J]*self.P[I,J]   
       
        self.COFP[:,:,1] = self.AIP[:,:]  
        self.COFP[:,:,2] = self.AIM[:,:]  
        self.COFP[:,:,3] = self.AJP[:,:]  
        self.COFP[:,:,4] = self.AJM[:,:]  

        self.NF=self.NP   
        self.solve()   
                             
        #COMPUTE U* AND V*. 
        self.NF=0   
        self.IST=3   
        self.JST=2   
        for K in range(0,6):   
          for J in range(self.JST-1,self.M2):   
            for I in range(self.IST-1,self.L2):   
              self.COF[I,J,K]=self.COFU[I,J,K]   
       
           
        self.CON[:,:] = self.COF[:,:,0]  
        self.AIP[:,:] = self.COF[:,:,1]  
        self.AIM[:,:] = self.COF[:,:,2]  
        self.AJP[:,:] = self.COF[:,:,3]  
        self.AJM[:,:] = self.COF[:,:,4]  
        self.AP[:,:] = self.COF[:,:,5]  
   
        for J in range(self.JST-1,self.M2):   
          for I in range(self.IST-1,self.L2):  
            self.CON[I,J]=self.CON[I,J]+self.DU[I,J]*self.AP[I,J]*(self.P[I-1,J]-self.P[I,J])   
        self.solve()   
                             
        self.NF=1   
        self.IST=2   
        self.JST=3   
        for K in range(0,6):   
          for J in range(self.JST-1,self.M2):   
            for I in range(self.IST-1,self.L2): 
              self.COF[I,J,K]=self.COFV[I,J,K]   
       
          
        self.CON[:,:] = self.COF[:,:,0]  
        self.AIP[:,:] = self.COF[:,:,1]  
        self.AIM[:,:] = self.COF[:,:,2]  
        self.AJP[:,:] = self.COF[:,:,3]  
        self.AJM[:,:] = self.COF[:,:,4]  
        self.AP[:,:] = self.COF[:,:,5]  

       
        for J in range(self.JST-1,self.M2):   
          for I in range(self.IST-1,self.L2):   
            self.CON[I,J]=self.CON[I,J]+self.DV[I,J]*self.AP[I,J]*(self.P[I,J-1]-self.P[I,J])   
        self.solve() 
        
        #COEFFICIENTS FOR THE PRESSURE CORRECTION EQUATION. 
        self.NF=2   
        self.reset()   
        self.IST=2   
        self.JST=2   
        for K in range(1,5):   
          for J in range(self.JST-1,self.M2):   
            for I in range(self.IST-1,self.L2):  
              self.COF[I,J,K]=self.COFP[I,J,K]   
        
        self.AIP[:,:] = self.COF[:,:,1]  
        self.AIM[:,:] = self.COF[:,:,2]  
        self.AJP[:,:] = self.COF[:,:,3]  
        self.AJM[:,:] = self.COF[:,:,4]  
         
        self.gamsor()  
        self.SMAX=0.   
        self.SSUM=0.   
        for J in range(1,self.M2):   
          for I in range(1,self.L2):   
            self.VOL=self.YCVR[J]*self.XCV[I]   
            self.CON[I,J]=self.CON[I,J]*self.VOL   
        for I in range(1,self.L2):  
          self.ARHO=self.R[0]*self.XCV[I]*self.RHO[I,0]   
          self.CON[I,1]=self.CON[I,1]+self.ARHO*self.V[I,1]   
    
        for J in range(1,self.M2): 
          self.ARHO=self.ARX[J]*self.RHO[0,J]   
          self.CON[1,J]=self.CON[1,J]+self.ARHO*self.U[1,J]   
          for I in range(1,self.L2):  
            if I<>self.L2-1:   
              self.ARHO=self.ARX[J]*(self.FX[I+1]*self.RHO[I+1,J]+self.FXM[I+1]*self.RHO[I,J])   
              self.FLOW=self.ARHO*self.U[I+1,J]   
              self.CON[I,J]=self.CON[I,J]-self.FLOW   
              self.CON[I+1,J]=self.CON[I+1,J]+self.FLOW   
            else:   
              self.ARHO=self.ARX[J]*self.RHO[self.L1-1,J]   
              self.CON[I,J]=self.CON[I,J]-self.ARHO*self.U[self.L1-1,J]   
            if J<>self.M2-1:   
              self.ARHO=self.RMN[J+1]*self.XCV[I]*(self.FY[J+1]*self.RHO[I,J+1]+self.FYM[J+1]*self.RHO[I,J])   
              self.FLOW=self.ARHO*self.V[I,J+1]   
              self.CON[I,J]=self.CON[I,J]-self.FLOW   
              self.CON[I,J+1]=self.CON[I,J+1]+self.FLOW   
            else: 
              self.ARHO=self.RMN[self.M1-1]*self.XCV[I]*self.RHO[I,self.M1-1]   
              self.CON[I,J]=self.CON[I,J]-self.ARHO*self.V[I,self.M1-1]   
            self.AP[I,J]=self.AIP[I,J]+self.AIM[I,J]+self.AJP[I,J]+self.AJM[I,J]   
            self.PC[I,J]=0.0   
            self.SMAX= max(self.SMAX,abs(self.CON[I,J]))   
            self.SSUM=self.SSUM+self.CON[I,J]   
        self.solve()   
        
        #COME HERE TO CORRECT THE VELOCITIES. 
        for J in range(1,self.M2):   
          for I in range(1,self.L2): 
            if I <> 1: 
              self.U[I,J]=self.U[I,J]+self.DU[I,J]*(self.PC[I-1,J]-self.PC[I,J])   
            if J <> 1: 
              self.V[I,J]=self.V[I,J]+self.DV[I,J]*(self.PC[I,J-1]-self.PC[I,J])   

       
      #COEFFICIENTS FOR OTHER EQUATIONS. 
      self.reset()   
      self.IST=2   
      self.JST=2   
      for N in range(3,self.NFMAX):   
        self.NF=N   
        if self.LSOLVE[self.NF]:   
          self.gamsor()   
          self.REL=1.0-self.RELAX[self.NF]   
          for I in range(1,self.L2):  
            self.AREA=self.R[0]*self.XCV[I]   
            self.FLOW=self.AREA*self.V[I,1]*self.RHO[I,0]   
            self.DIFF=self.AREA*self.GAM[I,0]/self.YDIF[1]   
            self.diflow()   
            self.AJM[I,1]=self.ACOF+ max(self.ZERO,self.FLOW)   
          for J in range(1,self.M2):  
            self.FLOW=self.ARX[J]*self.U[1,J]*self.RHO[0,J]   
            self.DIFF=self.ARX[J]*self.GAM[0,J]/(self.XDIF[1]*self.SX[J])   
            self.diflow()   
            self.AIM[1,J]=self.ACOF+ max(self.ZERO,self.FLOW)   
            for I in range(1,self.L2):   
              if I <> self.L2-1:   
                self.FLOW=self.ARX[J]*self.U[I+1,J]*(self.FX[I+1]*self.RHO[I+1,J]+self.FXM[I+1]*self.RHO[I,J])   
                self.DIFF=self.ARX[J]*2.0*self.GAM[I,J]*self.GAM[I+1,J]/((self.XCV[I]*self.GAM[I+1,J]+self.XCV[I+1]*self.GAM[I,J]+10.0**(-30))*self.SX[J])                                  
              else:  
                self.FLOW=self.ARX[J]*self.U[self.L1-1,J]*self.RHO[self.L1-1,J]   
                self.DIFF=self.ARX[J]*self.GAM[self.L1-1,J]/(self.XDIF[self.L1-1]*self.SX[J])   
              self.diflow()   
              self.AIM[I+1,J]=self.ACOF+ max(self.ZERO,self.FLOW)   
              self.AIP[I,J]=self.AIM[I+1,J]-self.FLOW   
              self.AREA=self.RMN[J+1]*self.XCV[I]   
              if J <> self.M2-1:   
                self.FLOW=self.AREA*self.V[I,J+1]*(self.FY[J+1]*self.RHO[I,J+1]+self.FYM[J+1]*self.RHO[I,J])   
                self.DIFF=self.AREA*2.0*self.GAM[I,J]*self.GAM[I,J+1]/(self.YCV[J]*self.GAM[I,J+1]+self.YCV[J+1]*self.GAM[I,J]+10.0**(-30))               
              else:
                self.FLOW=self.AREA*self.V[I,self.M1-1]*self.RHO[I,self.M1-1]   
                self.DIFF=self.AREA*self.GAM[I,self.M1-1]/self.YDIF[self.M1-1]   
              self.diflow()    
              self.AJM[I,J+1]=self.ACOF+ max(self.ZERO,self.FLOW)   
              self.AJP[I,J]=self.AJM[I,J+1]-self.FLOW   

          for I in range(0,self.L1):   
            for J in range(0,self.M1):   
              self.VOL=self.YCVR[J]*self.XCV[I]   
              self.APT=self.RHO[I,J]/self.DT   
              self.AP[I,J]=self.AP[I,J]-self.APT   
              if self.NF == 3: 
                self.CON[I,J]=self.CON[I,J]+self.APT*self.TO[I,J]   
              self.AP[I,J]=(-self.AP[I,J]*self.VOL+self.AIP[I,J]+self.AIM[I,J]+self.AJP[I,J]+self.AJM[I,J])/self.RELAX[self.NF] 
             
              self.AP1[I,J]=self.AP[I,J]

              if self.NF == 3: 
                self.CON[I,J]=self.CON[I,J]*self.VOL+self.REL*self.AP[I,J]*self.T[I,J]   
              if self.NF == self.NP-1:
                self.CON[I,J]=self.CON[I,J]*self.VOL+self.REL*self.AP[I,J]*self.P[I,J]   
              if self.NF == 3:
                self.TOLD[I,J]=self.T[I,J]  
          self.solve() 

          #THIS IS FOR ENTHALPY UPDATING 
          for J in range(1,self.M2):          
            for I in range(1,self.L2):
              self.VOL=self.YCVR[J]*self.XCV[I]      
              self.AP0[I,J]=self.RHO[I,J]*self.VOL/self.DT
              self.DELH[I,J]=self.DELH[I,J]+self.AP1[I,J]*self.CP*self.ARELAX*(self.T[I,J]-self.TMELT)/self.AP0[I,J]
              #PREVENT OVER OR UNDER SHOOTING
              if self.DELH[I,J] > self.ALATENT: 
                self.DELH[I,J]=self.ALATENT
              if self.DELH[I,J] <= 0.0: 
                self.DELH[I,J]=0.0
              self.EPSI[I,J]=self.DELH[I,J]/self.ALATENT
      
      #CONVERGENCE CRITERIA.          
      TMX = np.amax(abs(self.T[:,:]))  
      UMX = np.amax(abs(self.U[:,:]))  
      VMX = np.amax(abs(self.V[:,:]))
        
      DELT = np.amax(abs(self.T[:,:]-self.TOLD[:,:]))  
      DELU = np.amax(abs(self.U[:,:]-self.UOLD[:,:]))  
      DELV = np.amax(abs(self.V[:,:]-self.VOLD[:,:]))  
      
      if TMX > 0:  
        DELTMX=DELT/TMX  
      else:   
        DELTMX=0.    
       
      if UMX > 0:  
        DELUMX=DELU/UMX  
      else:   
        DELUMX=0.0  
        
      if VMX > 0:  
        DELVMX=DELV/VMX  
      else:   
        DELVMX=0.0  
      
      if self.TIME < self.TLAST:  
        print 'TIME=',self.TIME,'ITERL=',self.ITERL,'TMX=',TMX,'UMX=',UMX,'VMX=',VMX
        print 'DELTMX=',DELTMX,'DELUMX=',DELUMX,'DELVMX=',DELVMX   
      #print DELT,DELU,DELV,self.ITERL
      if DELT < self.ERT and DELU < self.ERU and DELV < self.ERV:  
        self.LCONV=True  
          
      if not(self.LCONV):  
        if DELTMX> self.ERT or DELUMX>self.ERU or DELVMX>self.ERV:
          if self.ITERL==self.ISTP:
            print 'EXCEEDED MAX. NO. OF ITERATIONS'
            self.LCONV=True
          else:
            self.ITERL=self.ITERL+1   
            for I in range(1,self.L1):   
              for J in range(1,self.M1):   
                self.UOLD[I,J]=self.U[I,J]   
                self.VOLD[I,J]=self.V[I,J]
            self.bound()   

    self.ITLL=self.ITLL+1     
    self.TIME=self.TIME+self.DT  

    if self.ITLL%4 == 0:
      self.vect_plot() 
      self.printout() 
    
    self.BBK=self.TIME % float(self.NTIME)   
    if self.BBK<0.00001:   
      self.ITER=self.ITER+1   
      self.bound()   

    if self.TIME >= self.TLAST:
      self.LSTOP=True   
    for I in range(1,self.L1):   
      for J in range(1,self.M1):  
        self.UOLD[I,J]=self.U[I,J]   
        self.VOLD[I,J]=self.V[I,J]
  
  def vect_plot(self):  
    """
    THIS FUNCTION FOR VECTOR PLOT
    """ 
    #INTERNAL GRID POINTS 
    for I in range(1,self.L2):   
      for J in range(1,self.M2): 
        self.Ur[I,J]=(self.U[I,J]+self.U[I+1,J])/2.0 
        self.Vr[I,J]=(self.V[I,J]+self.V[I,J+1])/2.0 
     
    #LEFT BOUNDARY (EXCLUDING CORNER GRIDS) 
    for J in range(1,self.M2):
      self.Ur[0,J]=self.U[1,J] 
      self.Vr[0,J]=(self.V[0,J]+self.V[0,J+1])/2.0 
     
    #BOTTOM BOUNDARY (EXCLUDING CORNER GRIDS) 
    for I in range(1,self.L2):
      self.Ur[I,0]=(self.U[I,0]+self.U[I+1,0])/2.0 
      self.Vr[I,0]=self.V[I,1] 
     
    #RIGHT BOUNDARY (EXCLUDING CORNER POINTS) 
    for J in range(1,self.M2):
      self.Ur[self.L1-1,J]=self.U[self.L1-1,J] 
      self.Vr[self.L1-1,J]=(self.V[self.L1-1,J]+self.V[self.L1-1,J+1])/2.0 
     
    #TOP BOUNDARY (EXCLUDING CORNER POINTS) 
    for I in range(1,self.L2):
      self.Ur[I,self.M1-1]=(self.U[I,self.M1-1]+self.U[I+1,self.M1-1])/2.0 
      self.Vr[I,self.M1-1]=self.V[I,self.M1-1] 

    #ALL CORNER POINTS 
    #LEFT  BOTTOM CORNER POINT 
    self.Ur[0,0]=self.U[1,0] 
    self.Vr[0,0]=self.V[0,1] 
     
    #LEFT TOP CORNER POINT 
    self.Ur[0,self.M1-1]=self.U[1,self.M1-1] 
    self.Vr[0,self.M1-1]=self.V[0,self.M1-1] 
     
    #RIGHT BOTTOM CORNER POINT 
    self.Ur[self.L1-1,0]=self.U[self.L1-1,0] 
    self.Vr[self.L1-1,0]=self.V[self.L1-1,1] 
     
    #RIGHT TOP CORNER POINT 
    self.Ur[self.L1-1,self.M1-1]=self.U[self.L1-1,self.M1-1] 
    self.Vr[self.L1-1,self.M1-1]=self.V[self.L1-1,self.M1-1] 

 
  def printout(self): 
    """
    THIS FUNCTION DIRECTLY GIVES MATLAB PLOTS OF PROBLEM VARIABLES 
    """ 

    FILENAME = "t0"+str(int(self.TIME))+".m"

    f = open(FILENAME,'w')  
    #f.write("CURRENT TIME IS"+str(int(TIME))+"\n") 
    f.write("x=["+"\n")  
    for I in range(0,self.L1):
      f.write(str(self.X[I])+"\n")        
    f.write("];"+"\n") 
    f.write("y=["+"\n")  
    
    for J in range(0,self.M1):
       f.write(str(self.Y[J])+"\n")        
    f.write("];"+"\n")  
    f.write("t=["+"\n")  

    for J in range(0,self.M1): 
      for I in range(0,self.L1):
        f.write(str(self.T[I,J])+" ")
      f.write(";"+"\n") 
    f.write("];"+"\n")  
    f.write("pcolor(x,y,t);"+"\n") 
    f.write("shading interp"+"\n") 
    f.write("colorbar"+"\n") 
    f.write("xlabel('Distance along X-axis (m)')"+"\n")
    f.write("ylabel('Distance along Y-axis (m)')"+"\n") 
    f.write("grid on"+"\n")
    f.write("axis([	0 "+str(self.XL)+" 0 "+str(self.YL)+" ])"+"\n") 
    f.write("title('Isotherms plot');"+"\n") 

    f.write( "u=["+"\n")   
    for J in range(0,self.M1): 
      for I in range(0,self.L1):
        f.write(str(self.Ur[I,J])+" ")
      f.write(";"+"\n")  
    f.write("];"+"\n")   
    f.write( "v=["+"\n")   
   
    for J in range(0,self.M1): 
      for I in range(0,self.L1):
        f.write(str(self.Vr[I,J])+" ")
      f.write(";"+"\n")  
    f.write("];"+"\n")
    f.write("figure(2);"+"\n")
	
    f.write("quiver(x,y,u,v)" +"\n")
    f.write("xlabel('Distance along X-axis (m)')"+"\n")  
    f.write("ylabel('Distance along Y-axis (m)')"+"\n")  
    f.write("grid on"+"\n") 
    f.write("axis([	0 "+str(self.XL)+" 0 "+str(self.YL)+"])"+"\n") 
    f.write("title('Velocity vector plot');"+"\n")  


    f.write( "epsi=["+"\n")  

    for J in range(0,self.M1): 
      for I in range(0,self.L1):
        f.write(str(self.EPSI[I,J])+" ")
      f.write(";"+"\n")
    f.write("];"+"\n")  
    f.write("figure(3);"+"\n")
    f.write("pcolor(x,y,epsi);"+"\n") 
    f.write("shading interp"+"\n") 
    f.write("colorbar"+"\n") 
    f.write("xlabel('Distance along X-axis (m)')"+"\n") 
    f.write("ylabel('Distance along Y-axis (m)')"+"\n") 
    f.write("grid on"+"\n") 
    f.write("axis([	0 "+str(self.XL)+" 0 "+str(self.YL)+" ])"+"\n") 
    f.write("title('EPSI plot');"+"\n")
    f.write("pause();"+"\n")     
    f.close()  

  def bound(self):
    """
    THIS FUNCTION GIVES BOUNDARY CONDITIONS FOR THE PROBLEM
    """   
    #VELOCITY BOUNDARY CONDITIONS
    for J in range(0,self.M1):
       self.U[1,J]=0.0   #LEFT FACE
       self.U[self.L1-1,J]=0.0  #RIGHT FACE
    for J in range(1,self.M1):
       self.V[0,J]=0.0   #LEFT FACE 
       self.V[self.L1-1,J]=0.0  #RIGHT FACE 
    for I in range(1,self.L1):
       self.U[I,0]=0.0  #BOTTOM FACE 
       self.U[I,self.M1-1]=0.0 #TOP FACE  
    for I in range(0,self.L1):
       self.V[I,1]=0.0  #BOTTOM FACE
       self.V[I,self.M1-1]=0.0 #TOP FACE
        
    #TEMPERATURE BOUNDARY CONDITIOMS 
    for J in range(0,self.M1):
       self.T[0,J]= self.THOT #LEFT FACE
       self.T[self.L1-1,J]= self.TINITIAL
    for I in range(0,self.L1):
       self.T[I,0]=self.T[I,1]   #BOTTOM FACE
       self.T[I,self.M1-1]=self.T[I,self.M2-1] #TOP FACE

    #LIQUID FRACTION BOUNDARY CONDITIOMS 
    for J in range(0,self.M1):
       self.EPSI[0,J]=1 #LEFT FACE
       self.EPSI[self.L1-1,J]= 0 #RIGHT FACE
    for I in range(0,self.L1):
       self.EPSI[I,0]=self.EPSI[I,1]   #BOTTOM FACE
       self.EPSI[I,self.M1-1]=self.EPSI[I,self.M2-1] #TOP FACE

if  __name__ =='__main__': 
  grid={'xLength':8.89E-2, 
        'yLength':6.35E-2, 
        'xGrids':42, 
        'yGrids':32}  
                
  material={'darcyConstant':1.6E6,
            'density':6093.0,   
            'conductivity':32.0,
            'meltingTemp':29.78,
            'specificHeatCapacity':381.5,
            'latentHeat':80160.0,
            'thermalExpansionCoeff':1.2E-4,
            'viscosity':1.81E-3,
            'referenceDensity':6095.0}
  
  ics={'TINITIAL':28.3} 
  bcs={'THOT':38} 
  times={'timeStep':5,
    'maxIter':1000,
    'TimeLast':600.0}
  a = SolidiPy(grid,material,ics,bcs,times)
  a.grid_initialize()
  a.geometry_initialize()
  a.start()
  while not a.LSTOP:  
    a.dense()  
    a.bound()   
    a.oldval() 
    a.coeff()  
  print("PROGRAM FINISHED")   
