from pcsolver import SolidiPy

grid={'xLength':9.00E-2, 
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
times={'steps':({'starttime':0,'timeStep':5,'maxIter':40},
                {'starttime':40,'timeStep':10,'maxIter':100},
                {'starttime':150,'timeStep':10,'maxIter':1000}),               
      'timeLast':600.0,
      'printInterval':20}

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