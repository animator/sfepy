import sys
sys.path.append( '.' )

import os
cwd = os.path.split(os.path.join(os.getcwd(), __file__))[0]
output_dir = os.path.join(cwd, 'output')
if not os.path.exists(output_dir):
	os.makedirs(output_dir)	

def define():           
	materials = {
    'gallium':({
		'darcyConstant':1.6E6,
    	'density':6093.0,   
    	'conductivity':32.0,
    	'meltingTemp':29.78,
    	'specificHeatCapacity':381.5,
    	'latentHeat':80160.0,
        'thermalExpansionCoeff':1.2E-4,
        'viscosity':1.81E-3,
        'referenceDensity':6095.0,
		},),
	}

	fields = {
	    'velocity': ('real', 'vector', 'Omega', 2),
	    'pressure': ('real', 'scalar', 'Omega', 1),
	    'temperature': ('real', 'scalar', 'Omega', 1),
	    'liquidfraction': ('real', 'scalar', 'Omega', 1),
	}

	variables = {
	    'u' : ('unknown field', 'velocity', 0),
	    'p' : ('unknown field', 'pressure', 1),
	    'T' : ('unknown field', 'temperature', 2),
	    'epsi' : ('unknown field', 'liquidfraction', 3),
	}

	grid={'xLength':9.00E-2, 
      'yLength':6.35E-2, 
      'xGrids':42, 
      'yGrids':32} 

	tinit = {'T_INITIAL':28.3} 

	tbcs = {'T_HOT':38}

	times={'steps':({'starttime':0,'timeStep':5,'maxIter':20},
	            {'starttime':40,'timeStep':10,'maxIter':100},
	            {'starttime':150,'timeStep':10,'maxIter':1000}),               
	  		'timeLast':600.0}
	return locals()

def main():
	from sfepy.base.base import output, Struct
	from sfepy.base.conf import ProblemConf, get_standard_keywords
	from sfepy.fem import ProblemDefinition
	from sfepy.fem.mesh import Mesh
	from sfepy.fem.meshio import MeshIO, VTKMeshIO
	from pcsolver import PhaseChangeSolver
	from sfepy.fem.variables import Variable 
	
	output.prefix = 'phasechange:'
	conf = ProblemConf.from_file(__file__)
	a = PhaseChangeSolver(conf)
	a.grid_initialize()
	a.geometry_initialize()
	name = os.path.join(output_dir, "pcsolver")
	coors, ngroups, conns, mat_ids, descs=a.get_mesh_vars()
	mesh =  Mesh.from_data(name,coors, ngroups, conns, mat_ids, descs)
	mesh.write()
	a.start()
	while not a.time_over():  
		a.dense()  
		a.bound()   
		a.oldval() 
		prntout = a.coeff()
		output(prntout) 
		time = str(int(a.get_current_time()*100))
		time = "0"*(7-len(time)) + time	
		epsi=a.get_field('epsi')
		u=a.get_field('u')
		T=a.get_field('T')
		vtkmesh = VTKMeshIO(mesh.name)
		vtkmesh.write(mesh.name+"_"+time+".vtk",mesh,out={"epsi":epsi,
															"u":u,
															"T":T})
		a.time_update()
	output('results saved in %s' %(output_dir))

if __name__ == '__main__':
    main()