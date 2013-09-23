#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

import os
cwd = os.path.split(os.path.join(os.getcwd(), __file__))[0]
output_dir = os.path.join(cwd, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def define():
    materials = {'gallium':({
        'darcyConstant':1.6E6,
        'density':6093.0,
        'conductivity':32.0,
        'meltingTemp':29.78,
        'specificHeatCapacity':381.5,
        'latentHeat':80160.0,
        'thermalExpansionCoeff':1.2E-4,
        'viscosity':1.81E-3,
        'referenceDensity':6095.0,
        }, )}

    fields = {
        'velocity':('real', 'vector', 'Omega', 2),
        'pressure':('real', 'scalar', 'Omega', 1),
        'temperature':('real', 'scalar', 'Omega', 1),
        'liquidfraction':('real', 'scalar', 'Omega', 1),
        }

    variables = {
        'u':('unknown field', 'velocity', 0),
        'p':('unknown field', 'pressure', 1),
        'T':('unknown field', 'temperature', 2),
        'epsi':('unknown field', 'liquidfraction', 3),
        }

    grid = {
        'xLength':9.00E-2,
        'yLength':6.35E-2,
        'xGrids':22,
        'yGrids':16,
        }

    tinit = {'T_INITIAL':28.3}

    tbcs = {'T_HOT':38}

    itertimes = {'maxiterlimit':[{'starttime':0, 'deltatime':5, 'maxIter':20},
                 {'starttime':40, 'deltatime':10, 'maxIter':80},
                 {'starttime':150, 'deltatime':10, 'maxIter':150}],
                 'timeLast':200.0}
    return locals()


required = [
    'field_[0-9]+|fields',
    'tinit',
    'tbcs',
    'variable_[0-9]+|variables',
    'itertimes',
    'material_[0-9]+|materials',
    ]


def main():
    from sfepy.base.base import output, Struct
    from sfepy.base.conf import ProblemConf, get_standard_keywords
    from sfepy.fem.meshio import MeshIO, VTKMeshIO
    from pcsolver import PhaseChangeSolver
    from sfepy.solvers.solvers import Solver
    from sfepy.fem.variables import Variable

    output.prefix = 'phasechange:'
    conf = ProblemConf.from_file(__file__, required=required)
    conf = PhaseChangeSolver.process_conf(conf)
    pcs = PhaseChangeSolver(conf)
    pcs, mesh = pcs(output_dir)
    while not pcs.time_over():
        pcs.dense()
        pcs.bound()
        pcs.oldval()
        prntout = pcs.coeff()
        output(prntout)
        time = str(int(pcs.get_current_time() * 100))
        time = '0' * (7 - len(time)) + time
        epsi = pcs.get_field('epsi')
        u = pcs.get_field('u')
        T = pcs.get_field('T')
        vtkmesh = VTKMeshIO(mesh.name)
        vtkmesh.write(mesh.name + '_' + time + '.vtk', mesh, out={'epsi':epsi,
                      'u':u, 'T':T})
        pcs.time_update()
    output('results saved in %s' % output_dir)


if __name__ == '__main__':
    main()
