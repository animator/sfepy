#!/usr/bin/env python
"""
First solve the stationary electric conduction problem. Then use its
results to solve the evolutionary heat conduction problem.

Run this example as on a command line::

    $ python <path_to_this_file>/convective_diffusive.py
"""
import sys
sys.path.append( '.' )
import os

from sfepy import data_dir

filename_mesh = data_dir + '/meshes/2d/rectangle_fine_quad.mesh'

cwd = os.path.split(os.path.join(os.getcwd(), __file__))[0]

options = {
    'absolute_mesh_path' : True,
    'output_dir' : os.path.join(cwd, 'output')
}

regions = {
    'Omega' : ('all', {}),
    'Surface' : ('nodes of surface', {}),
    'Wall' : ('nodes in (x > 4.999)', {}),
    'Entry' : ('nodes in (y < -9.999)', {}),
    'Top' : ('nodes in (y > 9.999)', {}),
    'Driven' : ('nodes in (x < -4.999)', {}),
}

materials = {
    'm' : ({
        'thermal_diffusivity' : 2.0,
        'viscosity' : 1.00e-2,
    },),
}

# The fields use the same approximation, so a single field could be used
# instead.
fields = {
    'velocity': ('real', 'vector', 'Omega', 2),
    'pressure': ('real', 'scalar', 'Omega', 1),
    'temperature': ('real', 'scalar', 'Omega', 1),
}

variables = {
    'u' : ('unknown field', 'velocity', 0),
    'v' : ('test field', 'velocity', 'u'),
    'p' : ('unknown field', 'pressure', 1),
    'q' : ('test field', 'pressure', 'p'),
    'T' : ('unknown field', 'temperature', 2),
    's' : ('test field', 'temperature', 'T'),
    'u_known' : ('parameter field', 'velocity', '(set-to-None)'),
}

ebcs = {
    'Driven' : ('Driven', {'T.0' : 40, 'u.1' : 0.1, 'u.0' : 0.0}),
    'Wall' : ('Wall', {'T.0' : 40, 'u.all' : 0.0}),
    'Entry' : ('Entry', {'T.0' : 20}),
}

integrals = {
    'i1' : ('s', 1),
}

equations = {
    'balance' :
    """+ dw_div_grad.5.Omega(m.viscosity, v, u)
       + dw_convect.5.Omega(v, u)
       - dw_stokes.5.Omega(v, p) = 0""",

    'incompressibility' :
    """dw_stokes.5.Omega(u, q) = 0""",

    'Laplace equation' :
    """dw_laplace.5.Omega( m.thermal_diffusivity, s, T )
     + dw_convect_v_grad_s.5.Omega( s, u_known, T )
     =0""",
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max'      : 15,
        'eps_a'      : 1e-10,
        'eps_r'      : 1.0,
        'problem'   : 'nonlinear',
    }),
}

def main():
    from sfepy.base.base import output
    from sfepy.base.conf import ProblemConf, get_standard_keywords
    from sfepy.fem import ProblemDefinition

    output.prefix = 'convdiff:'

    required, other = get_standard_keywords()
    conf = ProblemConf.from_file(__file__, required, other)

    problem = ProblemDefinition.from_conf(conf, init_equations=False)

    # Setup output directory according to options above.
    problem.setup_default_output()

    # First solve the stationary electric conduction problem.
    problem.set_equations({'balance' : conf.equations['balance'],
                           'incompressibility' : conf.equations['incompressibility']})
    problem.time_update()
    flow = problem.solve()
    out = flow.create_output_dict()

    # Then solve the evolutionary heat conduction problem, using state_el.
    problem.clear_equations()
    problem.set_equations({'Laplace equation'
                           : conf.equations['Laplace equation']})
    problem.time_update()
    u_var = problem.get_variables()['u_known']
    u_var.set_data(flow.get_parts()['u'])
    final = problem.solve()
    out.update(final.create_output_dict())

    problem.save_state(problem.get_output_name(), out=out)

    output('results saved in %s' % problem.get_output_name())

if __name__ == '__main__':
    main()
