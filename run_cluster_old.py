import fnmatch
import numpy
import os
import sys
import time as cpu_time

from amuse.lab import nbody_system, units, Particles, zero
from amuse.lab import new_kroupa_mass_distribution, new_plummer_model, write_set_to_file
from amuse.community.ph4.interface import Ph4
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.community.fractalcluster.interface import new_fractal_cluster_model

from src.make_jumbos import make_isolated_jumbos
from src.data_process import DataProcess

class RunCode(object):
    def __init__(self, mmin, mmax, Nstars, Njumbos, Nff,
                 Rvir, Fd, Qvir, data_direc, cluster_prof):

        self.pos_distr = cluster_prof
        self.mmin = mmin
        self.mmax = mmax
        self.Nstars = Nstars
        self.Njumbos = Njumbos
        self.Nff = Nff
        self.Rvir = Rvir
        self.Fd = Fd
        self.Qvir = Qvir

        self.event_key = [ ]
        self.event_mass = [ ]
        self.event_time = [ ]
        self.event_type = [ ]

        self.init_cpu_time = cpu_time.time()
        self.data_direc = data_direc
    
    def ang_cons(self, particles):
        ang = particles.total_angular_momentum()
        A = (ang[0]**2+ang[1]**2+ang[2]**2)**0.5

        return A

    def ZAMS_radius(self, mass):
        """Define particle radius"""

        log_mass = numpy.log10(mass.value_in(units.MSun))
        mass_sq = (mass.value_in(units.MSun))**2
        alpha = 0.08353 + 0.0565*log_mass
        beta  = 0.01291 + 0.2226*log_mass
        gamma = 0.1151 + 0.06267*log_mass
        r_zams = pow(mass.value_in(units.MSun), 1.25) \
                *(0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)

        return r_zams | units.RSun

    def merge_two_stars(self, bodies, particles_in_encounter):
        """Function to form a merger remnant"""

        com_pos = particles_in_encounter.center_of_mass()
        com_vel = particles_in_encounter.center_of_mass_velocity()
        
        new_particle=Particles(1)
        new_particle.mass = particles_in_encounter.total_mass()
        new_particle.position = com_pos
        new_particle.velocity = com_vel
        new_particle.radius = particles_in_encounter.radius.sum()
        self.event_key.append(particles_in_encounter.key)
        self.event_mass.append(particles_in_encounter.mass)
        if "JuMBOs" in particles_in_encounter.name:
            print("Merge with JuMBOs")
            new_particle.name = "JuMBOs"
            self.event_type.append("JuMBOs Merger")
        else:
            print("Merge two stars")
            new_particle.name = "star"
            self.event_type.append("Stellar Merger")
        new_particle.Nej = 0
        bodies.add_particles(new_particle)
        bodies.remove_particles(particles_in_encounter)

    def resolve_collision(self, coll_det, gravity, bodies):
        """Function to resolve mergers"""

        if coll_det.is_set():
            print("...Detection: Merger Event...")
            E_coll = gravity.kinetic_energy + gravity.potential_energy
            print("At time=", gravity.model_time.in_(units.Myr), \
                  "number of encounters=", len(coll_det.particles(0)))

            Nenc = 0
            for ci in range(len(coll_det.particles(0))): 
                particles_in_encounter \
                    = Particles(particles=[coll_det.particles(0)[ci],
                                           coll_det.particles(1)[ci]])
                particles_in_encounter \
                    = particles_in_encounter.get_intersecting_subset_in(bodies)

                self.merge_two_stars(bodies, particles_in_encounter)
                bodies.synchronize_to(gravity.particles)
                Nenc += 1
                self.event_time.append(gravity.model_time.in_(units.kyr))
                print("Resolve encounter Number:", Nenc)
            dE_coll = E_coll - (gravity.kinetic_energy + gravity.potential_energy)
            print("dE_coll =", dE_coll, "N_enc=", Nenc)
        sys.stdout.flush()

    def indiv_PE(self, particle, pset):        
        PE_all = pset.potential_energy()
        PE_m = pset[pset.key != particle.key].potential_energy()
        PE = (PE_all-PE_m)
        return PE

    def resolve_ejection(self, gravity, bodies, cluster_lim):
        """Function to resolve ejections. Remove particle if triggered."""

        particle_set = bodies.copy_to_memory()
        rel_vel = particle_set.velocity - particle_set.center_of_mass_velocity()
        rel_pos = particle_set.position - particle_set.center_of_mass()

        ejec = False
        for parti_ in range(len(particle_set)):
            if particle_set[parti_].Nej == 0:
                vel_vect = numpy.sqrt(numpy.dot(rel_vel[parti_], rel_vel[parti_]))
                dis_vect = numpy.sqrt(numpy.dot(rel_pos[parti_], rel_pos[parti_]))
                traj = (numpy.dot(rel_vel[parti_], rel_pos[parti_]))
                traj /= (vel_vect*dis_vect)
                if traj > 0 and rel_pos[parti_].length() > cluster_lim:
                    PE = self.indiv_PE(particle_set[parti_], particle_set)
                    KE = 0.5*particle_set[parti_].mass*rel_vel.length()**2
                    if KE > abs(PE):
                        particle_set[parti_].Nej = 1
                        self.event_key.append(particle_set[parti_].key)
                        self.event_mass.append(particle_set[parti_].mass)
                        self.event_time.append(gravity.model_time.in_(units.kyr))
                        self.event_type.append("Ejection")
                        Ef = bodies.potential_energy() + bodies.kinetic_energy()
                        ejec = True

        return ejec

    def make_initial_cluster(self, config, isol_choice):
        
        N = self.Nstars + self.Njumbos
        if (isol_choice):
            N += self.Nff
            self.isol_choice = True
        else:
            self.isol_choice = False

        Mmin = self.mmin
        Mmax = self.mmax
        masses = new_kroupa_mass_distribution(N, mass_min=Mmin, mass_max=Mmax)
        Mtot_init = masses.sum()
        converter = nbody_system.nbody_to_si(Mtot_init, self.Rvir)
        
        if self.pos_distr.lower() == "fractal":
            bodies = new_fractal_cluster_model(N, fractal_dimension=self.Fd, 
                                               convert_nbody=converter)
        else:
            bodies = new_plummer_model(N, convert_nbody=converter)

        bodies.mass = masses
        bodies.name = "star"
        bodies.radius = self.ZAMS_radius(bodies.mass)
        if self.Njumbos>0:
            JuMBOs = bodies.random_sample(self.Njumbos)
            JuMBOs.name = "JuMBOs"

            mvals = numpy.random.uniform(0.0006, 0.013, self.Njumbos) 
            JuMBOs.mass = mvals * (1 | units.MSun)

            bodies.scale_to_standard(convert_nbody=converter, 
                                     virial_ratio = self.Qvir)
            bodies.move_to_center()
            jumbos = make_isolated_jumbos(bodies, config, self.data_direc)
            bodies.remove_particles(JuMBOs)
            bodies.add_particles(jumbos)

        if (isol_choice):
            FreeFloaters = bodies[bodies.name != "JuMBOs"].random_sample(self.Nff)
            FreeFloaters.name = "FF"

            mvals = numpy.random.uniform(0.0006, 0.013, self.Nff)
            FreeFloaters.mass = mvals * (1 | units.MSun)
            rvals = (FreeFloaters.mass/(1 | units.MJupiter))**(1/3)
            FreeFloaters.radius =  rvals * (1 | units.RJupiter) 

            Mtot_init = bodies.mass.sum()
            converter = nbody_system.nbody_to_si(Mtot_init, self.Rvir)
            bodies.scale_to_standard(convert_nbody=converter, 
                                     virial_ratio = self.Qvir)
            bodies.move_to_center()

        return bodies
            
    def run_cluster(self, bodies, t_end, dt, code_dt, NSim):
        data_manip = DataProcess(self.data_direc)
        self.data_manip = data_manip
        direc = str(self.data_direc)+"simulation_snapshot/config_"+str(NSim)+"/"

        stars = bodies[bodies.name=="star"]
        hosts = bodies[bodies.name=="host"]
        jumbos = bodies-stars-hosts

        nStar = len(stars)
        nJuMBO = len(jumbos)

        converter=nbody_system.nbody_to_si(1 | units.Myr, bodies.mass.sum())
        gravity = Ph4(converter, number_of_workers=4)
        gravity.parameters.timestep_parameter = code_dt
        gravity.parameters.epsilon_squared = 0. | units.au**2
        gravity.particles.add_particles(bodies)
        coll_det = gravity.stopping_conditions.collision_detection
        coll_det.enable()

        channel_from_gd = gravity.particles.new_channel_to(bodies)
        channel_to_gd = bodies.new_channel_to(gravity.particles)
        index =0
        filename = "jumbos_snapshot_step"+str(index)+".amuse"
        write_set_to_file(bodies.savepoint(0|units.Myr), direc+filename, 'hdf5',
                          close_file=True, overwrite_file=True)
        
        A0 = self.ang_cons(bodies)
        E0 = bodies.kinetic_energy() + bodies.potential_energy()

        Qinit = -gravity.kinetic_energy/gravity.potential_energy
        rvirial = bodies.virial_radius().in_(units.pc)
        data_manip.init_arr(gravity, bodies, rvirial)
        print("Initial Q:     ", Qinit)
        print("Initial rvir:  ", rvirial.in_(units.pc))
        print("Min stellar body: ", min(bodies[bodies.name == "star"].mass))
        time = zero
        index = 0
        dt_diag = min(dt, 0.1|units.Myr)
        diag_time = time
        bodies.Nej = 0
        while time < t_end:
            time += dt

            channel_to_gd.copy_attributes(["mass", "radius", "vx", "vy", "vz"])

            gravity.evolve_model(time)
            self.resolve_collision(coll_det, gravity, bodies)

            channel_from_gd.copy()
            cluster_lim = 0.1*numpy.sqrt(nStar)*LagrangianRadii(bodies)[-2]
            self.resolve_ejection(gravity, bodies, cluster_lim)
                
            At = self.ang_cons(bodies)
            Et = bodies.potential_energy() + bodies.kinetic_energy()

            dA = abs(At-A0)/At
            dE = abs(E0-Et)/Et

            time = gravity.model_time

            sys.stdout.flush()
            if diag_time <= time:
                index += 1
                diag_time += dt_diag
                filename = "jumbos_snapshot_step"+str(index)+".amuse"
                write_set_to_file(bodies.savepoint(time), direc+filename, 'hdf5',
                                  close_file=True, overwrite_file=False)
            #Storing data    
            data_manip.append_data(gravity, bodies, cluster_lim, dE, dA)
        
        Qfinal = -(gravity.kinetic_energy/gravity.potential_energy)
        data_manip.lagrangian_data(data_manip.time_arr, data_manip.lag25_arr, 
                                    data_manip.lag50_arr, data_manip.lag75_arr, 
                                    data_manip.cejec_lim, NSim)
        data_manip.event_data(self.event_time, self.event_key, 
                              self.event_type, self.event_mass, 
                              NSim)
        data_manip.energy_data(data_manip.time_arr[1:], data_manip.dE, 
                               data_manip.dA, NSim)
        write_set_to_file(bodies.savepoint(time), direc+filename, 'hdf5',
                          close_file=True, overwrite_file=True)

        gravity.stop()
        end_time = cpu_time.time()
        #Logging initial conditions and Final Statistics
        cpu_time_min = (end_time-self.init_cpu_time)/60
        lines = ["Total CPU Time: {} minutes".format(cpu_time_min),
                "End Time: {}".format(t_end.in_(units.kyr)), 
                "Time step: {}".format(dt.in_(units.kyr)),
                "Initial Rvirial: {}".format(rvirial.in_(units.pc)), 
                "Cluster Distribution: {}".format(self.pos_distr),
                "Isolated Jupiters: {}".format(self.isol_choice), 
                "If True -> Nff: {}".format(self.Nff),
                "Init No. Stars: {}".format(nStar), 
                "Init No. JuMBOs: {}".format(nJuMBO),
                "Pos. Distribution: {}".format(self.pos_distr),
                "Stellar Mass range: [{}, {}] ".format(self.mmin.in_(units.MSun), self.mmax.in_(units.MSun)),
                "Initial Q: {}".format(Qinit), 
                "Final Q: {}".format(Qfinal)]

        with open(os.path.join(data_manip.path+str('simulation_stats'),
                                'simulation_stats_config_'+str(NSim)+'_2.0.txt'), 'w') as f:
            for line_ in lines:
                f.write(line_)
                f.write('\n')
    
def new_option_parser():
    cluster_choices = ["Plummer", "Fractal"]
    virial_radius = [0.5 | units.pc, 1.0 | units.pc]

    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("--Nstars", dest="Nstars", type="int", default = 25,
                      help="number of stars [%default]")
    result.add_option("--mmin", dest="mmin", type="float", 
                      default = 0.08 | units.MSun,
                      help="Minimum stellar mass [%default]")
    result.add_option("--mmax", dest="mmax", type="float", default = 30 | units.MSun,
                      help="Maximum stellar mass [%default]")
    result.add_option("--NJuMBOs", dest="Njumbos", type="int", default = 20,
                      help="number of JuMBOs [%default]")
    result.add_option("--Nff", dest="Nff", type="int",  default = 0,
                      help="number of Free Floaters [%default]")
    result.add_option("--Cdistr", dest="Cdistr", type="string", default = cluster_choices[1],
                      help="fractal dimension [%default]")
    result.add_option("--F", dest="Fd", type="float", default = 1.6,
                      help="fractal dimension [%default]")
    result.add_option("--Q", dest="Qvir", type="float", default = 0.5,
                      help="virial ratio [%default]")
    result.add_option("--dt", unit=units.Myr, dest="dt", type="float", default = 0.01 | units.Myr,
                      help="output timesteps [%default]")
    result.add_option("--R", unit=units.parsec, dest="Rvir", type="float", default = virial_radius[0],
                      help="cluster virial radius [%default.value_in(units.parsec)]")
    result.add_option("--t", unit=units.Myr, dest="t_end", type="float", default = 1.0 | units.Myr,
                      help="end time of the simulation [%default.value_in(units.Myr]")
    result.add_option("--code_dt", dest="code_dt", type="float", default=0.01, 
                      help="gravitational integrator internal time step [%default]")
    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    
    isol_jupiters = [True, False]
    isol_choice = isol_jupiters[0]

    data_direc = "data/Simulation_Data/"+str(o.Cdistr)+"_rvir"+str(o.Rvir.number)
    if (isol_choice):
        data_direc += "_FF/"
    else:
        data_direc += "/"
    
    for k_ in range(5):
       dir_count = len(fnmatch.filter(os.listdir(str(data_direc)+"simulation_snapshot/"), "*"))
       os.mkdir(str(data_direc)+"simulation_snapshot/config_"+str(dir_count)+"/")

       print("...Running configuration...")
       print(o.Cdistr, " distribution")
       print("Virial Radius: ", o.Rvir)
       print("Free floaters: ", isol_choice)
       sim = RunCode(o.mmin, o.mmax, o.Nstars, o.Njumbos, o.Nff, o.Rvir , o.Fd, 
                     o.Qvir, data_direc, o.Cdistr)
       bodies = sim.make_initial_cluster(dir_count, isol_choice)
       sim.run_cluster(bodies, o.t_end, o.dt, o.code_dt, dir_count)
    
