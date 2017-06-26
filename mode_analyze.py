from scipy.interpolate import InterpolatedUnivariateSpline as IUS
# import mesa_reader

import subprocess
import astropy.units as u
import cgs_const as cgs
from astropy.io import ascii
import numpy as np
import math

import mesa_reader

def prep_mesa(base):
	'''
	Prepare mesa model for use with the ModeAnalyzer class below

	Explicitly put units in...
	'''
	ld=mesa_reader.MesaLogDir(base+'/LOGS')
	pidx=mesa_reader.MesaProfileIndex(base+'/LOGS/profiles.index')
	idx_last=pidx.model_numbers[-1]
	prof=ld.profile_data(model_number=idx_last)

	prof.R=(prof.R*u.R_sun).cgs
	prof.mass=(prof.mass*u.M_sun).cgs

	prof.Rho=prof.Rho*u.g*u.cm**-3.
	prof.P=prof.P*u.g*u.cm**-1.*u.s**-2.
	return prof


def bash_command(cmd):
	'''Run command from the bash shell'''
	process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	return process.communicate()[0]

def to_str_vec(v):
	out="{"+str(v[0])
	for i in range(1, len(v)):
		out=out+","+str(v[i])
	out=out+"}"
	return out

def I(y, l, m):
	y_str=to_str_vec(y)
	tmp=bash_command('./IApprox "{0}" {1} {2}'.format(y_str, int(l), int(m))).lstrip('List(')
	tmp=tmp.replace(')', '')
	tmp=tmp.strip()
	tmp=np.array(tmp.split(',')).astype(float)
	return tmp
	
def Wlm(l, m):
	if ((l+m)/2.).is_integer():
		return (2.**(1 - l)*np.sqrt(np.pi)*np.sqrt((math.factorial(l - m)*math.factorial(l + m))/(1 + 2*l)))/(math.factorial((l - m)/2.)*math.factorial((l + m)/2.))
	else:
		return 0.

def eta(capt_params):
	ms=capt_params['ms']
	mc=capt_params['mc']
	lam=capt_params['lam']
	
	return (ms/(mc+ms))**0.5*(lam*(mc/ms)**(1./3.))**1.5


def get_mode_info(mode_file, dens):
	'''
	Extract frequency and multipole order of the mode from GYRE output. Second argument is
	intepolation function, which contains the the density profile of the model. 
	'''
	mode_dict={}
	mode_info=ascii.read(mode_file,header_start=1, data_end=3)
	mode_dict['omega']=float(mode_info['Re(omega)'])
	assert mode_info['Im(omega)']==0.
	mode_dict['l']=float(mode_info['l'])
	 
	dat_mode=ascii.read(mode_file, data_start=5, header_start=4)
	xs=dat_mode['x']
	rhos=dens(xs)
	xi_r=dat_mode['Re(xi_r)']
	xi_ri=dat_mode['Im(xi_r)']
	#assert np.all(dat_mode['Im(xi_r)']==0.)
	xi_h=dat_mode['Re(xi_h)']
	xi_hi=dat_mode['Im(xi_h)']
	#assert np.all(dat_mode['Im(xi_h)']==0.)
	
	norm1=IUS(xs, rhos*xs**2.*(xi_r**2.+xi_ri**2.)).integral(xs[0], xs[-1])
	norm2=(mode_dict['l']+1.)*mode_dict['l']*IUS(xs, rhos*xs**2.*(xi_h**2.+xi_hi**2.)).integral(xs[0], xs[-1])
	norm=(norm1+norm2)**0.5
	xi_r=xi_r/norm
	xi_h=xi_h/norm
	
	mode_dict['Q']=abs(IUS(xs, xs**2*rhos*mode_dict['l']*(xs**(mode_dict['l']-1.))*(xi_r+(mode_dict['l']+1.)*xi_h)).integral(xs[0], xs[-1]))
	##Definition of Q is ambiguous--should imaginary part be included?? 
	mode_dict['Qi']=abs(IUS(xs, xs**2*rhos*mode_dict['l']*(xs**(mode_dict['l']-1.))*(xi_ri+(mode_dict['l']+1.)*xi_hi)).integral(xs[0], xs[-1]))

	mode_dict['xi_r']=xi_r
	mode_dict['xi_h']=xi_h
	mode_dict['xs']=xs
	
	return mode_dict

class ModeAnalyzer(object):
	def __init__(self, StellarModel, ModeFiles, ModeKeys):
		'''
		Class storing information about stellar oscillation modes for a given stellar 
		model (stored in StellarModel--which must have a density, mass, and sound speed profile...)
		'''
		M=np.max(StellarModel.mass)
		R=np.max(StellarModel.R)

		self.rhos=StellarModel.Rho/(M/R**3.)
		self.rs=StellarModel.R/R
		##Hard-code gamma_ad=5/3 gas for now...
		self.cs=((5./3.)*StellarModel.P/StellarModel.Rho)**0.5/(cgs.G*M/R)**0.5
		order=np.argsort(self.rs)
		self.rhos=self.rhos[order]
		self.cs=self.cs[order]
		self.rs=self.rs[order]

		self.modes_dict={}
		for idx,ff in enumerate(ModeFiles):
			self.modes_dict[ModeKeys[idx]]=get_mode_info(ff, IUS(self.rs, self.rhos))

	def tidal_coupling(self, etas):
		Ts=np.zeros(len(etas))
		for key in self.modes_dict:
			Q=self.modes_dict[key]['Q']
			l=self.modes_dict[key]['l']
			wa=self.modes_dict[key]['omega']
			ys=etas*wa
			for m in range(-int(l), int(l)+1):
				Is=I(ys, l, m)
				Ts+=2.0*np.pi**2.*Q**2.*((Wlm(l,m)/(2.0*np.pi)*2.**1.5*etas)*Is)**2.
		return Ts

	def get_mode_vel(self, key, capt_params, m):
		mode_dict=self.modes_dict[key]
		Q=mode_dict['Q']
		wa=mode_dict['omega']
		l=float(mode_dict['l'])
		xi_r=mode_dict['xi_r']
		xi_h=mode_dict['xi_h']
		xs=mode_dict['xs']
		lam=capt_params['lam']

		eta1=eta(capt_params)
		pre=0.5*np.pi*lam**-3.*Q*(Wlm(l,m)/(2.0*np.pi)*2.**1.5*eta1)*I([eta1*wa], l, m)

		mode_vels=pre*((xi_r**2.+l*(l+1)*xi_h**2.))**0.5
		return xs, mode_vels

	def get_sf_rm(self, mode, capt_params, m, ri=0.99):
	    '''
	    Get shock formation location from Eq. 7 of Ro&Matzner 2017
	    
	    NB by default, the mode is deposited near the outer layers of the star unlike in Ro&Matzner
	    where it is deposited deep inside of the star. xi is the initial radius of the Perturbation
	    '''
	    Lmax=2.0*np.pi*self.rs**2.*self.rhos*self.cs**3.

	    xs2,vs=self.get_mode_vel(mode, capt_params, m)
	    ud=IUS(xs2, vs).derivative(1)(ri)
	    
	    Lmaxi=IUS(self.rs, Lmax)(ri)
	    g=5./3.
	    return self.rs, [IUS(self.rs, -ud*(g+1.)/2.*(Lmax/Lmaxi)**0.5/self.cs).integral(ri, rr) for rr in self.rs]
	


class n32_poly(object):
	def __init__(self):
		dat_poly=np.genfromtxt('poly32.tsv')
		self.R=dat_poly[:,0]
		self.Rho=dat_poly[:,1]
		self.mass=[IUS(self.R, 4.0*np.pi*self.R**2.*self.Rho).integral(self.R[0], rr) for rr in self.R]
		self.cs=(0.53*(5./3.)*(self.Rho/self.Rho[0])**(2./3.))**0.5*(cgs.G*self.mass[-1]/self.R[-1])**0.5
		self.P=self.cs**2.*self.Rho/(5./3.)

class n3_poly(object):
	def __init__(self):
		dat_poly=np.genfromtxt('poly3.tsv')
		self.R=dat_poly[:,0]
		self.Rho=dat_poly[:,1]
		self.mass=[IUS(self.R, 4.0*np.pi*self.R**2.*self.Rho).integral(self.R[0], rr) for rr in self.R]
		self.cs=(0.85*(5./3.)*(self.Rho/self.Rho[0])**(1./3.))**0.5*(cgs.G*self.mass[-1]/self.R[-1])**0.5
		self.P=self.cs**2.*self.Rho/(5./3.)
