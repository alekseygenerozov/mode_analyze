from scipy.interpolate import InterpolatedUnivariateSpline as IUS
# import mesa_reader

import subprocess
import astropy.units as u
import cgs_const as cgs
from astropy.io import ascii
import astropy.constants as const
import numpy as np
import math

import mesa_reader
from scipy.special import airy


def log_interp(x, xs, ys, **kwargs):
	return np.exp(IUS(np.log(xs),np.log(ys), **kwargs)(np.log(x)))
	
def log_integral(x1, x2, xs, ys):
	'''
	Compute \int_{u1}^{u2} e^u y(u) du, where u is log(x), u1=log(x1), and u2=log(x2). 
	y(u) is conputed from an interpolating spline constructed from xs and ys
	'''
	us=np.log(xs)
	return IUS(us, ys*np.exp(us)).integral(np.log(x1), np.log(x2))

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

@np.vectorize
def I2(y, l, m):
	if m==0:
		if l==0:
			return np.pi*(2.**(1./2.)*y)**(-1./3.)*airy((2**(1./2.)*y)**(2./3.))[0]
		elif l==1:
			if y<=4:
	 			return (1.5288 + 0.79192*np.sqrt(y) - 0.86606*y + 0.14593*y**1.5)/(np.exp((2*np.sqrt(2)*y)/3.)*(1 + 1.6449*np.sqrt(y) - 1.2345*y + 0.19392*y**1.5))
			else:
				return (1.4119 + 18.158*np.sqrt(y) + 22.152*y)/(np.exp((2*np.sqrt(2)*y)/3.)*(1 + 12.249*np.sqrt(y) + 28.593*y))
		elif l==2:
			return (np.sqrt(1 + (2*np.sqrt(2)*y)/3.)*(0.78374 + 1.5039*np.sqrt(y) + 1.0073*y + 0.71115*y**1.5))/(np.exp((2*np.sqrt(2)*y)/3.)*(1 + 1.9128*np.sqrt(y) + 1.0384*y + 1.2883*y**1.5))
		elif l==3:
			return ((1 + (2*np.sqrt(2)*y)/3.)*(0.58894 + 0.32381*np.sqrt(y) + 0.45605*y + 0.1522*y**1.5))/\
			(np.exp((2*np.sqrt(2)*y)/3.)*(1. + 0.54766*np.sqrt(y) + 0.7613*y + 0.53016*y**1.5))
		else:
			return (y**2.*I2(y,-4 + l,0))/((-3. + l)*(-2 + 2.*l)) + ((-3 + 2*l)*I2(y,-1 + l,0))/(-2. + 2.*l)
	elif m>0:
		return -((np.sqrt(2)*y*I2(y,-1 + l,-1 + m))/(1.*l)) - I2(y,l,-1 + m) + (2. - (2.*(-1 + m))/(1.*l))*I2(y,1 + l,-1 + m)
	else:
		return (np.sqrt(2.)*y*I2(y,-1 + l,1 + m))/(1.*l) - I2(y,l,1 + m) + (2. + (2.*(1. + m))/(1.*l))*I2(y,1 + l,1 + m)

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
	xi_h=dat_mode['Re(xi_h)']
	xi_hi=dat_mode['Im(xi_h)']
	if np.any(np.abs(xi_ri)>0):
		print np.max(np.abs(xi_ri[1:]/xi_r[1:]))
	if np.any(np.abs(xi_hi)>0):	
		print np.max(np.abs(xi_hi[1:]/xi_h[1:]))
	#assert np.all(dat_mode['Im(xi_h)']==0.)
	# norm1=IUS(xs, rhos*xs**2.*(xi_r**2.+xi_ri**2.)).integral(xs[0], xs[-1])
	# norm2=(mode_dict['l']+1.)*mode_dict['l']*IUS(xs, rhos*xs**2.*(xi_h**2.+xi_hi**2.)).integral(xs[0], xs[-1])
	norm1=IUS(xs, rhos*xs**2.*xi_r**2.).integral(xs[0], xs[-1])
	norm2=(mode_dict['l']+1.)*mode_dict['l']*IUS(xs, rhos*xs**2.*xi_h**2.).integral(xs[0], xs[-1])
	norm=(norm1+norm2)**0.5
	xi_r=xi_r/norm
	xi_h=xi_h/norm
	
	mode_dict['Q']=abs(IUS(xs, xs**2*rhos*mode_dict['l']*(xs**(mode_dict['l']-1.))*(xi_r+(mode_dict['l']+1.)*xi_h)).integral(xs[0], xs[-1]))
	# mode_dict['Q']=np.abs(log_integral(xs[1], xs[-1], xs[1:], xs[1:]**2*rhos[1:]*mode_dict['l']*(xs[1:]**(mode_dict['l']-1.))*(xi_r[1:]+(mode_dict['l']+1.)*xi_h[1:])))
	##Definition of Q is confusing--should imaginary part be included?? 
	mode_dict['Qi']=abs(IUS(xs, xs**2*rhos*mode_dict['l']*(xs**(mode_dict['l']-1.))*(xi_ri+(mode_dict['l']+1.)*xi_hi)).integral(xs[0], xs[-1]))
	assert np.abs(mode_dict['Qi']/mode_dict['Q'])<1.0e-6

	mode_dict['xi_r']=xi_r
	mode_dict['xi_h']=xi_h
	mode_dict['xs']=xs
	
	return mode_dict

class ModeAnalyzer(object):
	def __init__(self, StellarModel, ModeBase, n_min=-18, n_max=5, ls=[2]):
		'''
		Class storing information about stellar oscillation modes for a given stellar 
		model (stored in StellarModel--which must have a density, mass, and sound speed profile...)
		'''
		M=np.max(StellarModel.mass)
		R=np.max(StellarModel.R)
		self.M=M
		self.R=R

		self.rhos=(StellarModel.Rho/(M/R**3.)).cgs
		self.rs=(StellarModel.R/R).cgs
		self.ms=(StellarModel.mass/M).cgs
		##Hard-code gamma_ad=5/3 gas for now...
		self.cs=(((5./3.)*StellarModel.P/StellarModel.Rho)**0.5/(const.G*M/R)**0.5).cgs
		##Escape velocity considering only layers below...
		
		order=np.argsort(self.rs)
		self.rhos=self.rhos[order]
		self.cs=self.cs[order]
		self.rs=self.rs[order]
		self.ms=self.ms[order]
		self.v_esc_o=2.**0.5*(self.ms/self.rs)**0.5
		self.v_esc=2.**0.5*((self.ms/self.rs)+np.array([4.*np.pi*log_integral(rr, self.rs[-1], self.rs, self.rs*self.rhos) for rr in self.rs]))**0.5
		self.etas=np.linspace(0.7, 10, 100)
		self.cache={}

		self.modes_dict={}
		self.ns=range(n_min, n_max+1, 1)
		# for idx,ff in enumerate(ModeFiles):
		for nn in self.ns:
			for ll in ls:
				ff=ModeBase+'{0:+d}.l{1}.txt'.format(nn,ll)
				self.modes_dict[str(nn)+'_'+str(ll)]=get_mode_info(ff, IUS(self.rs, self.rhos))

	def tidal_coupling_alpha(self, key, m):
		'''
		Tidal coupling constant for a particular mode for a fixed grid of etas.
		'''
		keyb='T_{0}_{1}'.format(key, m)
		if not keyb in self.cache:
			Q=self.modes_dict[key]['Q']
			l=self.modes_dict[key]['l']
			wa=self.modes_dict[key]['omega']

			ys=self.etas*wa
			Is=I2(ys, l, m)
			self.cache[keyb]=2.0*np.pi**2.*Q**2.*((Wlm(l,m)/(2.0*np.pi)*2.**1.5*self.etas)*Is)**2.
		return self.cache[keyb]

	def tidal_coupling(self, l):
		'''
		Tidal coupling constant (see e.g. Stone, Kuepper and Ostriker 2016) 
		'''
		if not hasattr(self, 'T_'+str(l)):
			setattr(self, 'T_'+str(l), np.zeros(len(self.etas)))
			T=getattr(self,  'T_'+str(l))
			self.g=np.zeros(len(self.etas))
			self.p=np.zeros(len(self.etas))
			self.f=np.zeros(len(self.etas))
			for nn in self.ns:
				key=str(nn)+'_'+str(l)
				for m in range(-int(l), int(l)+1):
					tmp=self.tidal_coupling_alpha(key, m)
					T+=tmp
					if nn<0:
						self.g=self.g+tmp
					elif nn==0:
						self.f=self.f+tmp
					else:
						self.p=self.p+tmp
		return getattr(self, 'T_'+str(l))


	def en_tot(self, capt_params, lmax=2):
		'''
		Total energy deposited into the star
		'''
		capt_params['ms']=self.M
		q=(capt_params['ms']/capt_params['mc']).cgs.value
		delta_E=np.zeros(len(self.etas))

		for l in range(2, lmax+1):
			##Not sure about extra 1+q in the denominator
			print self.tidal_coupling(l)*(q**-2.)*(q/(1.+q))**((2.*l+2)/3)*(self.etas**(2./3.))**(-2.*l-2.)
			delta_E+=self.tidal_coupling(l)*(q**-2.)*(q/(1.+q))**((2.*l+2)/3)*(self.etas**(2./3.))**(-2.*l-2.)

		return delta_E

	##Initial semi-major axis of orbit (not accounting for mass loss).
	def a0(self, capt_params, mass_loss=True):
		'''
		Getting the initial semi-major axis after tidal capture
		'''
		#Calculate mass lost at pericenter
		delta_m=0.*u.g
		if mass_loss:
			delta_m=self.delta_m(capt_params)*self.M

		mc=capt_params['mc']
		lam=capt_params['lam']
		capt_params['ms']=self.M
		mtot=mc+self.M
		mu=mc*self.M/mtot
		mtot1=mtot-delta_m	
		mu1=mc*(self.M-delta_m)/mtot1

		eta1=eta(capt_params).cgs.value
		Ts=self.T
		T1=log_interp(eta1, self.etas, Ts)

		if 'vinf' in capt_params:
			a0=0.5*((const.G*mtot1*mu1)/(-0.5*mu*capt_params['vinf']**2.+self.en_diss(capt_params)))
		else:
			a0=0.5*((const.G*mtot1*mu1)/(0.5*(const.G*mtot*mu/capt_params['a'])+self.en_diss(capt_params)))

		if a0<0:
			return np.inf*u.cm
		return a0

	##Initial eccentricity of orbit (not accounting for mass loss)
	def e0(self, capt_params, mass_loss=True):
		#Calculate mass lost at pericenter
		delta_m=0.*u.g
		if mass_loss:
			delta_m=self.delta_m(capt_params)*self.M

		##Extract paramaters
		mc=capt_params['mc']
		lam=capt_params['lam']
		capt_params['ms']=self.M
		mtot=mc+self.M
		mu=mc*self.M/mtot	
		#Tidal radius, pericenter, and velocity at pericenter.
		rt=(mc/self.M)**(1./3.)*self.R
		rp=lam*rt
		if 'vinf' in capt_params:
			vp=capt_params['vinf']*(1.+2.*const.G*mtot/rp/capt_params['vinf']**2.)**0.5
		else:
			ecc=1.-rp/capt_params['a']
			vp=(const.G*mtot*(1.+ecc)/rp)**0.5
		#Semi-major axis, total mass, and reduced mass (accounting for mass loss)
		a1=self.a0(capt_params, delta_m)
		mtot1=mtot-delta_m	

		mu1=mc*(self.M-delta_m)/mtot1
		#Use angular momentum conservation to find pericenter of new orbit.
		v1=vp*mc/(self.M+mc)
		r1=rp*mc/(self.M+mc)

		ang_momentum=mu*rp*vp-delta_m*r1*v1
		return (1.-((ang_momentum)**2./(mu1**2.*mtot1*a1*const.G)))**0.5


	def get_mode_vel(self, key, m, capt_params):
		'''
		Shell averaged velocity of a particular mode (specified by key and m). capt_params gives the masses of the two bodies the pericenter.
		'''
		capt_params['ms']=self.M
		q=capt_params['ms']/capt_params['mc']
		mode_dict=self.modes_dict[key]
		l=float(mode_dict['l'])
		xi_r=mode_dict['xi_r']
		xi_h=mode_dict['xi_h']
		eta1=eta(capt_params)
		lam=capt_params['lam']

		Ts=self.tidal_coupling_alpha(key, m)
		if np.all(Ts==0):
			T1=0
		else:
			T1=log_interp(eta1, self.etas, Ts)

		mode_vels=(2.*T1/(4.*np.pi))**0.5*((xi_r**2.+l*(l+1)*xi_h**2.))**0.5*lam**(-(l+1))*(q)**(1-(l+1.)/3.)
		return mode_vels

	def get_mode_vel_tot(self, capt_params, lmax=2):
		'''
		Get total rms velocity adding all of the modes together.
		'''
		capt_params['ms']=self.M
		eta1=eta(capt_params)
		v_mode_2_regrid=0.
		lam=capt_params['lam']
		for key in self.modes_dict.keys():
			mode_dict=self.modes_dict[key]
			v_mode_2=0.
			xs=mode_dict['xs']
			l=mode_dict['l']
			if l>lmax:
				continue
			v_mode_2=np.sum([self.get_mode_vel(key, m, capt_params)**2. for m in range(-int(l), int(l)+1)], axis=0)
			v_mode_2_regrid=v_mode_2_regrid+log_interp(self.rs, xs[1:], v_mode_2[1:])
		return v_mode_2_regrid**0.5

	def sonic(self, capt_params):
		capt_params['ms']=self.M
		vs=self.get_mode_vel_tot(capt_params)
		if np.all(vs/self.cs<1.):
			return 0.
		return log_interp(1., vs/self.cs, self.rs)

	def delta_m(self, capt_params, lmax=2):
		capt_params['ms']=self.M
		vs=self.get_mode_vel_tot(capt_params, lmax)
		if np.all(vs/self.v_esc<1.):
			return 0.
		return 1.-log_interp(1., vs/self.v_esc, self.ms)

	def en_diss(self, capt_params):
		Ts=self.T
		capt_params['ms']=self.M
		eta1=eta(capt_params)
		T1=log_interp(eta1, self.etas, Ts)

		return (T1*(const.G*self.M**2./self.R)*capt_params['lam']**-6.).cgs

	def en_circ(self, capt_params):
		rt=(capt_params['mc']/self.M)**(1./3.)*self.R
		rp=capt_params['lam']*rt
		return const.G*self.M*capt_params['mc']/4./rp

	def get_sf_rm(self, mode, m, capt_params,  ri=0.99):
		'''
		Get shock formation location from Eq. 7 of Ro&Matzner 2017
		
		NB by default, the mode is deposited near the outer layers of the star unlike in Ro&Matzner
		where it is deposited deep inside of the star. xi is the initial radius of the Perturbation
		'''
		Lmax=2.0*np.pi*self.rs**2.*self.rhos*self.cs**3.

		xs2,vs=self.get_mode_vel(mode, m, capt_params)
		ud=IUS(xs2, vs).derivative(1)(ri)
		
		Lmaxi=IUS(self.rs, Lmax)(ri)
		g=5./3.
		return self.rs, [IUS(self.rs, -ud*(g+1.)/2.*(Lmax/Lmaxi)**0.5/self.cs).integral(ri, rr) for rr in self.rs]
	


class n32_poly(object):
	def __init__(self):
		dat_poly=np.genfromtxt('poly32.tsv')
		self.R=u.Quantity(dat_poly[:,0])
		self.Rho=u.Quantity(dat_poly[:,1])
		self.mass=u.Quantity([IUS(self.R, 4.0*np.pi*self.R**2.*self.Rho).integral(self.R[0], rr) for rr in self.R])
		self.cs=u.Quantity((0.53*(5./3.)*(self.Rho/self.Rho[0])**(2./3.))**0.5*(cgs.G*self.mass[-1]/self.R[-1])**0.5)
		self.P=u.Quantity(self.cs**2.*self.Rho/(5./3.))

class n3_poly(object):
	def __init__(self):
		dat_poly=np.genfromtxt('poly3.tsv')
		self.R=u.Quantity(dat_poly[:,0])
		self.Rho=u.Quantity(dat_poly[:,1])
		self.mass=u.Quantity([IUS(self.R, 4.0*np.pi*self.R**2.*self.Rho).integral(self.R[0], rr) for rr in self.R])
		self.cs=u.Quantity((0.85*(5./3.)*(self.Rho/self.Rho[0])**(1./3.))**0.5*(cgs.G*self.mass[-1]/self.R[-1])**0.5)
		self.P=u.Quantity(self.cs**2.*self.Rho/(5./3.))
