import os
import numpy as np
import datetime as dtm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from npat import Spectrum, Calibration, DecayChain, PeakFit, Isotope, colors


def calibrate():
	spectra = []
	calib = {'Ba133':['133BA',39.89E3],'Cs137':['137CS',38.55E3],'Eu152':['152EU',39.29E3]}
	for dr in calib:
		_path = 'count_room_data/calibration/'+dr
		for fl in list(os.walk(_path))[0][2]:
			if fl.endswith('.Spe'):
				shelf = fl.split('_')[-1].split('.')[0]
				sp = Spectrum(os.path.join(_path, fl), 'peak_data.db')
				sp.meta = {'istp':[calib[dr][0]], 'A0':calib[dr][1], 'ref_date':'01/01/2009 12:00:00', 'shelf':shelf}
				# sp.fit_config = {'xrays':True, 'E_min':20.0}
				spectra.append(sp)

	cb = Calibration()
	cb.calibrate(spectra)
	cb.plot()

def sum_spectra():
	spec = None
	_path = 'count_room_data/experiment/'
	for fl in sorted(list(os.walk(_path))[0][2]):
		if fl.endswith('.Spe') and fl.startswith('225Ac_separated_8cm'):
			sp = Spectrum(os.path.join(_path, fl))
			if spec is None:
				spec = sp
			else:
				spec += sp

	spec.saveas(os.path.join(_path, '225Ac_separated_8cm_summed.Spe'))


def fit_actinium_gammas():
	spectra = []
	
	sp = Spectrum('count_room_data/experiment/08cm_liquid_Eu_020319.Spe')
	sp.meta = {'istp':['152EU','154EU'], 'A0':[9.72E-9*3.7E10,1.04E-9*3.7E10], 'ref_date':'09/02/2017 12:00:00'}
	# sp = Spectrum('count_room_data/calibration/Eu152/AI20181101_Eu152_50cm.Spe')
	# sp.meta = {'istp':['152EU'], 'A0':39.29E3, 'ref_date':'01/01/2009 12:00:00'}
	spectra.append(sp)

	# sp = Spectrum('count_room_data/experiment/225Ac_separated_8cm_summed.Spe')
	sp = Spectrum('count_room_data/experiment/225Ac_separated_8cm_000.Spe')
	# sp = Spectrum('count_room_data/experiment/225Ac_separated_50cm.Spe')
	istp = ['221FR','221RN','213BI','209TL','226RA','214PB'] # '214BI','228AC','210TL','210BI','212PB','212BI','212PO','208TL','40K'

	sp.meta = {'istp':istp}
	
	spectra.append(sp)

	cb = Calibration()
	cb.calibrate(spectra)
	# cb.plot()

	t0 = dtm.datetime.strptime('11/20/2018 09:39:04', '%m/%d/%Y %H:%M:%S')
	sp.meta = {'istp':istp}
	sp.fit_config = {'I_min':0.1, 'pk_width':7.0, 'SNR_cut':4.5}

	start_time = (sp.meta['start_time']-t0).total_seconds()/(24.0*3600.0)
	end_time = start_time+(sp.meta['real_time']/(24.0*3600.0))
	counts = {}

	sp.summarize()
	sp.plot()

	for f in sp.fits:
		for n in range(len(f.gm)):
			if f.decays[1][n]<f.decays[0][n]/(0.15) and f.counts[0][n]>1:
				if f.istp[n] in counts:
					counts[f.istp[n]].append([start_time, end_time, f.decays[0][n], f.decays[1][n]])
				else:
					counts[f.istp[n]] = [[start_time, end_time, f.decays[0][n], f.decays[1][n]]]

	# counts['213BI'] = [c for c in counts['213BI'] if c[2]<1E10]

	dc = DecayChain('225AC', 'd', A0=3E4, time=90.0)
	dc.counts = counts
	dc.fit_A0()
	print(dc.activity('225AC', 0.0), dc.activity('225AC',0.0)/3.7E4)
	dc.plot(N_plot=10, logscale=False)


def get_radium_activity():
	spectra = []
	sp = Spectrum('clover_data/calib_test/combined_calib_7.Spe')
	sp.meta = {'istp':['133BA','137CS','152EU'],'A0':[39.89E3, 38.55E3, 39.29E3],'ref_date':'01/01/2009 12:00:00'}
	spectra.append(sp)

	sp = Spectrum('clover_data/mvmelst_021/mvmelst_021_Ch7.Spe')
	sp.meta = {'istp':['226RA','214PB']}
	sp.rebin(2**13)
	# spectra.append(sp)

	cb = Calibration()
	cb.calibrate(spectra)
	cb.plot()

	sp.meta = {'effcal':spectra[0].cb.effcal, 'unc_effcal':spectra[0].cb.unc_effcal}

	A, unc_A = [],[]
	for f in sp.fits:
		for n in range(len(f.gm)):
			if f.gm[n][1]*100.0>1.0:
				A.append(f.decay_rate[0][n])
				unc_A.append(f.decay_rate[1][n])

	print('estimated_activity', np.average(A, weights=1.0/np.array(unc_A)**2)/3.7E4, 'uCi')

	sp.summarize()
	sp.plot()



def fit_radium_alpha():

	def peak(x, A1, R2, mu, sig, tau1, tau2):
		r2 = 1.0/np.sqrt(2.0)
		x, x2 = x[(x-mu)<100],x[(x-mu)>=100]
		f = A1*np.exp((x-mu)/tau1+sig**2/(2.0*tau1**2))*erfc(r2*((x-mu)/sig+(sig/tau1)))
		f += R2*A1*np.exp((x-mu)/tau2+sig**2/(2.0*tau2**2))*erfc(r2*((x-mu)/sig+(sig/tau2)))
		return np.append(f, np.zeros(len(x2)))

	def Npeak(x, *args):
		p = np.zeros(len(x))
		for N in range(int(len(args)/6)):
			p += peak(x, *args[N*6:N*6+6])
		return p


	alphas = [['226RA',4601.0],['226RA',4784.3],['210PO',5304.3],['228TH',5340.36],
				['228TH',5423.15],['222RN',5489.48],['224RA',5685.37],['218PO',6002.35],
				['220RN',6288.08],['211BI',6622.9],['221FR',6341.0],['216PO',6778.3],['217AT',7066.9],
				['214PO',7686.82],['213PO',8376.0],['212PO',8784.86]]

	intensities = [Isotope(a[0]).alphas(E_lim=[a[1]-10, a[1]+10],I_lim=[0.1,None])['I'][0] for a in alphas]


	sp = Spectrum('count_room_data/experiment/Ra full spec.Spe')

	x = np.arange(sp.cb.map_idx(3E3),sp.cb.map_idx(10E3))
	hist = sp.hist[x[0]:x[-1]+1]

	energies = [i[1] for i in alphas]
	idx = [sp.cb.map_idx(i) for i in energies]
	A = [sp.hist[i] for i in idx]
	p0 = []
	bounds = [[],[]]
	for n,a in enumerate(A):
		p0 += [a, 0.1, idx[n]+5, 3.0, 3.0, 30.0]
		bounds[0] += [0.0, 0.01, idx[n]-10.0, 1.0, 1.0, 25.0]
		bounds[1] += [np.inf, 0.1, idx[n]+15.0, 5.0, 5.0, 35.0]

	fit, unc = curve_fit(Npeak, x, hist, p0=p0, bounds=tuple(bounds))



	f, ax = sp.plot(f=None, ax=None, show=False)
	pk_fit = Npeak(x, *fit)
	ax.plot(sp.cb.eng(x), np.where(pk_fit>0.1, pk_fit, 0.1))

	decays = []
	cm = colors()
	for n in range(int(len(fit)/6)):
		pk_fit = peak(x, *fit[6*n:6*n+6])
		ax.plot(sp.cb.eng(x), np.where(pk_fit>0.1, pk_fit, 0.1), ls='--')
		ax.text(alphas[n][1], 10.0**(1.35-1.0)*sp.hist[int(fit[6*n+2])]+30.0, Isotope(alphas[n][0]).TeX, 
				horizontalalignment='left', rotation='vertical', verticalalignment='bottom')

		alphas[n].append(int(sum(peak(x, *fit[6*n:6*n+6]))))
		decays.append([alphas[n][-1]/(0.01*intensities[n])])
		
		decays[n].append((np.sqrt(alphas[n][-1])/float(alphas[n][-1]))*decays[n][0])

		print(alphas[n])
		print(decays[n])
		print(map(round, fit[6*n:6*n+6], [3]*6))
		print

	plt.show()

	t0 = dtm.datetime.strptime('11/07/2018 12:03:00', '%m/%d/%Y %H:%M:%S')

	start_time = (sp.meta['start_time']-t0).total_seconds()/(24.0*3600.0)
	end_time = start_time+sp.meta['real_time']/(24.0*3600.0)
	counts = {}

	dc_226Ra = np.average([d[0] for d in decays[:2]], weights=[1.0/d[1]**2 for d in decays[:2]])

	dc_actual = sp.meta['real_time']*3.7E7*0.846#0.791
	eff = dc_226Ra/dc_actual

	for istp in ['221FR','217AT','213PO']:
		n = [n for n,a in enumerate(alphas) if a[0]==istp][0]
		counts[istp] = [[start_time, end_time, decays[n][0]/eff, decays[n][1]/eff]]


	dc = DecayChain('225RA','d',R=9.0,time=10.0/24.0)
	dc.append(DecayChain('225RA','d',R=2.0,time=33.0/24.0))
	dc.append(DecayChain('225RA','d',R=5.0,time=113.0/24.0))
	dc.append(DecayChain('225RA','d',time=90.0))
	dc.counts = counts
	dc.fit_R()
	print(dc.activity('225AC', start_time), dc.activity('225AC', start_time)/3.7E4)
	print(max(dc.activity('225AC')), max(dc.activity('225AC'))/3.7E4)

	dc.plot(N_plot=10, logscale=False)




def fit_radium_post_separation():
	

	def peak(x, A1, R2, mu, sig, tau1, tau2):
		r2 = 1.0/np.sqrt(2.0)
		x, x2 = x[(x-mu)<100],x[(x-mu)>=100]
		f = A1*np.exp((x-mu)/tau1+sig**2/(2.0*tau1**2))*erfc(r2*((x-mu)/sig+(sig/tau1)))
		f += R2*A1*np.exp((x-mu)/tau2+sig**2/(2.0*tau2**2))*erfc(r2*((x-mu)/sig+(sig/tau2)))
		return np.append(f, np.zeros(len(x2)))

	def Npeak(x, *args):
		p = np.zeros(len(x))
		for N in range(int(len(args)/6)):
			p += peak(x, *args[N*6:N*6+6])
		return p


	alphas = [['226RA',4601.0],['226RA',4784.3],['210PO',5304.3],['228TH',5340.36],
				['228TH',5423.15],['222RN',5489.48],['224RA',5685.37],['218PO',6002.35],
				['220RN',6288.08],['211BI',6622.9],['221FR',6341.0],['216PO',6778.3],['217AT',7066.9],
				['214PO',7686.82],['213PO',8376.0],['212PO',8784.86]]

	intensities = [Isotope(a[0]).alphas(E_lim=[a[1]-10, a[1]+10],I_lim=[0.1,None])['I'][0] for a in alphas]

	sp = Spectrum('count_room_data/experiment/Ra full 11-29_after_separation.Spe')

	x = np.arange(sp.cb.map_idx(3E3),sp.cb.map_idx(10E3))
	hist = sp.hist[x[0]:x[-1]+1]

	energies = [i[1] for i in alphas]
	idx = [sp.cb.map_idx(i) for i in energies]
	A = [sp.hist[i] for i in idx]
	p0 = []
	bounds = [[],[]]
	for n,a in enumerate(A):
		p0 += [a, 0.1, idx[n]+5, 3.0, 3.0, 30.0]
		bounds[0] += [0.0, 0.01, idx[n]-10.0, 1.0, 1.0, 25.0]
		bounds[1] += [np.inf, 0.1, idx[n]+15.0, 5.0, 5.0, 35.0]

	fit, unc = curve_fit(Npeak, x, hist, p0=p0, bounds=tuple(bounds))



	f, ax = sp.plot(f=None, ax=None, show=False)
	pk_fit = Npeak(x, *fit)
	ax.plot(sp.cb.eng(x), np.where(pk_fit>0.1, pk_fit, 0.1))

	decays = []
	cm = colors()
	for n in range(int(len(fit)/6)):
		pk_fit = peak(x, *fit[6*n:6*n+6])
		ax.plot(sp.cb.eng(x), np.where(pk_fit>0.1, pk_fit, 0.1), ls='--')
		ax.text(alphas[n][1], 10.0**(1.35-1.0)*sp.hist[int(fit[6*n+2])]+30.0, Isotope(alphas[n][0]).TeX, 
				horizontalalignment='left', rotation='vertical', verticalalignment='bottom')

		alphas[n].append(int(sum(peak(x, *fit[6*n:6*n+6]))))
		decays.append([alphas[n][-1]/(0.01*intensities[n])])
		
		decays[n].append((np.sqrt(alphas[n][-1])/float(alphas[n][-1]))*decays[n][0])

		print(alphas[n])
		print(decays[n])
		print(map(round, fit[6*n:6*n+6], [3]*6))
		print

	plt.show()

	t0 = dtm.datetime.strptime('11/20/2018 09:39:04', '%m/%d/%Y %H:%M:%S')

	start_time = (sp.meta['start_time']-t0).total_seconds()/(24.0*3600.0)
	end_time = start_time+sp.meta['real_time']/(24.0*3600.0)
	counts = {}

	dc_226Ra = np.average([d[0] for d in decays[:2]], weights=[1.0/d[1]**2 for d in decays[:2]])

	dc_actual = sp.meta['real_time']*3.7E7*0.088
	eff = dc_226Ra/dc_actual
	# print eff

	for istp in ['221FR','217AT','213PO']:
		n = [n for n,a in enumerate(alphas) if a[0]==istp][0]
		counts[istp] = [[start_time, end_time, decays[n][0]/eff, decays[n][1]/eff]]


	# dc = DecayChain('225RA','d',R=9.0,time=10.0/24.0)
	# dc.append(DecayChain('225RA','d',R=2.0,time=33.0/24.0))
	# dc.append(DecayChain('225RA','d',R=5.0,time=113.0/24.0))
	# dc.append(DecayChain('225RA','d',time=90.0))
	# dc.counts = counts
	# dc.fit_R()
	# print dc.activity('225AC', start_time), dc.activity('225AC', start_time)/3.7E4
	# print max(dc.activity('225AC')), max(dc.activity('225AC'))/3.7E4

	# dc.plot(N_plot=10, logscale=False)

	dc = DecayChain('225AC', 'd', A0=3E4, time=90.0)
	dc.counts = counts
	dc.fit_A0()
	print(dc.activity('225AC', 0.0), dc.activity('225AC',0.0)/3.7E4)
	dc.plot(N_plot=10, logscale=False)



def fit_all_summed():
	spectra = []
	
	sp = Spectrum('count_room_data/experiment/08cm_liquid_Eu_020319.Spe')
	sp.meta = {'istp':['152EU'], 'A0':10.76E-9*3.7E10, 'ref_date':'10/26/2017 12:00:00'}
	spectra.append(sp)

	istp = ['221FR','221RN','213BI','209TL','226RA','214PB','214BI','228AC','210TL','210BI','212PB','212BI','212PO','208TL','40K']
	sp = Spectrum('count_room_data/experiment/225Ac_separated_8cm_summed.Spe')
	sp.meta = {'istp':istp}
	spectra.append(sp)

	cb = Calibration()
	cb.calibrate(spectra)

	sp.plot()

def calc_mda():
	spectra = []
	
	sp = Spectrum('count_room_data/experiment/08cm_liquid_Eu_020319.Spe')
	sp.meta = {'istp':['152EU'], 'A0':10.76E-9*3.7E10, 'ref_date':'10/26/2017 12:00:00'}
	spectra.append(sp)

	istp = ['221FR','221RN','213BI','209TL','226RA','214PB','214BI','228AC','210TL','210BI','212PB','212BI','212PO','208TL','40K']
	sp = Spectrum('count_room_data/experiment/225Ac_separated_8cm_summed.Spe')
	sp.meta = {'istp':istp}
	spectra.append(sp)

	cb = Calibration()
	cb.calibrate(spectra)

	eng = np.array([235.96, 256.23, 269.46])
	I_g = np.array([12.9, 7.0, 13.9])*0.01
	idx = sp.cb.map_idx(eng)
	eff = sp.cb.eff(eng)
	bg = np.array([np.std(sp.hist[i-20:i+20]) for i in idx])
	mda = 1.5*bg/(I_g*eff*sp.meta['live_time']*0.9862)
	mda = np.average(mda)
	print(mda, mda/3.7E4, (mda/53.725E3)*(21.772*365.0/10.0))






if __name__=="__main__":
	# sp = Spectrum('count_room_data/experiment/225Ac_separated_8cm_summed.Spe', 'peak_data.db')
	# sp.meta = {'istp':istp}
	# sp.summarize()
	# sp.plot()
	# sum_spectra()
	fit_radium_alpha()
	# fit_radium_post_separation()
	# fit_actinium_gammas()
	# get_radium_activity()
	# fit_all_summed()
	# calc_mda()
	# calibrate()
