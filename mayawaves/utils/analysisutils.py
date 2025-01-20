# what is the plan?
# waveform object that will save parameters
#   print function
# functions
#   taper_time_series                     (DONE)
#   get_detector_frame_modes_from_NR_hdf5 (DONE)
#   get_model_waveform_polarizations      (DONE)
#   mismatch                              (DONE)
#   get_detector_frame_polarizations


import numpy as np
import lal
import lalsimulation as lalsim
import h5py
import sys
import romspline
from scipy import interpolate
import inspect

MSUN = lal.MSUN_SI # solar mass in kgs
MPC = 10**6 * lal.PC_SI # mega parsecs in meters
G = lal.G_SI # gravitational constants in m^3/kg/s^2
C = lal.C_SI # speed of light in m/s

class waveform:
    """Waveform class to store binary parameters and metadata. Should used in conjuction with functions defined here.
    Tip: wf.__dict__  should give you the list of attributes/parameters (here wf is the waveform object).
    """
    def __init__(self,  
                m1= 50,           # MSUN
                m2= 50,           # MSUN
                a1x=0.0, 
                a1y=0.0, 
                a1z=0.0, 
                a2x=0.0, 
                a2y=0.0, 
                a2z=0.0, 
                eccentricity=0.0,
                meanPerAno=0.0,   # radian
                distance=400,     # Mpc 
                phiref=0.0,       # radian
                inclination=0.0,  # radian
                psi=0.0,          # radian

                approximant="SEOBNRv4",  
                fmin=20.,          # Hz
                fmax=1024,         # Hz
                deltaT=1./4096,    # s
                deltaF= 1./8,      # Hz
                fref=20., 
                extra_lal_params=lal.CreateDict(), 
                declination=0.0, 
                rightascenion=0.0,
                NR_hdf5_path=None):
    
        ##intrinsic variables
        self.m1=m1
        self.m2=m2
        self.a1x=a1x
        self.a1y=a1y
        self.a1z=a1z
        self.a2x=a2x
        self.a2y=a2y
        self.a2z=a2z
        self.eccentricity=eccentricity
        self.meanPerAno=meanPerAno

        ##extrinsic variables
        self.distance=distance
        self.phiref=phiref
        self.inclination=inclination
        self.psi=psi
        self.declination=declination   
        self.rightascenion=rightascenion

        
        self.fmin=fmin
        self.fmax=fmax
        self.fref=fref
        self.deltaT=deltaT
        self.deltaF=deltaF

        self.extra_lal_params=extra_lal_params
        self.approximant=approximant
        self.NR_hdf5_path=NR_hdf5_path

def get_taper_vector(taper_type, length):
    """Generates a tapering vector based on the specified taper type.

    Args:
        taper_type (str): The type of taper to generate. Currently supported type is "cosine".
        length (int): The length of the taper vector to be generated.    
    """
    length = int(length)
    if taper_type=="cosine":
        vectaper = 0.5 - 0.5*np.cos(np.pi*np.arange(length)/(1.*length))
        return vectaper
    else:
        print(f"Tapering type '{taper_type}' not yet implemented/")

def taper_time_series(time_series, taper_percent=1, taper_type="cosine"):
    """Applies a tapering function to a time series to gradually increase amplitude at the starting edge.

    Args:
        time_series (numpy.ndarray, lal.REAL8TimeSeries, or lal.COMPLEX16TimeSeries): The input time series to be tapered. It can be one of the following.
        taper_percent (float, optional): The percentage of the time series length to taper at the beginning and end. Default is 1%.
        taper_type (str, optional): The type of taper to apply. Default is "cosine".
    """
    assert 0<=taper_percent <=100, "taper_percent should be between 0 and 100."
    if isinstance(time_series, np.ndarray):
        ntaper = int(taper_percent/100 * len(time_series))
        vectaper = get_taper_vector(taper_type, ntaper)
        time_series[:ntaper]*=vectaper
    elif isinstance(time_series, lal.REAl8TimeSeries):
        ntaper = int(taper_percent/100 * len(time_series.data.length))
        vectaper = get_taper_vector(taper_type, ntaper)
        time_series.data.data[:ntaper]*=vectaper
    elif isinstance(time_series, lal.COMPLEX16TimeSeries):
        ntaper = int(taper_percent/100 * len(time_series.data.length))
        vectaper = get_taper_vector(taper_type, ntaper)
        time_series.data.data.real[:ntaper]*=vectaper
        time_series.data.data.complex[:ntaper]*=vectaper
    return time_series

def get_detector_frame_modes_from_NR_hdf5(waveform_object, lmax=None, modes=None, include_m_0_modes=False, waveform_convention=1, verbose=False):
    assert waveform_convention in [1, -1], "Waveform convention can either be 1 or -1."

    # For unit conversion
    MSUN_sec = G/C**3
    mtot_in_sec = (waveform_object.m1 + waveform_object.m2) * MSUN_sec
    dist_in_sec = waveform_object.dist/C

    # just to know what time array we are dealing with
    data_1 = h5py.File(waveform_object.NR_hdf5_path)

    # mass and fmin (mass set by total mass from the waveform object)
    mtotal = (waveform_object.m1 + waveform_object.m2)
    m1 = data_1.attrs["mass1"] * mtotal 
    m2 = data_1.attrs["mass2"] * mtotal
    fmin = data_1.attrs["f_lower_at_1MSUN"] * MSUN/mtotal
    if verbose:
        print(f"Smallest possible fmin for this waveform {fmin} Hz. fmin at 1 solar mass is {data_1.attrs['f_lower_at_1MSUN']}")
        a1x, a1y, a1z, a2x, a2y, a2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(waveform_object.fref, mtotal/lal.MSUN_SI, waveform_object.NR_hdf5_path)
        print(f"Generating waveform with m1 = {m1/lal.MSUN_SI:0.4f} MSUN, m2 = {m2/lal.MSUN_SI:0.4f} MSUN \n a1 = {a1x, a1y, a1z}, a2 = {a2x, a2y, a2z}\n fmin = {fmin} Hz")

    # Which modes to get
    modes_array = []
    if modes == None and lmax == None:
        lmax = data_1.attrs["Lmax"]
        print(f"Using all available modes with lmax = {lmax}")
    if modes==None and lmax is not None:
        for l in range(2,lmax+1):
            if include_m_0_modes:
                for m in range(-l,l+1):
                    modes_array.append((l,m))
            else:
                for m in range(-l,0):
                    modes_array.append((l,m))
                for m in range(1,l+1):
                    modes_array.append((l,m))
    elif modes is not None and lmax is None:
        for j in modes:
            modes_array.append(j)
    else:
        print("Inconsistent input, use either lmax or modes.")
        sys.exit()
    if verbose:
        print(f"modes used = {modes_array}")

    #interpolating using romspline
    hlm = {}
    for i in range(len(modes_array)):
        amp22_time_0=np.array(data_1[f"phase_l{modes_array[i][0]}_m{modes_array[i][1]}"]["X"])

        amp = romspline.readSpline(waveform_object.NR_hdf5_path, f"amp_l{modes_array[i][0]}_m{modes_array[i][1]}")
        phase = romspline.readSpline(waveform_object.NR_hdf5_path, f"phase_l{modes_array[i][0]}_m{modes_array[i][1]}")
        
        amp22_time_0 = np.arange(np.min(amp22_time_0), np.max(amp22_time_0), waveform_object.deltaT/mtot_in_sec)
        generated_amp = amp(amp22_time_0)
        generated_phase = phase(amp22_time_0)

        wf_data = mtot_in_sec/dist_in_sec * generated_amp * np.exp(waveform_convention*1j*generated_phase) # Note: waveform convention

        max_Re, max_Im = np.max(np.real(wf_data)), -np.max(np.imag(wf_data))
        if verbose:
            print(f"Reading mode {modes_array[i]}, max for this mode: {max_Re, max_Im}")
        wf = lal.CreateCOMPLEX16TimeSeries("hlm", 0, 0, waveform_object.deltaT, lal.DimensionlessUnit, len(wf_data))
        wf.data.data = wf_data
        hlm[modes_array[i][0],modes_array[i][1]] = wf
    
    return hlm


def get_model_waveform_polarizations(waveform_object, modes=None, lmax=None, verbose=True):
    modes_array = []
    if modes==None and lmax is not None:
        for l in range(2,lmax+1):
            for m in range(-l,0):
                if waveform_object.approximant == "NRHybSur3dq8" and l==4 and (m==0 or m==-1): #Throws an error for these modes instead of pass nothing like a normal person
                    continue
                modes_array.append((l,m))
            for m in range(1,l+1):
                if waveform_object.approximant == "NRHybSur3dq8" and l==4 and (m==0 or m==1): #Throws an error for these modes instead of pass nothing like a normal person
                    continue
                modes_array.append((l,m))
    elif modes is not None and lmax is None:
        for j in modes:
            modes_array.append(j)
    else:
        print("Inconsistent input, use either lmax or modes.")
        sys.exit()
    
    if verbose:
        print(f"Using modes {modes_array}")

    ma = lalsim.SimInspiralCreateModeArray()
    for mode in modes_array:
        l,m = mode
        lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
    lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_object.extra_lal_params, ma)

    if verbose:
        print(f"Using approximant {waveform_object.approximant}")

    hp, hc = lalsim.SimInspiralChooseTDWaveform(waveform_object.m1, waveform_object.m2, waveform_object.a1x, waveform_object.a1y, waveform_object.a1z, waveform_object.a2x, waveform_object.a2y, waveform_object.a2z, waveform_object.distance, waveform_object.inclination, \
    waveform_object.phiref, waveform_object.psi, waveform_object.eccentricity, waveform_object.meanPerAno, waveform_object.deltaT, waveform_object.fmin, waveform_object.fref,
    waveform_object.extra_lal_params, getattr(waveform_object.approximant))
    return hp, hc


def resample_psd(psd, fvals):   #this acts weird due to non integer steps size, need to test it
    """Takes in a PSD which should have two columns, frequnecy and data, and returns it with a new  deltaF"""
    frequency, data = np.loadtxt(psd, delimiter=" ", comments="#",unpack=True)
    interp = interpolate.interp1d(frequency, data, fill_value = 'extrapolate')
    return fvals, interp(fvals)
    
def mismatch(waveform_time_series1, waveform_time_series2, deltaT_1, deltaT_2, psd="H1", flow=20, fhigh=2048, resize="power_2", phase_maximization_trick=False, output_overlap_time_series=False, verbose=True, plots=False):
    assert deltaT_1 == deltaT_2, f'deltaT of two time series should be the same, you have entered deltaT_1 = {deltaT_1} s, deltaT_2 = {deltaT_2} s.'
    
    # variables for resizing
    len_1, len_2 = len(waveform_time_series1), len(waveform_time_series2)
    max_len, min_len = np.max([len_1, len_2]), np.min([len_1, len_2])
    power2_len = int(2**np.ceil(np.log2(max_len)))

    # resizing
    if resize == 'max':
        if verbose:
            print(f"Resizing to {max_len} from len_1 = {len_1}, len_2 = {len_2}")
        wf_tseries_1 = np.zeros(max_len, dtype=complex)
        wf_tseries_1[:len(waveform_time_series1)] = waveform_time_series1

        wf_tseries_2 = np.zeros(max_len, dtype=complex)
        wf_tseries_2[:len(waveform_time_series2)] = waveform_time_series2
    
    elif resize == 'min':
        if verbose:
            print(f"Resizing to {min_len} from len_1 = {len_1}, len_2 = {len_2}")
        wf_tseries_1 = np.zeros(min_len, dtype=complex)
        wf_tseries_1[:len(waveform_time_series1)] = waveform_time_series1

        wf_tseries_2 = np.zeros(min_len, dtype=complex)
        wf_tseries_2[:len(waveform_time_series2)] = waveform_time_series2
    
    elif resize == "power_2":
        if verbose:
            print(f"Resizing to {power2_len} from len_1 = {len_1}, len_2 = {len_2}")
        wf_tseries_1 = np.zeros(power2_len, dtype=complex)
        wf_tseries_1[:len(waveform_time_series1)] = waveform_time_series1

        wf_tseries_2 = np.zeros(power2_len, dtype=complex)
        wf_tseries_2[:len(waveform_time_series2)] = waveform_time_series2

    # FFT
    wf_1_FD = np.fft.fft(wf_tseries_1)
    wf_2_FD = np.fft.fft(wf_tseries_2)
    deltaF  = 1 / (len(waveform_time_series1) * deltaT_1)
    fvals = np.fft.fftfreq(len(wf_tseries_1), deltaT_1)

    # Load PSD
    curr_path=inspect.getfile(inspect.currentframe())
    index_path=curr_path.find("analysisutils")
    if psd == "H1":
        psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt"
    if psd == "L1":
        psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
    if psd == "V1":
        psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
    if psd == "ET":
        psd=curr_path[:index_path]+"/PSD/ET.txt"
    if psd == "CE":
        psd=curr_path[:index_path]+"/PSD/CE.txt"
    if psd == "LISA":
            psd=curr_path[:index_path]+"/PSD/LISA.txt"
    if psd == "Flat":
        data = np.ones(len(fvals))
    else:
        frequency, data = resample_psd(psd, np.abs(fvals))
    if verbose:
        print(f"Using PSD {psd}")
        print(f"Integrating from flow={flow} Hz, fhigh={fhigh} Hz")
    # the psd data will be defined over all the frequency, now we have to choose
    index_1 = np.argwhere(np.abs(fvals)>= flow)
    index_2 = np.argwhere(np.abs(fvals)<= fhigh)
    combined = np.intersect1d(index_1, index_2)

    weights = np.zeros(len(data))
    weights[combined] = 1/data[combined]

    # Norms
    norm_1 = np.sqrt(2 * 4 * deltaF * np.sum(wf_1_FD.conj() * wf_1_FD * weights) / len(wf_1_FD)**2 )
    norm_2 = np.sqrt(2 * 4 * deltaF * np.sum(wf_2_FD.conj() * wf_2_FD * weights) / len(wf_1_FD)**2 )
    if verbose:
        print(f"norm-1 = {norm_1}, norm-2 = {norm_2}")

    # IP
    integrand = 2 * wf_1_FD.conj() * wf_2_FD
    overlap_time_shift = np.fft.ifft(integrand)/len(integrand) * 2

    if phase_maximization_trick:
        overlap_time_series = np.abs(overlap_time_shift)
    else:
        overlap_time_series = np.real(overlap_time_shift)
    
    time_max_match = np.max(overlap_time_series)
    if output_overlap_time_series:
        return 1 - time_max_match, overlap_time_series
    else:
        return 1 - time_max_match