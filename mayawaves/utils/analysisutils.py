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
    """Waveform class to store binary system parameters and metadata.

    Attributes can be accessed as a dictionary using `wf.__dict__`."""
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
    """
    Generate a taper window for a signal to minimize spectral leakage.

    Args:
        taper_type (str): Taper type. Currently only supports 'cosine'.
        length (int): Length of the taper window.

    Returns:
        numpy.ndarray: Array containing the taper values.
    """
    length = int(length)
    if taper_type=="cosine":
        vectaper = 0.5 - 0.5*np.cos(np.pi*np.arange(length)/(1.*length))
        return vectaper
    else:
        print(f"Tapering type '{taper_type}' not yet implemented/")

def taper_time_series(time_series, taper_percent=1, taper_type="cosine"):
    """
    Apply a tapering window to the beginning a time series.

    Args:
        time_series (numpy.ndarray or lal.TimeSeries): The signal to taper.
        taper_percent (float): Percentage of the data length to taper.
        taper_type (str): Type of taper (default "cosine").

    Returns:
        Same type as input: Tapered time series.
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

def Ylm(inclination, phiref, l ,m, s = -2):
    """
    Compute the spin-weighted spherical harmonic.

    Args:
        inclination (float): Inclination angle (rad).
        phiref (float): Reference phase (rad).
        l (int): Spherical harmonic index.
        m (int): Azimuthal index.
        s (int): Spin weight (default -2 for GW).

    Returns:
        complex: The value of the spherical harmonic.
    """
    return lal.SpinWeightedSphericalHarmonic(inclination,phiref,s,l,m)

def get_detector_frame_modes_from_NR_hdf5(waveform_object, lmax=None, modes=None, include_m_0_modes=False, waveform_convention=1, verbose=False):
    """
    Extract (l, m) modes from an NR HDF5 file and return them as lal.COMPLEX16TimeSeries.

    Args:
        waveform_object (waveform): Object containing system and waveform parameters.
        lmax (int, optional): Maximum l to include.
        modes (list, optional): List of (l, m) mode tuples.
        include_m_0_modes (bool, optional): Whether to include m=0 modes.
        waveform_convention (int, optional): +1 or -1 phase convention.
        verbose (bool): Print verbose output.

    Returns:
        dict: Dictionary of {(l, m): COMPLEX16TimeSeries} pairs.
    """
    assert waveform_convention in [1, -1], "Waveform convention can either be 1 or -1."

    # For unit conversion
    MSUN_sec = G/C**3
    mtot_in_sec = (waveform_object.m1 + waveform_object.m2) * MSUN_sec * MSUN
    dist_in_sec = waveform_object.distance*MPC/C

    # just to know what time array we are dealing with
    data_1 = h5py.File(waveform_object.NR_hdf5_path)

    # mass and fmin (mass set by total mass from the waveform object)
    mtotal = (waveform_object.m1 + waveform_object.m2)*MSUN
    m1 = data_1.attrs["mass1"] * mtotal 
    m2 = data_1.attrs["mass2"] * mtotal
    fmin = data_1.attrs["f_lower_at_1MSUN"] * MSUN/mtotal
    if verbose:
        print(f"Smallest possible fmin for this waveform {fmin} Hz. fmin at 1 solar mass is {data_1.attrs['f_lower_at_1MSUN']}")
        a1x, a1y, a1z, a2x, a2y, a2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(waveform_object.fref, mtotal/lal.MSUN_SI, waveform_object.NR_hdf5_path)
        print(f"Generating waveform with m1 = {m1/MSUN:0.4f} MSUN, m2 = {m2/MSUN:0.4f} MSUN \n a1 = {a1x, a1y, a1z}, a2 = {a2x, a2y, a2z}\n fmin = {fmin} Hz")

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

def get_detector_frame_polarizations_from_NR_hdf5(waveform_object,  lmax=None, modes=None, include_m_0_modes=False, waveform_convention=1, verbose=False):
    """
    Generate plus and cross polarizations from NR HDF5 modes for a given binary configuration.

    Args:
        waveform_object (waveform): The waveform object with parameters and HDF5 path.
        lmax, modes, include_m_0_modes, waveform_convention, verbose: See `get_detector_frame_modes_from_NR_hdf5`.

    Returns:
        tuple: (h_plus, h_cross) as lal.REAL8TimeSeries.
    """
    hlm = get_detector_frame_modes_from_NR_hdf5(waveform_object, lmax=lmax, modes=modes, include_m_0_modes=include_m_0_modes, waveform_convention=waveform_convention, verbose=verbose)
    keys = list(hlm.keys())
    for i in range(len(keys)):
        if i == 0 :
            tmp = hlm[keys[i]].data.data * Ylm(waveform_object.inclination,waveform_object.phiref, keys[i][0], keys[i][1])
        else:
            tmp +=hlm[keys[i]].data.data * Ylm(waveform_object.inclination,waveform_object.phiref, keys[i][0], keys[i][1])

        h_p = lal.CreateREAL8TimeSeries("hlm",0,0, waveform_object.deltaT,lal.DimensionlessUnit,len(tmp))
        h_p.data.data = np.real(tmp)
        h_c = lal.CreateREAL8TimeSeries("hlm",0,0, waveform_object.deltaT,lal.DimensionlessUnit,len(tmp))
        h_c.data.data = -np.imag(tmp)
    return h_p, h_c

def get_model_waveform_polarizations(waveform_object, modes=None, lmax=None, verbose=True, include_negative_m_modes=False):
    """
    Generate time-domain polarizations (hp, hc) for waveform models using LALSimulation.

    Args:
        waveform_object (waveform): The waveform object with model and system parameters.
        modes (list): Specific modes to include.
        lmax (int): Maximum l value for mode summation.
        verbose (bool): Verbosity.
        include_negative_m_modes (bool): Whether to include negative-m modes.

    Returns:
        tuple: (hp, hc) as lal.REAL8TimeSeries.
    """
    modes_array = []
    if modes==None and lmax is not None and include_negative_m_modes==True:
        for l in range(2,lmax+1):
            for m in range(-l,0):
                if waveform_object.approximant == "NRHybSur3dq8" and l==4 and (m==0 or m==-1): #Throws an error for these modes instead of pass nothing like a normal person
                    continue
                modes_array.append((l,m))
            for m in range(1,l+1):
                if waveform_object.approximant == "NRHybSur3dq8" and l==4 and (m==0 or m==1): #Throws an error for these modes instead of pass nothing like a normal person
                    continue
                modes_array.append((l,m))
    if modes==None and lmax is not None and include_negative_m_modes==False:
        for l in range(2,lmax+1):
            for m in range(1,l+1):
                if waveform_object.approximant == "NRHybSur3dq8" and l==4 and (m==0 or m==1): #Throws an error for these modes instead of pass nothing like a normal person
                    continue
                modes_array.append((l,m))
    if modes is not None and lmax is None:
        for j in modes:
            modes_array.append(j)
    if modes is not None and lmax is not None:
        print("Inconsistent input, use either lmax or modes.")
        sys.exit()
    
    if verbose:
        print(f"Using modes {modes_array}")

    ma = lalsim.SimInspiralCreateModeArray()
    if modes or lmax:
        for mode in modes_array:
            l,m = mode
            lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_object.extra_lal_params, ma)

    if verbose:
        print(f"Using approximant {waveform_object.approximant}")

    hp, hc = lalsim.SimInspiralChooseTDWaveform(waveform_object.m1*MSUN, waveform_object.m2*MSUN, waveform_object.a1x, waveform_object.a1y, waveform_object.a1z, waveform_object.a2x, waveform_object.a2y, waveform_object.a2z, waveform_object.distance*MPC, waveform_object.inclination, \
    waveform_object.phiref, waveform_object.psi, waveform_object.eccentricity, waveform_object.meanPerAno, waveform_object.deltaT, waveform_object.fmin, waveform_object.fref,
    waveform_object.extra_lal_params, getattr(lalsim,waveform_object.approximant))
    return hp, hc


def resample_psd(psd, df=None):   #this acts weird due to non integer steps size, need to test it
    """
    Resample a power spectral density (PSD) to a new frequency resolution.

    Args:
        psd (str): Path to PSD text file with frequency and PSD values.
        df (float): Desired frequency spacing.

    Returns:
        tuple: (new_frequency, new_psd) arrays.
    """
    frequency, data = np.loadtxt(psd, delimiter=" ", comments="#",unpack=True)
    f0, deltaF, f_final = frequency[0], frequency[1]-frequency[0], frequency[-1]
    interp = interpolate.interp1d(frequency, data, fill_value = 'extrapolate')
    new_frequency = np.arange(f0, f_final+5*df, df or deltaF)
    return new_frequency, interp(new_frequency)
    
def mismatch(waveform_time_series1, waveform_time_series2, deltaT_1, deltaT_2, psd="H1", flow=20, fhigh=2048, resize="power_2", phase_maximization_trick=False, output_mismatch_time_series=False, verbose=True, integral_factor=4):
    """
    Compute the mismatch between two time-domain waveforms using their FFTs and a detector PSD.

    Args:
        waveform_time_series1, waveform_time_series2 (np.ndarray): Complex time-domain data.
        deltaT_1, deltaT_2 (float): Sampling intervals.
        psd (str): detector name.
        flow (float): Lower frequency cutoff.
        fhigh (float): Upper frequency cutoff.
        resize (str, optional): How to resize waveforms ("min", "max", or "power_2"). Stick to "power_2", unless absolutely necessary.
        phase_maximization_trick (bool): If True, marginalize over phase. Approximation fails if higher order mode content is significant.
        output_mismatch_time_series (bool): If True, return time-dependent mismatch.
        verbose (bool): Verbosity.
        integral_factor (float, optional): The co-efficient for integral. Should be 2 but the mismatch values are independent of this choice as the factors cancel out.

    Returns:
        float: Mismatch value.
    """
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
    wf_1_FD_og = np.fft.fft(wf_tseries_1)
    wf_1_FD =  wf_1_FD_og * deltaT_1 # FFT output from numpy is missing deltaT
    wf_1_FD = np.roll(wf_1_FD, len(wf_1_FD)//2) # make it continuous, not following numpy's packaging
    
    wf_2_FD_og = np.fft.fft(wf_tseries_2)
    wf_2_FD = wf_2_FD_og * deltaT_2 # FFT output from numpy is missing deltaT
    wf_2_FD = np.roll(wf_2_FD, len(wf_2_FD)//2) # make it continuous, not following numpy's packaging

    T = (len(wf_tseries_1) * deltaT_1)
    deltaF  = 1/T
    n = len(wf_2_FD)
    fvals_FFT = deltaF*(np.arange(n) - n/2+1) 
    
    if verbose:
        print(f'Rolling amount is {len(wf_1_FD)/2}, time = {T}, deltaF = {deltaF}, ')
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
        frequency = np.arange(flow-10*deltaF, fhigh+10*deltaF, deltaF)
        data = np.ones(len(frequency))
    else:
        frequency, data = resample_psd(psd, deltaF)
    indices = np.arange(len(frequency))
    indices_integration = indices[np.logical_and(frequency>=flow, frequency<=fhigh)]
    if verbose:
        print(f"Using PSD {psd}")
        print(f"Integrating from flow={frequency[indices_integration[0]]} Hz, fhigh={frequency[indices_integration[-1]]} Hz")
    weights_one_sided = np.zeros(int(len(wf_2_FD)/2+1))
    weights_one_sided[indices_integration] = 1/data[indices_integration]

    weights_two_sided = np.zeros(len(wf_2_FD))
    weights_two_sided[:len(weights_one_sided)]=weights_one_sided[::-1]    #[-N2--->0]
    weights_two_sided[len(weights_one_sided)-1:]=weights_one_sided[:-1]   #[0--->N/2)   #zero index filled twice, +N/2 not there

    # Norms
    # This integration is from -fmax to fmax, so the integral_factor should really be 2. It doesn't matter here since the factors cancel out and mismatch stays the same.
    norm_1 = np.sqrt(integral_factor * deltaF * np.sum(wf_1_FD.conj() * wf_1_FD * weights_two_sided)).real
    norm_2 = np.sqrt(integral_factor * deltaF * np.sum(wf_2_FD.conj() * wf_2_FD * weights_two_sided)).real
    if verbose:
        print(f"norm-1 = {norm_1}, norm-2 = {norm_2}")

    # IP
    integrand = integral_factor * np.roll((wf_1_FD.conj() * wf_2_FD * weights_two_sided), -n//2)
    overlap_time_shift = np.fft.ifft(integrand) * (deltaF * n)

    if phase_maximization_trick:
        overlap_time_series = np.abs(overlap_time_shift)
    else:
        overlap_time_series = np.real(overlap_time_shift)
    
    time_max_match = np.max(overlap_time_series)
    if output_mismatch_time_series:
        if verbose:
            max_index = np.argmax(overlap_time_series)
            print(f'Minimum mismatch occurs at time shift of {overlap_time_series[max_index]} s')
        tvals = np.arange(len(overlap_time_series))*deltaT_1
        return 1 - time_max_match/norm_1/norm_2, [tvals, 1-overlap_time_series/norm_1/norm_2]
    else:
        return 1 - time_max_match/norm_1/norm_2