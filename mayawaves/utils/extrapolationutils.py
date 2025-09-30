import numpy as np
import math
import scipy.optimize
import scipy.interpolate
from mayawaves.radiation import RadiationMode, RadiationSphere

###########################################################################################
# Psi4 to strain conversion functions
###########################################################################################
def get_Fourier_Transform(time_vals, time_series):
    """
    Compute the Fourier Transform of a uniformly spaced time series.

    Args:
        time_vals (array-like): Array of time points, must be uniformly spaced.
        time_series (array-like): Time series of which you want to compute FFT.

    Returns:
        tuple: (frequency_vals, frequency_series) arrays, both centered so that 
               zero frequency is at the center.
               - frequency_vals (numpy.ndarray): Frequency points corresponding to the FFT.
               - frequency_series (numpy.ndarray): Complex Fourier coefficients.
    """
    deltaT = np.diff(time_vals)[0]
    assert np.allclose(np.diff(time_vals), deltaT), "Time values must be uniformly spaced."

    frequency_series = np.fft.fft(time_series, norm="ortho")
    frequency_vals = np.fft.fftfreq(len(frequency_series), d=deltaT)

    # center the two arrays, such that f=0 at the center of the array
    # n = len(frequency_series)//2
    # frequency_series_rolled = np.roll(frequency_series, n)
    # frequency_vals_rolled = np.roll(frequency_vals, n)
    frequency_series_rolled = np.fft.fftshift(frequency_series)
    frequency_vals_rolled = np.fft.fftshift(frequency_vals)

    return frequency_vals_rolled, frequency_series_rolled


def get_inverse_Fourier_Transform(frequency_vals, frequency_series, t0=0.0):
    """
    Compute the inverse Fourier Transform to return to the time domain.

    Args:
        frequency_vals (array-like): Array of frequency points, must be uniformly spaced.
        frequency_series (array-like): Frequency series of which you want to compute IFFT.
        t0 (float, optional): Starting time value for the returned time series. Defaults to 0.0.

    Returns:
        tuple: (time_vals, time_series) arrays.
               - time_vals (numpy.ndarray): Time points corresponding to the inverse FFT.
               - time_series (numpy.ndarray): Reconstructed time-domain signal (complex).
    """
    deltaF = np.diff(frequency_vals)[0]
    assert np.allclose(np.diff(frequency_vals), deltaF), "Frequency values must be uniformly spaced."

    # return to original numpy's packaging of frequency series
    # n = len(frequency_series)//2
    # frequency_series_unrolled = np.roll(frequency_series, -n)
    frequency_series_unrolled = np.fft.ifftshift(frequency_series)

    time_series = np.fft.ifft(frequency_series_unrolled, norm="ortho")

    time_vals = t0 + np.arange(len(time_series)) * 1 / (len(frequency_series_unrolled) * deltaF)
    return time_vals, time_series

# def get_Fourier_Transform(t0, complexPsi):
#     """
#     Transforms the complexPsi data to frequency space

#     t0 = time data points
#     complexPsi = data points of Psi to be transformed
#     """
#     psif = np.fft.fft(complexPsi, norm="ortho")
#     l = len(complexPsi)
#     n = int(math.floor(l/2.))
#     newpsif = psif[l-n:]
#     newpsif = np.append(newpsif, psif[:l-n])
#     T = np.amin(np.diff(t0))*l
#     freq = range(-n, l-n)/T
#     return freq, newpsif

# #Inverse Fourier Transform
# def get_inverse_Fourier_Transform(freq, hf, t0):
#     l = len(hf)
#     n = int(math.floor(l/2.))
#     newhf = hf[n:]
#     newhf = np.append(newhf, hf[:n])
#     amp = np.fft.ifft(newhf, norm="ortho")
#     df = np.amin(np.diff(freq))
#     time = t0 + range(0, l)/(df*l)
#     return time, amp


def integrate_using_fixed_frequency_integration(frequency_vals, complex_psi4f, cutoff_frequency, suppress_lowf_factor=1.0):
    """
    Integrate a complex frequency-domain signal using Fixed-Frequency Integration (FFI).

    This method divides the Fourier-domain signal by (i*omega) while suppressing 
    contributions below a cutoff frequency to avoid low-frequency divergences.

    Args:
        frequency_vals (array-like): Frequencies corresponding to the input Fourier signal.
        complex_psi4f (array-like): Complex frequency-domain signal (e.g., Fourier transform of ψ₄).
        cutoff_frequency (float): Low-frequency cutoff for the integration, in radians/sec.
        suppress_lowf_factor (float, optional): Factor to suppress low-frequency components. Defaults to 1.0.

    Returns:
        numpy.ndarray: Complex frequency-domain signal after integration.
    """
    f1 = cutoff_frequency/(2*math.pi)

    # create masks
    mask1 = (np.sign((frequency_vals/f1) - 1) + 1)/2.
    mask2 = (np.sign((-frequency_vals/f1) - 1) + 1)/2.
    mask = 1 - (1 - mask1) * (1 - mask2)
    frequency_vals_new = mask * frequency_vals + (1-mask) * f1 * suppress_lowf_factor * np.sign(frequency_vals - np.finfo(float).eps)

    FFI_output = complex_psi4f/(2*math.pi*1.j*frequency_vals_new)
    return FFI_output

def convert_psi4_to_strain(psi4_table, cutoff_frequency, suppress_lowf_factor=1.0):
    """
    Convert a time-domain ψ₄ waveform to gravitational-wave strain h using double integration in Fourier space.

    This function uses the Fixed-Frequency Integration (FFI) method twice to integrate ψ₄
    to obtain h, while suppressing low-frequency artifacts.

    Args:
        psi4_table (numpy.ndarray): Nx3 array with columns [time, psi4_real, psi4_imag].
        cutoff_frequency (float): Low-frequency cutoff for the FFI integration, in radians/sec.
        suppress_lowf_factor (float, optional): Factor to suppress low-frequency contributions. Defaults to 1.0.

    Returns:
        numpy.ndarray: Nx2 array with columns [time, h], where h is the complex strain (h_+ - i h_×).
    """
    time = psi4_table[:, 0]
    complexPsi = psi4_table[:, 1]+1.j*psi4_table[:, 2] # NR convention

    frequency_vals, psi4f = get_Fourier_Transform(time, complexPsi)
    dhf = integrate_using_fixed_frequency_integration(frequency_vals, psi4f, cutoff_frequency, suppress_lowf_factor)
    hf = integrate_using_fixed_frequency_integration(frequency_vals, dhf, cutoff_frequency, suppress_lowf_factor)

    time, h = get_inverse_Fourier_Transform(frequency_vals, hf, time[0])
    hTable = np.column_stack((time, h))
    return hTable

def get_cleaned_psi4(psi4_table, remove_upto=75, zero_pad=True):
    """
    Clean and preprocess a ψ₄ time series by removing initial junk radiation and resizing the array.

    The function can optionally zero-pad the array to the nearest power of two to facilitate FFTs.

    Args:
        psi4_table (numpy.ndarray): Nx3 array with columns [time, psi4_real, psi4_imag].
        remove_upto (float, optional): Time value up to which the data is considered junk and removed. Defaults to 75.
        zero_pad (bool, optional): Whether to zero-pad the array to the next power of two. Defaults to True.

    Returns:
        numpy.ndarray: Preprocessed Nx3 array with time, psi4_real, and psi4_imag. 
                       The initial junk portion is zeroed out, and the array may be zero-padded.
    """
    # what all do I want to do?
    # 1) Remove junk radiation: Time after subtracting RadialToTortoise(radius, ADMMass) should have junk radiation at 0. Removing upto 50
    # 2) Resize to power 2
    len_original = len(psi4_table[:,0])
    if zero_pad:
        nearest_power_2 = int(2**(np.ceil(np.log2(len_original))))
    else:
        nearest_power_2 = len_original
    t0, deltaT = psi4_table[0,0], np.diff(psi4_table[:,0])[1]
    print(f'Resizing from {len_original} to {nearest_power_2}')
    psi4_data_new = np.zeros((nearest_power_2, 3))

    index_at_remove_upto = np.argmin(np.abs(psi4_table[:,0]-remove_upto))
    print(f'Removing content upto = {psi4_table[index_at_remove_upto, 0]}')
    psi4_data_new[:, 0] = np.arange(nearest_power_2) * deltaT + t0
    psi4_data_new[index_at_remove_upto:len_original, 1] = psi4_table[index_at_remove_upto:len_original,1]
    psi4_data_new[index_at_remove_upto:len_original, 2] = psi4_table[index_at_remove_upto:len_original,2]
    return psi4_data_new

###########################################################################################
# Power method functions
###########################################################################################

def convert_radial_to_tortoise(r, M):
    """
    Convert a radial coordinate to the tortoise coordinate.

    The tortoise coordinate accounts for the effect of the black hole mass
    in "stretching" the radial coordinate near the horizon.

    Args:
        r (float): Radial coordinate.
        M (float): ADM mass of the black hole.

    Returns:
        float: Tortoise coordinate corresponding to the input radial coordinate.
    """
    return r + 2. * M * math.log( r / (2. * M) - 1.)

def get_ADMMass_From_TwoPunctureBBH(TwoPunc_dict):
    """
    Extract the initial ADM mass of the system from a TwoPunctures.bbh metadata dictionary.

    Args:
        TwoPunc_dict (dict): Dictionary containing metadata parsed from a
            `TwoPunctures.bbh` file.

    Returns:
        float: Initial ADM mass of the binary system.
    """

    ADMmass = float(TwoPunc_dict['metadata']['initial-adm-energy'])

    return ADMmass

def angular_momentum(x, q, m, chi1, chi2, LInitNR):
    # x is a scalar but numpy always passes arrays, triggering a warning
    # further down
    assert(x.shape == (1,))
    x = x[0]

    eta = q/(1.+q)**2
    m1 = (1.+math.sqrt(1.-4.*eta))/2.
    m2 = m - m1
    S1 = m1**2. * chi1
    S2 = m2**2. * chi2
    Sl = S1+S2
    Sigmal = S2/m2 - S1/m1
    DeltaM = m1 - m2
    mu = eta
    nu = eta
    GammaE = 0.5772156649
    e4 = -(123671./5760.)+(9037.* math.pi**2.)/1536.+(896.*GammaE)/15.+(-(498449./3456.)+(3157.*math.pi**2.)/576.)*nu+(301. * nu**2.)/1728.+(77.*nu**3.)/31104.+(1792. *math.log(2.))/15.
    e5 = -55.13
    j4 = -(5./7.)*e4+64./35.
    j5 = -(2./3.)*e5-4988./945.-656./135. * eta
    CapitalDelta = (1.-4.*eta)**0.5

    # Eq 4.7 here https://arxiv.org/pdf/1212.5520.pdf Bohe et al.
    # "Next-to-next-to-leading order spin-orbit effects in the near-zone metric
    # and precession equations of compact binaries"
    # Eq. 2.36 of https://arxiv.org/pdf/1111.5378.pdf "The First Law of Binary
    # Black Hole Mechanics in General Relativity and Post-Newtonian Theory"
    l = (eta/x**(1./2.)*(
        1. +
        x*(3./2. + 1./6.*eta) +
        x**2. *(27./8. - 19./8.*eta + 1./24.*eta**2.) +
        x**3. *(135./16. + (-6889./144. + 41./24. * math.pi**2.)*eta + 31./24.*eta**2. + 7./1296.*eta**3.) +
        x**4. *((2835./128.) + eta*j4 - (64.*eta*math.log(x)/3.))+
        x**5. *((15309./256.) + eta*j5 + ((9976./105.) + (1312.*eta/15.))*eta*math.log(x))+
        x**(3./2.)*(-(35./6.)*Sl - 5./2.*DeltaM* Sigmal) +
        x**(5./2.)*((-(77./8.) + 427./72.*eta)*Sl + DeltaM* (-(21./8.) + 35./12.*eta)*Sigmal) +
        x**(7./2.)*((-(405./16.) + 1101./16.*eta - 29./16.*eta**2.)*Sl + DeltaM*(-(81./16.) + 117./4.*eta - 15./16.*eta**2.)*Sigmal) +
        (1./2. + (m1 - m2)/2. - eta)* chi1**2. * x**2. +
        (1./2. + (m2 - m1)/2. - eta)* chi2**2. * x**2. +
        2.*eta*chi1*chi2*x**2. +
        ((13.*chi1**2.)/9. +
        (13.*CapitalDelta*chi1**2.)/9. -
        (55.*nu*chi1**2.)/9. -
        29./9.*CapitalDelta*nu*chi1**2. +
        (14.*nu**2. *chi1**2.)/9. +
        (7.*nu*chi1*chi2)/3. +
        17./18.* nu**2. * chi1 * chi2 +
        (13.* chi2**2.)/9. -
        (13.*CapitalDelta*chi2**2.)/9. -
        (55.*nu*chi2**2.)/9. +
        29./9.*CapitalDelta*nu*chi2**2. +
        (14.*nu**2. * chi2**2.)/9.)
        * x**3.))
    return l - LInitNR

def get_Cutoff_Frequency_From_TwoPuncturesBBH(TwoPunc_dict):
    """
    Estimate the cutoff frequency for fixed frequency integration from a TwoPunctures.bbh metadata dictionary.

    Args:
        TwoPunc_dict (dict): Dictionary containing metadata parsed from a
            `TwoPunctures.bbh` file. Must include keys for initial black hole
            positions, momenta, spins, and puncture ADM masses.

    Returns:
        float: Estimated cutoff GW frequency of the simulation.
    """

    position1x = float(TwoPunc_dict['metadata']['initial-bh-position1x'])
    position1y = float(TwoPunc_dict['metadata']['initial-bh-position1y'])
    position1z = float(TwoPunc_dict['metadata']['initial-bh-position1z'])
    position2x = float(TwoPunc_dict['metadata']['initial-bh-position2x'])
    position2y = float(TwoPunc_dict['metadata']['initial-bh-position2y'])
    position2z = float(TwoPunc_dict['metadata']['initial-bh-position2z'])
    momentum1x = float(TwoPunc_dict['metadata']['initial-bh-momentum1x'])
    momentum1y = float(TwoPunc_dict['metadata']['initial-bh-momentum1y'])
    momentum1z = float(TwoPunc_dict['metadata']['initial-bh-momentum1z'])
    momentum2x = float(TwoPunc_dict['metadata']['initial-bh-momentum2x'])
    momentum2y = float(TwoPunc_dict['metadata']['initial-bh-momentum2y'])
    momentum2z = float(TwoPunc_dict['metadata']['initial-bh-momentum2z'])
    spin1x = float(TwoPunc_dict['metadata']['initial-bh-spin1x'])
    spin1y = float(TwoPunc_dict['metadata']['initial-bh-spin1y'])
    spin1z = float(TwoPunc_dict['metadata']['initial-bh-spin1z'])
    spin2x = float(TwoPunc_dict['metadata']['initial-bh-spin2x'])
    spin2y = float(TwoPunc_dict['metadata']['initial-bh-spin2y'])
    spin2z = float(TwoPunc_dict['metadata']['initial-bh-spin2z'])
    mass1 = float(TwoPunc_dict['metadata']['initial-bh-puncture-adm-mass1'])
    mass2 = float(TwoPunc_dict['metadata']['initial-bh-puncture-adm-mass2'])

    angularmomentum1x = position1y * momentum1z - position1z * momentum1y
    angularmomentum1y = position1z * momentum1x - position1x * momentum1z
    angularmomentum1z = position1x * momentum1y - position1y * momentum1x

    angularmomentum2x = position2y * momentum2z - position2z * momentum2y
    angularmomentum2y = position2z * momentum2x - position2x * momentum2z
    angularmomentum2z = position2x * momentum2y - position2y * momentum2x

    angularmomentumx = angularmomentum1x + angularmomentum2x
    angularmomentumy = angularmomentum1y + angularmomentum2y
    angularmomentumz = angularmomentum1z + angularmomentum2z

    LInitNR = math.sqrt(angularmomentumx**2 + angularmomentumy**2 + angularmomentumz**2)
    S1 = math.sqrt(spin1x**2 + spin1y**2 + spin1z**2)
    S2 = math.sqrt(spin2x**2 + spin2y**2 + spin2z**2)

    M = mass1+mass2
    q = mass1/mass2
    chi1 = S1/mass1**2
    chi2 = S2/mass2**2
    # .014 is the initial guess for cutoff frequency
    omOrbPN = scipy.optimize.fsolve(angular_momentum, .014, (q, M, chi1, chi2, LInitNR))[0]
    omOrbPN = omOrbPN**(3./2.)
    omGWPN = 2. * omOrbPN
    omCutoff = 0.75 * omGWPN
    return omCutoff


def extrapolate_using_power_method(radiation_bundle):
    """
    Extrapolate gravitational-wave strain to infinity using the power-law method.

    This function takes a `RadiationBundle` containing ψ₄ data at multiple 
    extraction radii and computes the strain (h_+, h_×) at infinity by 
    performing the following steps for each included mode:
        1. Convert ψ₄ at each radius to strain using Fixed-Frequency Integration.
        2. Multiply ψ₄ by the extraction radius and shift time to the 
           corresponding tortoise coordinate.
        3. Compute the amplitude and phase of the strain.
        4. Interpolate amplitude and phase onto a common, uniform time grid.
        5. Align phases across different radii to remove 2π jumps.
        6. Fit amplitude and phase as a function of 1/radius and extrapolate 
           to infinity using polynomial fits.

    Args:
        radiation_bundle (RadiationBundle): Object containing ψ₄ data, 
            extraction radii, included modes, and related metadata.

    Returns:
        RadiationSphere: Object containing extrapolated complex strain 
        modes at infinity. Each mode is stored as a `RadiationMode` with 
        `strain_plus` (h_+) and `strain_cross` (h_×) arrays, along with 
        time and extrapolation metadata.
    """
    print(f'Using power method to extrapolate to infinity using radii: {radiation_bundle.radii_list_for_power_method}')

    modes = radiation_bundle.included_modes
    radii = radiation_bundle.radii_list_for_power_method
    if radiation_bundle.TwoPunctures_content is not None:
        f0 = get_Cutoff_Frequency_From_TwoPuncturesBBH(radiation_bundle.TwoPunctures_content)
        ADMMass = get_ADMMass_From_TwoPunctureBBH(radiation_bundle.TwoPunctures_content)
        
    else:
        print('TwoPunctures content is None, ususing default functions to calculate f0.')
        f0 = radiation_bundle.radiation_spheres[radii[0]].modes[(2, 2)].omega_start
        ADMMass = 1
    # collect amplitude and phase
    extrapolated_strains = {}
    for (el,em) in modes:
        mp_psi4_vars = []
        strain = []
        phase = []
        amp = []
        # collect strain amplitde and phase from psi4
        for i in range(len(radii)):
            
            radius = radii[i]
            time, real_part, imag_part = radiation_bundle.get_time(radius)[:, None], radiation_bundle.get_psi4_real_for_mode(el, em, radius)[:, None], radiation_bundle.get_psi4_imaginary_for_mode(el, em, radius)[:, None]
            mp_psi4 = np.hstack([time, real_part, imag_part])
            mp_psi4_vars.append(mp_psi4)

            mp_psi4_vars[i][:, 0] -= convert_radial_to_tortoise(radius, ADMMass)
            mp_psi4_vars[i][:, 1] *= radii[i]
            mp_psi4_vars[i][:, 2] *= radii[i]

            # Get strain
            hTable = convert_psi4_to_strain(mp_psi4_vars[i], f0)  # table of strain
            time = hTable[:, 0].real
            h = hTable[:, 1]
            hplus = h.real
            hcross = h.imag
            newhTable = np.column_stack((time, hplus, hcross))
            strain.append(newhTable)

            #Get phase and amplitude of strain
            h_phase = np.unwrap(np.angle(h))
            angleTable = np.column_stack((time, h_phase))     
            phase.append(angleTable)                          
            h_amp = np.absolute(h)
            ampTable = np.column_stack((time, h_amp))
            amp.append(ampTable)

        # interpolate phase and amplitude to same time grid
        tmin = max([phase[i][ 0,0] for i in range(len(phase))])
        tmax = min([phase[i][-1,0] for i in range(len(phase))])

        # smallest timestep in any series
        dtmin = min([np.amin(np.diff(phase[0][:,0])) for i in range(len(phase))])

        # uniform, common time
        t = np.arange(tmin, tmax, dtmin)

        # Interpolate phase and amplitude
        interpolation_order = 9
        for i in range(len(radii)):
            interp_function = scipy.interpolate.interp1d(amp[i][:, 0], amp[i][:, 1], kind=interpolation_order)
            resampled_amp_vals = interp_function(t)
            amp[i] = np.column_stack((t, resampled_amp_vals))

            interp_function = scipy.interpolate.interp1d(phase[i][:, 0], phase[i][:, 1], kind=interpolation_order)
            resampled_phase_vals = interp_function(t)
            # try and keep all phases at the amplitude maximum within 2pi of each other
            # alignment is between neighbhours just in case there actually ever is
            # >2pi difference between the innermost and the ohtermost detector
            if(i > 0):
                junk_time = 50
                post_junk_idx_p = amp[i-1][:,0] > junk_time
                post_junk_idx = amp[i][:,0] > junk_time
                # data comes from the same simulation, time steps must be identical
                assert((post_junk_idx_p == post_junk_idx).all())
                maxargp = np.argmax(amp[i-1][post_junk_idx_p,1])
                maxarg = np.argmax(amp[i][post_junk_idx,1])
                phase_shift = round((resampled_phase_vals[post_junk_idx][maxarg] - phase[i-1][post_junk_idx_p][maxargp,1])/(2.*math.pi))*2.*math.pi
                resampled_phase_vals -= phase_shift
            phase[i] = np.column_stack((t, resampled_phase_vals))

        # Extrapolate
        phase_extrapolation_order = 1
        amp_extrapolation_order = 2
        radii = np.asarray(radii, dtype=float)

        b_phase = np.empty(dtype=radii.dtype, shape=(len(radii), len(t)))
        b_amp = np.empty(dtype=radii.dtype, shape=(len(radii), len(t)))

        for j in range(len(radii)):
            b_phase[j] = phase[j][:, 1]
            b_amp[j] = amp[j][:, 1]

        radially_extrapolated_amp=np.polyfit(1/radii,b_amp,amp_extrapolation_order)[-1]
        radially_extrapolated_phase=np.polyfit(1/radii,b_phase,phase_extrapolation_order)[-1]

        radially_extrapolated_h_plus = radially_extrapolated_amp * np.cos(radially_extrapolated_phase)
        radially_extrapolated_h_cross = radially_extrapolated_amp * np.sin(radially_extrapolated_phase)

        extrapolated_strains[(el, em)] = RadiationMode(psi4_real=None, psi4_imaginary=None, l=el, m=em, rad=0.0, time=t, extrapolated=True, strain_plus=radially_extrapolated_h_plus, strain_cross=radially_extrapolated_h_cross)
    return RadiationSphere(mode_dict=extrapolated_strains, time=t, radius=0.0, extrapolated=True)