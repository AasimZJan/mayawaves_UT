import numpy as np
import math
import scipy.optimize
import scipy.interpolate
from mayawaves.radiation import RadiationMode, RadiationSphere
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.signal.windows import blackmanharris
import warnings

# Open questions: 
# 1) Small phase shift relative to perturbative
# 2) What does the junk radiation block do in extrapolate_using_power_method
# 3) I am using real FFT logic to surpress frequency content but here the FFT is of complex time series.
# 4) Post merger noise
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
    frequency_series_unrolled = np.fft.ifftshift(frequency_series)

    time_series = np.fft.ifft(frequency_series_unrolled, norm="ortho")

    time_vals = t0 + np.arange(len(time_series)) * 1 / (len(frequency_series_unrolled) * deltaF)
    return time_vals, time_series


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

    # information to supress low and high frequency noise in the primary extrapolation function
    amplitude_f = np.abs(hf)
    max_amplitude_f = np.max(amplitude_f)
    criterion = 10**(-5) # decrease this to increase fmax. The logic supresses frequency content where the amplitude(f) < critertaion*max_amplitude(f)
    indices_criterion = np.argwhere(amplitude_f > max_amplitude_f*criterion).flatten()
    fvals_criterion = frequency_vals[indices_criterion]
    time, h = get_inverse_Fourier_Transform(frequency_vals, hf, time[0])
    hTable= np.column_stack((time, h))

    return hTable, fvals_criterion

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

    # modes to extrapolate
    modes = radiation_bundle.included_modes

    # for loop needs to start with (2,2) mode
    assert (2,2) in modes, "(2,2) mode needs to be present."
    modes.remove((2,2))
    modes.insert(0, (2,2))

    # radii to use for extrapolation
    radii = radiation_bundle.radii_list_for_power_method

    # default values. Do we need to be exact? Information needs to be read from TwoPunctures.bbh
    ADMMass = 1

    # collect amplitude and phase
    extrapolated_strains = {}
    for (el,em) in modes:
        collect_psi4 = []
        collect_strain = []
        collect_phase = []
        collect_amp = []
        # collect strain amplitde and phase from psi4
        for i in range(len(radii)):
            radius = radii[i]
            time, real_part, imag_part = radiation_bundle.get_time(radius)[:, None], radiation_bundle.get_psi4_real_for_mode(el, em, radius)[:, None], radiation_bundle.get_psi4_imaginary_for_mode(el, em, radius)[:, None]
            psi4_tmp = np.hstack([time, real_part, imag_part])
            collect_psi4.append(psi4_tmp)

            collect_psi4[i][:, 0] -= convert_radial_to_tortoise(radius, ADMMass)
            collect_psi4[i][:, 1] *= radii[i]
            collect_psi4[i][:, 2] *= radii[i]

            # # set the lower omega cutoff for FFI. Matching mayawaves' approach 
            omega_22_start = radiation_bundle.radiation_spheres[radii[i]].modes[(2, 2)].omega_start
            if em == 0:
                omega_0 = 0.25 * omega_22_start
            else:
                omega_0 = 0.75*(abs(em) / 2) * omega_22_start
                if em < 2:
                    omega_0 = omega_22_start

            # convert to strain 
            hTable, fvals_criterion= convert_psi4_to_strain(collect_psi4[i], omega_0)  # table of strain
            time = hTable[:, 0].real
            h = hTable[:, 1]
            hplus = h.real
            hcross = h.imag
            strain_tmp = np.column_stack((time, hplus, hcross))
            collect_strain.append(strain_tmp)

            # get phase and amplitude of strain
            h_phase = np.unwrap(np.angle(h))
            phase_tmp = np.column_stack((time, h_phase))     
            collect_phase.append(phase_tmp)                          
            h_amp = np.absolute(h)
            amp_tmp = np.column_stack((time, h_amp))
            collect_amp.append(amp_tmp)

        # interpolate phase and amplitude to same time grid
        tmin = max([collect_phase[i][ 0,0] for i in range(len(collect_phase))])
        tmax = min([collect_phase[i][-1,0] for i in range(len(collect_phase))])

        # smallest timestep in any series
        dtmin = min([np.amin(np.diff(collect_phase[i][:,0])) for i in range(len(collect_phase))])

        # uniform, common time
        t = np.arange(tmin, tmax, dtmin)

        # Interpolate phase and amplitude
        interpolation_order = 9
        for i in range(len(radii)):
            interp_function = scipy.interpolate.interp1d(collect_amp[i][:, 0], collect_amp[i][:, 1], kind=interpolation_order)
            resampled_amp_vals = interp_function(t)
            collect_amp[i] = np.column_stack((t, resampled_amp_vals))

            interp_function = scipy.interpolate.interp1d(collect_phase[i][:, 0], collect_phase[i][:, 1], kind=interpolation_order)
            resampled_phase_vals = interp_function(t)

            # try and keep all phases at the amplitude maximum within 2pi of each other
            # alignment is between neighbhours just in case there actually ever is
            # >2pi difference between the innermost and the ohtermost detector
            if(i > 0):
                junk_time = 50
                post_junk_idx_p = collect_amp[i-1][:,0] > junk_time
                post_junk_idx = collect_amp[i][:,0] > junk_time
                # data comes from the same simulation, time steps must be identical
                assert((post_junk_idx_p == post_junk_idx).all())
                maxargp = np.argmax(collect_amp[i-1][post_junk_idx_p,1])
                maxarg = np.argmax(collect_amp[i][post_junk_idx,1])
                phase_shift = round((resampled_phase_vals[post_junk_idx][maxarg] - collect_phase[i-1][post_junk_idx_p][maxargp,1])/(2.*math.pi))*2.*math.pi
                resampled_phase_vals -= phase_shift
            collect_phase[i] = np.column_stack((t, resampled_phase_vals))

        # Extrapolate
        phase_extrapolation_order = 1
        amp_extrapolation_order = 1  # 2 overfits to noise
        radii = np.asarray(radii, dtype=float)

        b_phase = np.empty(dtype=radii.dtype, shape=(len(radii), len(t)))
        b_amp = np.empty(dtype=radii.dtype, shape=(len(radii), len(t)))

        for j in range(len(radii)):
            b_phase[j] = collect_phase[j][:, 1]
            b_amp[j] = collect_amp[j][:, 1]

        radially_extrapolated_amp=np.polyfit(1/radii, b_amp, amp_extrapolation_order)[-1]
        radially_extrapolated_phase=np.polyfit(1/radii, b_phase, phase_extrapolation_order)[-1]

        # construct polarizations
        radially_extrapolated_h_plus = radially_extrapolated_amp * np.cos(radially_extrapolated_phase)
        radially_extrapolated_h_cross = radially_extrapolated_amp * np.sin(radially_extrapolated_phase)

        # determine frequencies to surpress 
        fmin_clean = omega_0/(2*np.pi) # for m < 2, the low frequency noise amplitude can be close to the amplitde of the mode itself. This logic helps for such modes.
        fmax_clean = fvals_criterion.max()      
        
        # surpress frequency outside the range. sos prevents blowup, which happened atleast in one case.
        sos = butter(4, [fmin_clean, fmax_clean], btype='band', fs=1/(np.diff(t)[0]), output='sos')
        strain_plus = sosfiltfilt(sos, radially_extrapolated_h_plus)
        strain_cross = sosfiltfilt(sos, radially_extrapolated_h_cross)

   
        # window strain 
        window_length_in_time = 30
        start_window_after_max = 120
        
        if el == 2 and em ==2:
            max_time_index = np.argmax(np.sqrt(strain_plus**2 + strain_cross**2))
            max_22_time = t[max_time_index]
    
        start_window_time = max_22_time + start_window_after_max
        end_window_time = start_window_time + window_length_in_time
        if t[-1] < start_window_time:
            warnings.warn("Not enough time after max to window")
            pass
        else:
            # windowing logic starts
            end_window_time = min(end_window_time, t[-2])

            start_window_index = np.argmax(t > start_window_time)
            end_window_index = np.argmax(t > end_window_time)

            window = np.zeros(t.shape)

            # set to 1's before the windowing
            window[:start_window_index] = np.ones(start_window_index)

            # set the windowing
            window_width = end_window_index - start_window_index
            blackmanharris_window_symmetric = blackmanharris(2 * window_width)
            blackmanharris_window_half = blackmanharris_window_symmetric[window_width:]
            window[start_window_index:end_window_index] = blackmanharris_window_half

            # smooth window with low pass filter
            step_size = t[1] - t[0]
            fs_win = 1.0 / step_size
            cutoff = 1.0

            # NOTE: your original Wn expression is kept as-is.
            # If cutoff is in Hz, the usual choice would be: Wn = cutoff / (fs_win/2)
            Wn = 2 * (cutoff / (2 * np.pi)) / fs_win

            b, a = butter(4, Wn, analog=False)
            smoothed_window = filtfilt(b, a, window, axis=0)

            strain_plus  = strain_plus  * smoothed_window
            strain_cross = strain_cross * smoothed_window

        # save the extrapolated mode
        extrapolated_strains[(el, em)] = RadiationMode(psi4_real=None, psi4_imaginary=None, l=el, m=em, rad=0.0, time=t, extrapolated=True, strain_plus=strain_plus, strain_cross=strain_cross)
    return RadiationSphere(mode_dict=extrapolated_strains, time=t, radius=0.0, extrapolated=True)