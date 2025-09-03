# File for extrapolation 

import numpy as np
import math
import scipy.optimize
import scipy.interpolate
from mayawaves.radiation import RadiationMode, RadiationSphere

def RadialToTortoise(r, M):
    """
    Convert the radial coordinate to the tortoise coordinate

    r = radial coordinate
    M = ADMMass used to convert coordinate
    return = tortoise coordinate value
    """
    return r + 2. * M * math.log( r / (2. * M) - 1.)

#Convert modified psi4 to strain
def psi4ToStrain(mp_psi4, f0):
    """
    Convert the input mp_psi4 data to the strain of the gravitational wave

    mp_psi4 = Weyl scalar result from simulation
    f0 = cutoff frequency
    return = strain (h) of the gravitational wave
    """
    #TODO: Check for uniform spacing in time
    t0 = mp_psi4[:, 0]
    list_len = len(t0)
    complexPsi = mp_psi4[:, 1]+1.j*mp_psi4[:, 2]

    freq, psif = myFourierTransform(t0, complexPsi)
    dhf = ffi(freq, psif, f0)
    hf = ffi(freq, dhf, f0)

    time, h = myFourierTransformInverse(freq, hf, t0[0])
    hTable = np.column_stack((time, h))
    return hTable

#Fixed frequency integration
# See https://arxiv.org/abs/1508.07250 for method
def ffi(freq, data, f0):
    """
    Integrates the data according to the input frequency and cutoff frequency

    freq = fourier transform frequency
    data = input on which ffi is performed
    f0 = cutoff frequency
    """
    f1 = f0/(2*math.pi)
    fs = freq
    gs = data
    mask1 = (np.sign((fs/f1) - 1) + 1)/2.
    mask2 = (np.sign((-fs/f1) - 1) + 1)/2.
    mask = 1 - (1 - mask1) * (1 - mask2)
    fs2 = mask * fs + (1-mask) * f1 * np.sign(fs - np.finfo(float).eps)
    new_gs = gs/(2*math.pi*1.j*fs2)
    return new_gs

#Fourier Transform
def myFourierTransform(t0, complexPsi):
    """
    Transforms the complexPsi data to frequency space

    t0 = time data points
    complexPsi = data points of Psi to be transformed
    """
    psif = np.fft.fft(complexPsi, norm="ortho")
    l = len(complexPsi)
    n = int(math.floor(l/2.))
    newpsif = psif[l-n:]
    newpsif = np.append(newpsif, psif[:l-n])
    T = np.amin(np.diff(t0))*l
    freq = range(-n, l-n)/T
    return freq, newpsif

#Inverse Fourier Transform
def myFourierTransformInverse(freq, hf, t0):
    l = len(hf)
    n = int(math.floor(l/2.))
    newhf = hf[n:]
    newhf = np.append(newhf, hf[:n])
    amp = np.fft.ifft(newhf, norm="ortho")
    df = np.amin(np.diff(freq))
    time = t0 + range(0, l)/(df*l)
    return time, amp


def getADMMassFromTwoPunctureBBH(config):
    """
    Determine cutoff frequency of simulation

    meta_filename = path to TwoPunctures.bbh
    return = initial ADM mass of system
    """

    ADMmass = float(config['metadata']['initial-adm-energy'])

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
    GammaE = 0.5772156649;
    e4 = -(123671./5760.)+(9037.* math.pi**2.)/1536.+(896.*GammaE)/15.+(-(498449./3456.)+(3157.*math.pi**2.)/576.)*nu+(301. * nu**2.)/1728.+(77.*nu**3.)/31104.+(1792. *math.log(2.))/15.
    e5 = -55.13
    j4 = -(5./7.)*e4+64./35.
    j5 = -(2./3.)*e5-4988./945.-656./135. * eta;
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

def getCutoffFrequencyFromTwoPuncturesBBH(config):
    """
    Determine cutoff frequency of simulation

    meta_filename = path to TwoPunctures.bbh
    return = cutoff frequency
    """

    position1x = float(config['metadata']['initial-bh-position1x'])
    position1y = float(config['metadata']['initial-bh-position1y'])
    position1z = float(config['metadata']['initial-bh-position1z'])
    position2x = float(config['metadata']['initial-bh-position2x'])
    position2y = float(config['metadata']['initial-bh-position2y'])
    position2z = float(config['metadata']['initial-bh-position2z'])
    momentum1x = float(config['metadata']['initial-bh-momentum1x'])
    momentum1y = float(config['metadata']['initial-bh-momentum1y'])
    momentum1z = float(config['metadata']['initial-bh-momentum1z'])
    momentum2x = float(config['metadata']['initial-bh-momentum2x'])
    momentum2y = float(config['metadata']['initial-bh-momentum2y'])
    momentum2z = float(config['metadata']['initial-bh-momentum2z'])
    spin1x = float(config['metadata']['initial-bh-spin1x'])
    spin1y = float(config['metadata']['initial-bh-spin1y'])
    spin1z = float(config['metadata']['initial-bh-spin1z'])
    spin2x = float(config['metadata']['initial-bh-spin2x'])
    spin2y = float(config['metadata']['initial-bh-spin2y'])
    spin2z = float(config['metadata']['initial-bh-spin2z'])
    mass1 = float(config['metadata']['initial-bh-puncture-adm-mass1'])
    mass2 = float(config['metadata']['initial-bh-puncture-adm-mass2'])

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

    print(f'Using power method to extrapolate to infinity using radii: {radiation_bundle.radii_list_for_power_method}')
    f0 = getCutoffFrequencyFromTwoPuncturesBBH(radiation_bundle.TwoPunctures_content)
    ADMMass = getADMMassFromTwoPunctureBBH(radiation_bundle.TwoPunctures_content)
    modes = radiation_bundle.included_modes
    radii = radiation_bundle.radii_list_for_power_method
    print(f0, ADMMass)

    # collect amplitude and phase
    extrapolated_strains = {}
    for (el,em) in modes:
        mp_psi4_vars = []
        strain = []
        phase = []
        amp = []
        for i in range(len(radii)):
            
            radius = radii[i]
            time, real_part, imag_part = radiation_bundle.get_time(radius)[:, None], radiation_bundle.get_psi4_real_for_mode(el, em, radius)[:, None], radiation_bundle.get_psi4_imaginary_for_mode(el, em, radius)[:, None]
            mp_psi4 = np.hstack([time, real_part, imag_part])
    #         if padding > 0.:
    #             # construct a dataset of times after the end of the simulation
    #             # with all values being zero
    #             dt = mp_psi4[1,0] - mp_psi4[0,0]
    #             diff_dt = np.abs(np.diff(mp_psi4[:,0]) - dt)
    #             if(np.amax(diff_dt / dt) > 1e-4): # timestep not constant to 1e-4
    #                 raise ValueError("Time step not constant")
    #             zeros = np.zeros_like(mp_psi4, shape=(int(np.ceil(padding/dt)), mp_psi4.shape[1]))
    #             times = mp_psi4[-1,0] + np.arange(1, 1+zeros.shape[0], 1)*dt
    #             zeros[:,0] = times
    #             mp_psi4 = np.concatenate((mp_psi4, zeros))

            mp_psi4_vars.append(mp_psi4)

            mp_psi4_vars[i][:, 0] -= RadialToTortoise(radius, ADMMass)
            mp_psi4_vars[i][:, 1] *= radii[i]
            mp_psi4_vars[i][:, 2] *= radii[i]

    #         #Fixed-frequency integration twice to get strain
    #         #-----------------------------------------------------------------
    #         # Strain Conversion
    #         #-----------------------------------------------------------------

            hTable = psi4ToStrain(mp_psi4_vars[i], f0)  # table of strain

            time = hTable[:, 0].real
            h = hTable[:, 1]
            hplus = h.real
            hcross = h.imag
            newhTable = np.column_stack((time, hplus, hcross))
            strain.append(newhTable)

            #-------------------------------------------------------------------
            # Analysis of Strain
            #-------------------------------------------------------------------
            #Get phase and amplitude of strain
            h_phase = np.unwrap(np.angle(h))
            # print(len(h_phase), "h_phase length")
            # print(len(time), "time length")
            angleTable = np.column_stack((time, h_phase))     ### start here
            phase.append(angleTable)                          ### time here
            h_amp = np.absolute(h)
            ampTable = np.column_stack((time, h_amp))
            amp.append(ampTable)

        #----------------------------------------------------------------------
        # Extrapolation
        #----------------------------------------------------------------------

        # get common range in times
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
                # for some modes (post 2,2) the initial junk can be the
                # largest amplitude contribution, so w try to skip it
                # when looking for maxima
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

        #Extrapolate
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

        #extrapolated_strains[el,em] = np.column_stack((t, radially_extrapolated_h_plus, radially_extrapolated_h_cross))
        extrapolated_strains[(el, em)] = RadiationMode(psi4_real=None, psi4_imaginary=None, l=el, m=em,
                             rad=0.0, time=t, extrapolated=True, strain_plus=radially_extrapolated_h_plus, strain_cross=radially_extrapolated_h_cross)
    return RadiationSphere(mode_dict=extrapolated_strains, time=t, radius=0.0,
                                              extrapolated=True)