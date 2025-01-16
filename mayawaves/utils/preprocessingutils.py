import os
import numpy as np
from scipy.optimize import root
import mayawaves.utils.postnewtonianutils as pn
import pathlib


def _valid_spin(dimensionless_spin_vector: np.ndarray) -> bool:
    """Check if the provided dimensionless spin vector is valid.

    Check if the given dimensionless spin vector has length of 3 and magnitude not greater than 1.

    Args:
        dimensionless_spin_vector (numpy.ndarray): dimensionless spin vector

    Returns:
        bool: whether the provided dimensionless spin is valid

    """
    # check if it's a numpy array
    if not type(dimensionless_spin_vector) == np.ndarray:
        print("The dimensionless spin has not been provided as a numpy array.")
        return False

    # check length of spin vector
    if len(dimensionless_spin_vector) != 3:
        print(f"The spin vector should have length 3, but actually has length {len(dimensionless_spin_vector)}")
        return False

    # check magnitude of spin vector is not greater than 1
    magnitude = np.linalg.norm(dimensionless_spin_vector)
    if magnitude > 1:
        print(f"The magnitude of the spin vector should be less than 1 but is actually {magnitude}")
        return False

    return True


def _is_precessing(dimensionless_spin_vector: np.ndarray) -> bool:
    """Check if the spin is precessing.

    Check whether the x and y directions of the spin are nonzero.

    Args:
        dimensionless_spin_vector (numpy.ndarray): dimensionless spin vector

    Returns:
        bool: whether the spin is precessing

    """
    if dimensionless_spin_vector[0] != 0 or dimensionless_spin_vector[1] != 0:
        return True
    return False


def _offset_from_separation(separation: float, mass_ratio: float) -> float:
    """Offset to keep center of mass at the origin.

    Compute the offset of the middle of the compact objects from the origin to keep the center of mass at the origin.

    Args:
        separation (float): the distance between the compact objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`

    Returns:
        float: offset of the center of the two objects

    """
    m2 = 1 / (mass_ratio + 1)

    x1 = separation * m2
    offset = (separation / 2) - x1

    return offset


def _initial_momentum(initial_separation: float, mass_ratio: float,
                      primary_dimensionless_spin: np.ndarray, secondary_dimensionless_spin: np.ndarray,
                      eccentricity: float) -> tuple:
    """Compute the tangential and radial momenta necessary to obtain the desired eccentricity.

    Args:
        initial_separation (float): initial distance between the compact objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): dimensionless spin of the larger compact object
        secondary_dimensionless_spin (numpy.ndarray): dimensionless spin of the smaller compact object
        eccentricity (float): desired eccentricity

    Returns:
        tuple: initial tangential momentum, initial radial momentum

    """
    qc_tangential_momentum = pn.tangential_momentum_from_separation(separation=initial_separation, mass_ratio=mass_ratio,
                                                                 primary_dimensionless_spin=primary_dimensionless_spin,
                                                                 secondary_dimensionless_spin=secondary_dimensionless_spin)
    qc_radial_momentum = pn.radial_momentum_from_separation(separation=initial_separation, mass_ratio=mass_ratio,
                                                         primary_dimensionless_spin=primary_dimensionless_spin,
                                                         secondary_dimensionless_spin=secondary_dimensionless_spin)

    # if not eccentric, return the quasicircular momentum
    if eccentricity == 0:
        return qc_tangential_momentum, qc_radial_momentum

    # otherwise, multiply by a factor to add eccentricity
    epsilon = 1 - np.sqrt(1 - eccentricity)
    momentum_modifier = 1 - epsilon
    tangential_momentum = qc_tangential_momentum*momentum_modifier
    return tangential_momentum, qc_radial_momentum


def _grid_structure(mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                    secondary_dimensionless_spin: np.ndarray) -> dict:
    """Define the grid structure for the given mass ratio.

    Given the mass ratio, compute the grid structure including number of refinement levels and their radii and
    resolutions.

    Args:
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`

    Returns:
        dict: dictionary defining all elements of the grid structure

    """
    full_domain_radius = 400

    radius_small_bh = 1/(2*(mass_ratio+1))
    radius_big_bh = mass_ratio/(2*(mass_ratio+1))

    radius_of_finest_grid = np.ceil(radius_small_bh*1.01*100.0)/100.0

    cells_on_finest_grid = 48

    maximum_factor = np.ceil(np.log2(full_domain_radius/radius_of_finest_grid))
    all_refinement_factors = [16, 15, 14, 13, 12, 11, 10, 8, 7, 5, 4, 3, 2, 1, 0]

    refinement_factors = [factor for factor in all_refinement_factors if factor <= maximum_factor]

    refinement_radii = [radius_of_finest_grid * np.power(2, factor) for factor in refinement_factors]


    refinement_levels_1 = [i for i, refinement_radius in enumerate(refinement_radii) if
                           refinement_radius > radius_big_bh and not i == 0]
    refinement_levels_2 = [i for i, refinement_radius in enumerate(refinement_radii) if
                           refinement_radius > radius_small_bh and not i == 0]
    refinement_levels_3 = [1, 2, 3] #todo is this always good?

    time_refinement_factors = []
    for i in range(len(refinement_factors)):
        if i < 5:
            time_refinement_factors.append(1)
        else:
            time_refinement_factors.append(2**(i-4))

    # symmetries
    precessing = _is_precessing(primary_dimensionless_spin) or _is_precessing(secondary_dimensionless_spin)
    if precessing:
        reflect_z = False
    else:
        reflect_z = True
    reflect_x = False
    reflect_y = False

    grid_structure = {
        "cells_on_finest_grid": cells_on_finest_grid,
        "radius_of_finest_grid": radius_of_finest_grid,
        "refinement_factors": refinement_factors,
        "refinement_levels_1": refinement_levels_1,
        "refinement_levels_2": refinement_levels_2,
        "refinement_levels_3": refinement_levels_3,
        "time_refinement_factors": time_refinement_factors,
        "resolution": cells_on_finest_grid/radius_of_finest_grid,
        "reflect_x": reflect_x,
        "reflect_y": reflect_y,
        "reflect_z": reflect_z
    }
    return grid_structure


def _estimated_memory(grid_structure: dict):
    refinement_levels_1 = grid_structure["refinement_levels_1"]
    refinement_levels_2 = grid_structure["refinement_levels_2"]
    refinement_levels_3 = grid_structure["refinement_levels_3"]
    radius_of_finest_grid = grid_structure["radius_of_finest_grid"]
    cells_on_finest_grid = grid_structure["cells_on_finest_grid"]
    refinement_factors = grid_structure["refinement_factors"]

    ghost_zones = 4

    num_grid_points = 0

    for i in range(len(refinement_levels_1)):
        refinement_level = refinement_levels_1[i]
        h = radius_of_finest_grid / cells_on_finest_grid
        dx = h * 2 ** (len(refinement_factors) - refinement_level - 1)
        radius = radius_of_finest_grid * 2 ** refinement_factors[refinement_level]
        grid_points_on_refinement = ((2 * radius / dx) + ghost_zones * 4) ** 3
        num_grid_points += grid_points_on_refinement

    for i in range(len(refinement_levels_2)):
        refinement_level = refinement_levels_2[i]
        h = radius_of_finest_grid / cells_on_finest_grid
        dx = h * 2 ** (len(refinement_factors) - refinement_level - 1)
        radius = radius_of_finest_grid * 2 ** refinement_factors[refinement_level]
        grid_points_on_refinement = ((2 * radius / dx) + ghost_zones * 4) ** 3
        num_grid_points += grid_points_on_refinement

    for i in range(len(refinement_levels_3)):
        refinement_level = refinement_levels_3[i]
        h = radius_of_finest_grid / cells_on_finest_grid
        dx = h * 2 ** (len(refinement_factors) - refinement_level - 1)
        radius = radius_of_finest_grid * 2 ** refinement_factors[refinement_level]
        grid_points_on_refinement = ((2 * radius / dx) + ghost_zones * 4) ** 3
        num_grid_points += grid_points_on_refinement

    reflect_x = grid_structure["reflect_x"]
    reflect_y = grid_structure["reflect_y"]
    reflect_z = grid_structure["reflect_z"]
    symmetry_factor = 1 * (0.5 if reflect_x else 1) * (0.5 if reflect_y else 1) * (0.5 if reflect_z else 1)

    num_grid_functions = 229
    bits_to_bytes = 0.125
    bits_per_double = 64
    gigabyte_factor = 1e-9
    total_memory = (num_grid_points * num_grid_functions * symmetry_factor) \
                   * bits_per_double * bits_to_bytes * gigabyte_factor

    print(f'Estimated {total_memory} Gigabytes of memory required')

    stampede_memory_per_node = 80  # todo hone in on this number

    print(f'Estimated number of stampede nodes required: {int(np.ceil(total_memory/stampede_memory_per_node))}')

    return total_memory


def _create_header(mass_ratio: float, initial_separation: float, primary_dimensionless_spin: np.ndarray,
                   secondary_dimensionless_spin: np.ndarray, tangential_momentum: float, radial_momentum: float) -> str:
    """Create the header for the parameter file for the given system.

    Given the desired parameters and the provided momenta, create the header for the .rpar file.

    Args:
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        initial_separation (float): initial distance between the compact objects
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object
        tangential_momentum (float): the tangential momentum
        radial_momentum (float): the radial momentum

    Returns:
        str: header for the rpar file

    """
    m1 = mass_ratio / (mass_ratio + 1.0)
    m2 = 1.0 / (mass_ratio + 1.0)

    offset = _offset_from_separation(separation=initial_separation, mass_ratio=mass_ratio)
    par_b = initial_separation / 2.0
    ppx = -1*radial_momentum
    ppy = tangential_momentum
    ppz = 0

    primary_dimensional_spin = primary_dimensionless_spin * m1 ** 2
    secondary_dimensional_spin = secondary_dimensionless_spin * m2 ** 2

    spx = primary_dimensional_spin[0]
    spy = primary_dimensional_spin[1]
    spz = primary_dimensional_spin[2]
    smx = secondary_dimensional_spin[0]
    smy = secondary_dimensional_spin[1]
    smz = secondary_dimensional_spin[2]
    qq = mass_ratio

    rpar_header = f"""$offset = {offset:.6f};
$par_b  = {par_b:.6f};

$ppx     = {ppx:.6f};
$ppy     = {ppy:.6f};
$ppz     = {ppz:.6f};

$spx     = {spx:.6f};
$spy     = {spy:.6f};
$spz     = {spz:.6f};

$smx     = {smx:.6f};
$smy     = {smy:.6f};
$smz     = {smz:.6f};

$qq      = {qq:.6f};"""

    return rpar_header


def _assemble_parameter_file(grid_structure: dict, header: str, filepath: str, mass_ratio: float,
                             primary_dimensionless_spin: np.ndarray, secondary_dimensionless_spin: np.ndarray):
    """Create and save the parameter file.

    Given the provided grid structure, header, and binary parameters, create and save the parameter file.

    Args:
        grid_structure (dict): dictionary defining the structure of the grid
        header (str): header for the .rpar file
        filepath (str): filepath to the parameter file to create
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    """
    CURR_DIR = pathlib.Path(__file__).parent.resolve()
    rpar_body = os.path.join(CURR_DIR, "rpar_bodies/rpar_skeleton.txt")

    high_q = mass_ratio >= 3
    precessing = _is_precessing(primary_dimensionless_spin) or _is_precessing(secondary_dimensionless_spin)

    # construct symmetries
    if precessing:
        zmin_shiftout = "0"
        zmin = "-$R0"
        reflectz = "no"
    else:
        zmin_shiftout = "1"
        zmin = "0"
        reflectz = "yes"

    max_runtime = 24 * 60 - 30
    final_time = 2399

    # output frequency
    out_1d_every = 0
    out_2d_every = 0
    out_3d_every = 0

    # construct grid structure
    cells_on_finest_grid = grid_structure["cells_on_finest_grid"]
    radius_of_finest_grid = grid_structure["radius_of_finest_grid"]
    refinement_factors = grid_structure["refinement_factors"]
    maxrl = len(refinement_factors)
    maxrl_minus_1 = maxrl - 1
    maxrl_minus_2 = maxrl - 2

    refinement_levels = ""
    refinement_levels_as_list = "("
    grid_structure_string = "# rl\tdx\tResolution\n"
    for i, factor in enumerate(refinement_factors):
        if i == 0:
            level_string = f"$R0 = $rf*2**{factor};\n"
            refinement_levels_as_list = refinement_levels_as_list + f"$R{i},"
        elif factor == 0:
            level_string = f"$r{i}= $rf;"
            refinement_levels_as_list = refinement_levels_as_list + f"$r{i})"
        else:
            level_string = f"$r{i} = $rf*2**{factor};\n"
            refinement_levels_as_list = refinement_levels_as_list + f"$r{i},"

        refinement_levels = refinement_levels + level_string
        grid_structure_string = grid_structure_string + f"# {i}\t$dx[{i}]\tM/$res[{i}]\t$rr[{i}]\n"
    grid_structure_string = grid_structure_string + "# dt = $dt"

    # construct refinement levels per BH and central
    refinement_levels_1 = grid_structure["refinement_levels_1"]
    refinement_levels_2 = grid_structure["refinement_levels_2"]
    refinement_levels_3 = grid_structure["refinement_levels_3"]

    carpet_radii_1 = ""
    for level in refinement_levels_1:
        carpet_radii_1 = carpet_radii_1 + f"CarpetRegrid2::radius_1[{level}] = $r{level}\n"

    carpet_radii_2 = ""
    for level in refinement_levels_2:
        carpet_radii_2 = carpet_radii_2 + f"CarpetRegrid2::radius_2[{level}] = $r{level}\n"

    carpet_radii_3 = ""
    for level in refinement_levels_3:
        carpet_radii_3 = carpet_radii_3 + f"CarpetRegrid2::radius_3[{level}] = $r{level}\n"

    maxrl_1 = len(refinement_levels_1) + 1
    maxrl_2 = len(refinement_levels_2) + 1
    maxrl_3 = len(refinement_levels_3) + 1

    # construct time refinement factors
    time_refinement_factors = '"['
    time_refinement_factor_values = grid_structure["time_refinement_factors"]
    for value in time_refinement_factor_values:
        if not value == time_refinement_factor_values[-1]:
            time_refinement_factors = time_refinement_factors + f'{value},'
        else:
            time_refinement_factors = time_refinement_factors + f'{value}]"'

    # construct dissipation
    dissipation = f""
    for i in range(len(refinement_factors)):
        if i < 4:
            dissipation += f"Dissipation::epsdis_for_level[{i}]\t= 0.5\n"
        else:
            dissipation += f"Dissipation::epsdis_for_level[{i}]\t= 0.1\n"

    # construct shift condition / external eta beta
    # mass ratio dependent
    if high_q:
        external_eta_beta_thorn = " ExternalEtaBeta"
        eta_beta = 1.31
        shift_condition = "NASAExternalEtaBeta"
        external_eta_beta = """#############################################################
# ExternalEtaBeta
#############################################################

ExternalEtaBeta::n                      = 1
ExternalEtaBeta::w1                     = 12
ExternalEtaBeta::w2                     = 12
ExternalEtaBeta::Mplus                  = $target_mplus
ExternalEtaBeta::Mminus                 = $target_mminus
ExternalEtaBeta::surfindex1             = 0
ExternalEtaBeta::surfindex2             = 1
ExternalEtaBeta::etaBeta_form           = "polynomial"
ExternalEtaBeta::rescale_etaBeta        = "tanh"
ExternalEtaBeta::Rcutoff                = 150
ExternalEtaBeta::sigma			= 15"""
        external_eta_beta_output_variable = " ExternalEtaBeta::etaBeta"
    else:
        external_eta_beta_thorn = ""
        eta_beta = 1.35
        shift_condition = "NASA6th"
        external_eta_beta = ""
        external_eta_beta_output_variable = ""

    with open(rpar_body, 'r') as f:
        text = f"{f.read()}".format(**locals())
        with open(filepath, 'w') as parfile:
            parfile.write(text)


def _rpar_filename(mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                   secondary_dimensionless_spin: np.ndarray, initial_separation: float, eccentricity: float,
                   grid_structure: dict) -> str:
    """The name for the parameter file.

    Generate the name for the simulation such as D11_q1_a1_0_0_0_a2_0_0_0_m140_e0.2.rpar where 11 is the separation,
    the mass ratio (q) is 1, both compact objects are nonspinning, there are 140 points per M in the finest resolution,
    and the eccentricity is 0.2.

    Args:
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object
        initial_separation (float): initial distance between the compact objects
        eccentricity (float): desired eccentricity
        grid_structure (dict): dictionary defining the grid structure

    Returns:
        str: the name for the parameter file

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]
    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]
    resolution = grid_structure["resolution"]
    if type(resolution) != int:
        resolution = round(resolution*100.0)/100.0

    if type(initial_separation) != int:
        initial_separation = round(initial_separation*100.0)/100.0

    if eccentricity == 0:
        filename = f"D{initial_separation}_q{mass_ratio}_a1_{a1x}_{a1y}_{a1z}_a2_{a2x}_{a2y}_{a2z}_m{resolution}.rpar"
    else:
        filename = f"D{initial_separation}_q{mass_ratio}_a1_{a1x}_{a1y}_{a1z}_a2_{a2x}_{a2y}_{a2z}_m{resolution}" \
                   f"_e{eccentricity}.rpar"
    return filename


def create_bbh_parameter_file(output_directory, mass_ratio: float,
                              primary_dimensionless_spin: np.ndarray = np.array([0, 0, 0]),
                              secondary_dimensionless_spin: np.ndarray = np.array([0, 0, 0]),
                              initial_separation: float = None, initial_orbital_frequency: float = None,
                              eccentricity: float = 0):
    """Create a parameter file given the initial parameters of the system.

    Args:
        output_directory (str): location to store the created parameter file
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`.
        primary_dimensionless_spin (:obj:`numpy.ndarray`, optional): the dimensionless spin of the larger object.
            Defaults to (0, 0, 0).
        secondary_dimensionless_spin (:obj:`numpy.ndarray`, optional): the dimensionless spin of the smaller object.
            Defaults to (0, 0, 0).
        initial_separation (:obj:`float`, optional): initial distance between the compact objects. Defaults to 12.
        initial_orbital_frequency (:obj:`float`, optional): initial orbital frequency. Only provide if not providing the
            initial separation.
        eccentricity (:obj:`float`, optional): desired eccentricity. Defaults to 0.

    """
    if not _valid_spin(primary_dimensionless_spin):
        raise IOError("Primary dimensionless spin is not valid.")

    if not _valid_spin(secondary_dimensionless_spin):
        raise IOError("Secondary dimensionless spin is not valid.")

    # establish the initial separation

    # raise error if both initial separation and initial frequency are provided
    if initial_separation is not None and initial_orbital_frequency is not None:
        raise IOError(
            'Both initial separation and initial frequency were provided. Please provide only one or the other.')

    # default to separation of 12 if neither initial separation nor initial frequency is provided
    if initial_separation is None and initial_orbital_frequency is None:
        initial_separation = 12

    # if frequency is given, convert to separation
    if initial_orbital_frequency is not None:
        initial_separation = pn.separation_from_orbital_frequency(orbital_frequency=initial_orbital_frequency,
                                                                  mass_ratio=mass_ratio,
                                                                  primary_dimensionless_spin=primary_dimensionless_spin,
                                                                  secondary_dimensionless_spin=
                                                                  secondary_dimensionless_spin)

    # determine initial momenta
    tangential_momentum, radial_momentum = _initial_momentum(initial_separation=initial_separation,
                                                             mass_ratio=mass_ratio,
                                                             primary_dimensionless_spin=primary_dimensionless_spin,
                                                             secondary_dimensionless_spin=secondary_dimensionless_spin,
                                                             eccentricity=eccentricity)

    # determine grid structure
    grid_structure = _grid_structure(mass_ratio=mass_ratio,
                                     primary_dimensionless_spin=primary_dimensionless_spin,
                                     secondary_dimensionless_spin=secondary_dimensionless_spin)

    # estimate memory
    estimated_memory = _estimated_memory(grid_structure=grid_structure)

    # create header
    header = _create_header(mass_ratio=mass_ratio, initial_separation=initial_separation,
                            primary_dimensionless_spin=primary_dimensionless_spin,
                            secondary_dimensionless_spin=secondary_dimensionless_spin,
                            tangential_momentum=tangential_momentum, radial_momentum=radial_momentum)

    # create filename
    filename = _rpar_filename(mass_ratio=mass_ratio, primary_dimensionless_spin=primary_dimensionless_spin,
                              secondary_dimensionless_spin=secondary_dimensionless_spin,
                              initial_separation=initial_separation, eccentricity=eccentricity,
                              grid_structure=grid_structure)
    filepath = os.path.join(output_directory, filename)

    # create parameter file
    _assemble_parameter_file(grid_structure=grid_structure, header=header,
                             filepath=filepath, mass_ratio=mass_ratio,
                             primary_dimensionless_spin=primary_dimensionless_spin,
                             secondary_dimensionless_spin=secondary_dimensionless_spin)

