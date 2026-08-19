import numpy as np
import openturns as ot
ot.Log.Show(ot.Log.NONE)
import time


def input_OpenTurns(
    X=None,
    descriptions=None,
    myLSF=None,
    failure_threshold=0.0,
    input_distribution=None,
):
    """Set up OpenTURNS objects for MCS and FORM analysis.

    Parameters
    ----------
    X : list of ot.Distribution, optional
        Marginal distributions. They are combined independently when
        input_distribution is not supplied.
    descriptions : list of str, optional
        Variable descriptions.
    myLSF : callable
        Limit-state function returning one output value.
    failure_threshold : float
        Failure occurs when the LSF is below this threshold.
    input_distribution : ot.Distribution, optional
        Pre-built multivariate distribution, including its copula. Use this
        for correlated hydraulic variables.
    """
    global inputDistribution, inputRandomVector
    global myfunction, outputvector, failureevent, failureThreshold, optimAlgo
    global start_pt, start, algo

    if myLSF is None:
        raise ValueError("myLSF must be provided.")

    if input_distribution is None:
        if X is None or len(X) == 0:
            raise ValueError(
                "Provide X for independent inputs or input_distribution "
                "for a correlated multivariate model."
            )
        inputDistribution = ot.JointDistribution(X)
    else:
        inputDistribution = input_distribution

    dimension = int(inputDistribution.getDimension())
    if descriptions is None:
        descriptions = list(inputDistribution.getDescription())
    if len(descriptions) != dimension:
        raise ValueError(
            f"Expected {dimension} descriptions, received {len(descriptions)}."
        )

    inputDistribution.setDescription(descriptions)
    inputRandomVector = ot.RandomVector(inputDistribution)

    myfunction = ot.PythonFunction(dimension, 1, myLSF)

    # Vector obtained by applying limit state function to X1 and X2
    outputvector = ot.CompositeRandomVector(myfunction, inputRandomVector)

    # Define failure event: here when the limit state function takes negative values
    failureThreshold = failure_threshold
    failureevent = ot.ThresholdEvent(outputvector, ot.Less(), failureThreshold)
    failureevent.setName(f'LSF inferior to {failureThreshold}')

    optimAlgo = ot.AbdoRackwitz()
    optimAlgo.setMaximumIterationNumber(1000)
    optimAlgo.setMaximumAbsoluteError(1e-7)
    optimAlgo.setMaximumRelativeError(1e-7)
    optimAlgo.setMaximumResidualError(1e-7)
    optimAlgo.setMaximumConstraintError(1e-7)

def failure_domain_reachable(myLSF, dist, n_grid=9, p_tail=1e-10):
    """Coarse check: can the LSF reach the failure threshold anywhere
    in the input support? Returns (reachable: bool, min_Z: float)."""
    import itertools
    dim = dist.getDimension()
    grids = []
    for i in range(dim):
        m = dist.getMarginal(i)
        lo = m.computeQuantile(p_tail)[0]
        hi = m.computeQuantile(1.0 - p_tail)[0]
        grids.append(np.linspace(lo, hi, n_grid))
    min_Z = min(myLSF(list(pt))[0] for pt in itertools.product(*grids))
    return bool(min_Z < failureThreshold), float(min_Z)

def run_FORM_analysis(algorithm: str = "AbdoRackwitz"):
    start = time.time()


    start_pt = inputDistribution.getMean()
    std_pt = inputDistribution.getStandardDeviation()
    descriptions = list(inputDistribution.getDescription())

    # Move toward the expected failure region, when present.
    shifts = {
        "tan_phi": -2.0,
        "theta_R":   -2.0,
        "phi_stone": -2.0,
        "hgwl": 2.0,
        "SWL": 2.0,
        "Hm0": 2.0,
    }

    for variable, sigma_shift in shifts.items():
        if variable in descriptions:
            index = descriptions.index(variable)
            start_pt[index] += sigma_shift * std_pt[index]
    if algorithm == "Cobyla":
        optimAlgo_local = ot.Cobyla()
        optimAlgo_local.setMaximumIterationNumber(2000)
    else:
        optimAlgo_local = optimAlgo
    algo = ot.FORM(optimAlgo_local, failureevent, start_pt)


    try:
        algo.run()
    except RuntimeError as e:
        # An incomplete result can look like Pf=0 even when the mean state fails.
        detail = str(e).removeprefix("Exception : ").strip()
        raise RuntimeError(
            "FORM failed to find a valid design point. "
            "Do not use the incomplete FORM result as a failure probability. "
            f"OpenTURNS: {detail}"
        ) from e

    result = algo.getResult()
    x_star = result.getPhysicalSpaceDesignPoint()
    u_star = result.getStandardSpaceDesignPoint()
    pf     = result.getEventProbability()
    beta   = result.getHasoferReliabilityIndex()

    end = time.time()
    print(f'The FORM analysis took {end-start:.3f} seconds')
    print('FORM result, pf = {:.6e}'.format(pf))
    print('FORM result, reliability index beta (Hasofer) = {:.3f}\n'.format(beta))
    print('The design point in the u space: ', u_star)
    print('The design point in the x space: ', x_star)

    return result, x_star, u_star, pf, beta

def run_MonteCarloSimulation(mc_size, return_details=False):
    '''Run MCS using OpenTurns and return the probability of failure.

    Inputs:
    mc_size (int): Number of samples to generate.
    return_details (bool): Return diagnostics in addition to the estimate.

    Returns:
    pf_mc (float) or dict: Probability of failure, optionally with diagnostics.
    '''


    # Start timer
    start = time.time()
    if mc_size <= 0:
        raise ValueError("mc_size must be a positive integer")

    outputSample = outputvector.getSample(mc_size)
    output_array = np.asarray(outputSample, dtype=float).reshape(-1)

    number_failures = int(np.count_nonzero(output_array < failureThreshold))
    pf_mc = number_failures / mc_size

    # End timer and print the time
    end = time.time()
    print(f'The MCS took {end-start:.3f} seconds to '+
          f'evaluate {mc_size} samples.')
    
    print(f'Failures: {number_failures}/{mc_size}')
    print('pf for MCS: ', pf_mc)

    if return_details:
        return {
            "pf": pf_mc,
            "failure_count": number_failures,
            "sample_size": mc_size,
            "failure_threshold": failureThreshold,
            "mean_limit_state": float(np.mean(output_array)),
            "minimum_limit_state": float(np.min(output_array)),
        }

    return pf_mc



def importance_factors(result):
    """Compute and print the importance factors using FORM results.
    
    Inputs:
    result (ot.FORMResult): The result of the FORM analysis.

    Returns:
    alpha_ot (list): Importance factors from OpenTURNs.
    alpha (list): Importance factors based on the normal vector in U-space.
    sens (list): Sensitivity of the beta to the multivariate distribution.
    """
    print(f'--- FORM Importance Factors (alpha) ---')
    import matplotlib.pyplot as plt
    plt.ion()
    print()
    alpha_ot = result.getImportanceFactors()
    print(f'\nImportance factors, from OpenTURNs:')
    [print(f'  {i:6.3f}') for i in alpha_ot]

    u_star = result.getStandardSpaceDesignPoint()
    inverseTransform = inputDistribution.getInverseIsoProbabilisticTransformation()
    failureBoundaryStandardSpace = ot.ComposedFunction(myfunction, inverseTransform)
    du0 = failureBoundaryStandardSpace.getGradient().gradient(u_star)
    g_grad = np.array(du0).transpose()[0]
    alpha = -g_grad/np.linalg.norm(g_grad)
    print('\nImportance factors, based on normal vector in U-space = ')
    [print(f'  {i:6.3f}') for i in alpha]
    print('Note: this will be different from'
          + ' result.getImportanceFactors()'
          + '\nif there are resistance variables.')

    sens = result.getHasoferReliabilityIndexSensitivity()

    print(f'\nSensitivity of Reliability Index to Multivariate Distribution')
    for i, j in enumerate(result.getHasoferReliabilityIndexSensitivity()):
        print(f'\nDistribution item number: {i}')
        print(f'  Item name: {j.getName()}')
    for k, l in zip(j.getDescription(), j):
        print(f'    {l:+6.3e} for parameter {k}')

    return alpha_ot, alpha, sens
