import numpy as np
from numba import njit, vectorize, float64, prange  
from pyparsing import line
from j1 import j1
from numba.typed import Dict
from kronrod import kronrod_points_dict
from tuned_quad import RegisteredParametersDictType, ParametersDictType, tuned_quad_integrate
from quad_tuning import tune

a = 0
b = np.pi/2


@vectorize([float64(float64)])
def J1x_nb(x):
    return j1(x)/x if x != 0 else 0.5

reg_params2 = Dict.empty(*RegisteredParametersDictType)
reg_params2['A'] = np.geomspace(1, 1000000, 15)
reg_params2['B'] = np.geomspace(1, 1000000, 15)


@njit
def integrand_2param(
    x, 
    params):
    A = params['A']
    B = params['B']

    return (np.sinc(A * np.cos(x)/np.pi) * J1x_nb(B * np.sin(x)))**2*np.sin(x)

@njit(parallel=True)
def computeIq_2param(q, L, R, tuned_quad, kronrod_points_dict):
    Iq = np.empty_like(q, dtype=np.float64)
    
    for i in prange(len(q)):
        params = Dict.empty(*ParametersDictType)
        A = q[i]*L/2
        B = q[i]*R
        params["A"] = A
        params["B"] = B
        Iq[i] = tuned_quad_integrate(tuned_quad, integrand_2param, a, b, params, kronrod_points_dict)
    return Iq


if __name__ == "__main__":
    kronrod_space = np.concatenate([np.arange(10, 999, 20, dtype=int), np.arange(1050, 11850, 50, dtype=int),  np.arange(12050, 19950, 100, dtype=int)] )
    tuned_quad_params2 = tune("params2_new_test_15", integrand_2param, a, b, reg_params2, 1e-5, n_kronrod=kronrod_space, update=True)
    
    q_points = np.array([0.000549, 0.0005746905736831113, 0.0006015833433155252,    0.0006297345659165824, 0.0006592031310650013,    0.0006900506840900704, 0.0007223417550275881,    0.0007561438936103278, 0.0007915278105754121,    0.0008285675255841902, 0.0008673405220640696,    0.0009079279092961918, 0.000950414592088044, 0.0009948894483859357,   0.0010414455151988844, 0.00109018018322286, 0.0011411954005725038,    0.0011945978860465212, 0.0012504993523728735,    0.001309016739900776, 0.001370272461228366, 0.0014343946572777746,    0.0015015174653532858, 0.0015717812997433369,    0.0016453331454533413, 0.001722326865683822, 0.001802923523697032,    0.0018872917197453982, 0.001975607943766609, 0.002068056944583143,    0.002164832116378558, 0.0022661359032590473, 0.002372180222746517,    0.002483186909089121, 0.00259938817731661, 0.0027210271090112553,    0.00284835816081052, 0.0029816476967052265, 0.003121174545246728,    0.003267230582828718, 0.0034201213442638318, 0.003580166661932304,    0.003747701334839754, 0.003923075828983646, 0.004106657010493549,    0.004298828913078885, 0.004499993541389527, 0.00471057171196984,    0.004931003933565359, 0.005161751328623596, 0.0054032965979166825,    0.005656145030303718, 0.005920825559745213, 0.006197891871780709,    0.006487923561784237, 0.006791527347420688, 0.007109338337839223,    0.007442021362258942, 0.007790272360725975, 0.008154819839951356,    0.008536426397275157, 0.008935890315944847, 0.009354047235044993,    0.009791771897571825, 0.010249979980309137, 0.010729630009333812,    0.011231725365157699, 0.011757316381700599, 0.012307502543485178,    0.012883434785649992, 0.01348631790159225, 0.014117413063276626,    0.014778040459482495, 0.015469582057508613, 0.016193484494112186,    0.01695126210173023, 0.01774450007631354, 0.01857485779340006,    0.019444072279364833, 0.020353961845107585, 0.021306429889779658,   0.02230346888250722, 0.023347164530439883, 0.024439700141844222,    0.025583361193369174, 0.026780540111037483, 0.028033741274964857,   0.029345586258275907, 0.03071881931117631, 0.03215631310165348,   0.03366107472481417, 0.03523625199343086, 0.036885140022855104,   0.03861118812407374, 0.04041800701932692, 0.04230937639538253,    0.04428925281026789, 0.04636177796999855, 0.04853128739261839,    0.05080231947767544, 0.05317962500010563, 0.05566817704838513,    0.05827318142774036, 0.061000087550178723, 0.06385459983412063,   0.06684268963747965, 0.06997060774915359, 0.07324489746505833,   0.07667240827605651, 0.08026031019641647, 0.08401610876277418,   0.08794766073497524, 0.09206319053164037, 0.09637130743483782,   0.10088102359985145, 0.10560177290771966, 0.1105434306999838,    0.11571633443692701, 0.12113130532252157, 0.1267996709413188,    0.13273328895463796, 0.13894457190562318, 0.14544651318505805,    0.15225271421225728, 0.1593774128878923, 0.1668355133782749,   0.1746426172934012, 0.18281505632397949, 0.191369926405716,   0.20032512348232367, 0.2096993809420704, 0.21951230880617825,   0.2297844347510507, 0.24053724705014706, 0.2517932395253284,    0.2635759586017121, 0.27591005256447015, 0.28882132312060604,   0.3023367793735816, 0.31648469432369364, 0.3312946640124,    0.3467976694343168, 0.3630261413463998, 0.3800140281098884, 0.3977968667069247, 0.4164118570804129, 0.4358979399526274, 0.4562958782853583, 0.4776483425520067, 0.5])
    import scipy.interpolate as interp
    no_q = len(q_points)*10
    q_interpolated = interp.interp1d(np.linspace(0, 1, len(q_points)), q_points, kind="cubic")
    q_points_new = q_interpolated(np.linspace(0.6667, 0.6668, no_q))

    L = 1000
    R = 1000

    # sld = 4
    # solvent_sld = 1
    # background = 0.00000001

    print("Computing I(q) for 2 parameter model")
    vol = np.pi* R**2 * L
    for R in [1000000]:
        L = R
        Iq_2param = computeIq_2param(q_points_new, L, R, tuned_quad_params2, kronrod_points_dict)# * (sld-solvent_sld) + background
        print(Iq_2param)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.plot(q_points_new, Iq_2param, label="2 parameter")
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Cylinder Model -> I(q) vs q, {R=}Å, {L=}Å")
    # Integrate[F[q, alpha]^2*Sin[alpha]]
        plt.savefig(f"cylinder_model_{R}_{L}.png")
    # plt.close()
    # tuned_quad_params1 = tune("params1", integrand_1param, a, b, reg_params1, 1e-5, n_kronrod=kronrod_space)
    # tuned_quad_params6 = tune("params6", integrand_6param, a, b, reg_params6, 1e-5, n_kronrod=kronrod_space)
    # tuned_quad_params4 = tune("params4", integrand_4param, a, b, reg_params4, 1e-5, n_kronrod=kronrod_space)

    # print("Computing I(q) for 1 parameter model")
    # Iq_1param = computeIq_1param(q_points_new, L, R, tuned_quad_params1)

    # print("Computing I(q) for 4 parameter model")
    # Iq_4param = computeIq_4param(q_points_new, L, R, tuned_quad_params4)

    # print("Computing I(q) for 6 parameter model")
    # Iq_6param = computeIq_6param(q_points_new, L, R, tuned_quad_params6)

    # plt.figure(figsize=(10, 10))
    # plt.plot(q_points_new, Iq_1param, label="1 parameter")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("I(q) vs q, 1 parameter")

    # plt.figure(figsize=(10, 10))
    # plt.plot(q_points_new, Iq_4param, label="4 parameter")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("I(q) vs q, 4 parameters")

    # plt.figure(figsize=(10, 10))
    # plt.plot(q_points_new, Iq_6param, label="6 parameter")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("I(q) vs q, 6 parameters")
    # plt.legend()

    plt.show()