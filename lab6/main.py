import os
import networkx as nx
import ndlib.models.opinions as op
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend


def visualize(model, trends, sub_dir):
    viz_img_name = model.get_name()
    for param, value in model.params['model'].items():
        viz_img_name = f'{viz_img_name};{param}({str(value).replace(".", ",")})'
    if not os.path.exists('./viz/'):
        os.mkdir('./viz/')
    if not os.path.exists(f'./viz/{sub_dir}'):
        os.mkdir(f'./viz/{sub_dir}')
    DiffusionTrend(model, trends).plot(filename=f'./viz/{sub_dir}/{viz_img_name}.png')


def get_si_params():
    print('\n\tInput SI model parameters')
    beta = float(input('beta(infection probability)[0, 1]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)  # 0.01
    cfg.add_model_parameter("fraction_infected", fraction_infected)  # 0.05
    return cfg


def get_sis_params():
    print('\n\tInput SIS model parameters')
    beta = float(input('beta(infection probability)[0, 1]:'))
    lambda_param = float(input('lambda(recovery probability)[0, 1]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)  # 0.01
    cfg.add_model_parameter('lambda', lambda_param)  # 0.005
    cfg.add_model_parameter("fraction_infected", fraction_infected)  # 0.05
    return cfg


def get_seis_params():
    print('\n\tInput SEIS model parameters')
    beta = float(input('beta(infection probability)[0, 1]: '))
    lambda_param = float(input('lambda(recovery probability)[0, 1]: '))
    alpha = float(input('alpha(latent period)[0, 1]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)  # 0.01
    cfg.add_model_parameter('lambda', lambda_param)  # 0.005
    cfg.add_model_parameter('alpha', alpha)  # 0.05
    cfg.add_model_parameter("fraction_infected", fraction_infected)  # 0.05
    return cfg


def get_seir_params():
    print('\n\tInput SEIR model parameters')
    beta = float(input('beta(infection probability)[0, 1]: '))
    gamma = float(input('gamma(removal probability)[0, 1]: '))
    alpha = float(input('alpha(latent period)[0, 1]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)  # 0.01
    cfg.add_model_parameter('gamma', gamma)  # 0.005
    cfg.add_model_parameter('alpha', alpha)  # 0.05
    cfg.add_model_parameter("fraction_infected", fraction_infected)  # 0.05
    return cfg


def get_voter_params():
    print('\n\tInput Voter model parameters')
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('fraction_infected', fraction_infected)  # 0.1
    return cfg


def get_qvoter_params():
    print('\n\tInput QVoter model parameters')
    q = int(input('q(number of neighbours)[0, V(G)]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter("q", q)  # 5
    cfg.add_model_parameter('fraction_infected', fraction_infected)  # 0.1
    return cfg


def get_majority_rules_params():
    print('\n\tInput Majority Rules model parameters')
    q = int(input('q(number of neighbours)[0, V(G)]: '))
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter("q", q)  # 5
    cfg.add_model_parameter('fraction_infected', fraction_infected)  # 0.1
    return cfg


def get_sznajd_params():
    print('\n\tInput SZNAJD model parameters')
    fraction_infected = float(input('fraction_infected(initial)[0, 1]: '))

    cfg = mc.Configuration()
    cfg.add_model_parameter('fraction_infected', fraction_infected)  # 0.1
    return cfg


if __name__ == '__main__':
    g = nx.erdos_renyi_graph(1000, 0.1)
    num_iterations = int(input('Number of iterations for all models: '))  # 200

    ###############################################################

    SIModel = ep.SIModel(g.copy())
    SIModel.set_initial_status(get_si_params())
    SI_iterations = SIModel.iteration_bunch(num_iterations)
    SI_trends = SIModel.build_trends(SI_iterations)
    visualize(SIModel, SI_trends, sub_dir='epidemics')

    SISModel = ep.SISModel(g.copy())
    SISModel.set_initial_status(get_sis_params())
    SIS_iterations = SISModel.iteration_bunch(num_iterations)
    SIS_trends = SISModel.build_trends(SIS_iterations)
    visualize(SISModel, SIS_trends, sub_dir='epidemics')

    SEISModel = ep.SEISModel(g.copy())
    SEISModel.set_initial_status(get_seis_params())
    SEIS_iterations = SEISModel.iteration_bunch(num_iterations)
    SEIS_trends = SEISModel.build_trends(SEIS_iterations)
    visualize(SEISModel, SEIS_trends, sub_dir='epidemics')

    SEIRModel = ep.SEIRModel(g.copy())
    SEIRModel.set_initial_status(get_seir_params())
    SEIR_iterations = SEIRModel.iteration_bunch(num_iterations)
    SEIR_trends = SEIRModel.build_trends(SEIR_iterations)
    visualize(SEIRModel, SEIR_trends, sub_dir='epidemics')

    ###############################################################

    voter_model = op.VoterModel(g.copy())
    voter_model.set_initial_status(get_voter_params())
    voter_iterations = voter_model.iteration_bunch(num_iterations)
    voter_trends = voter_model.build_trends(voter_iterations)
    visualize(voter_model, voter_trends, sub_dir='opinions')

    QVoter_model = op.QVoterModel(g.copy())
    QVoter_model.set_initial_status(get_qvoter_params())
    QVoter_iterations = QVoter_model.iteration_bunch(num_iterations)
    QVoter_trends = QVoter_model.build_trends(QVoter_iterations)
    visualize(QVoter_model, QVoter_trends, sub_dir='opinions')

    majority_rule_model = op.MajorityRuleModel(g.copy())
    majority_rule_model.set_initial_status(get_majority_rules_params())
    majority_rule_iterations = majority_rule_model.iteration_bunch(num_iterations)
    majority_rule_trends = majority_rule_model.build_trends(majority_rule_iterations)
    visualize(majority_rule_model, majority_rule_trends, sub_dir='opinions')

    sznajd_model = op.SznajdModel(g.copy())
    sznajd_model.set_initial_status(get_sznajd_params())
    sznajd_iterations = sznajd_model.iteration_bunch(num_iterations)
    sznajd_trends = sznajd_model.build_trends(sznajd_iterations)
    visualize(sznajd_model, sznajd_trends, sub_dir='opinions')
