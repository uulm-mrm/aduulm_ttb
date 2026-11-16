import matplotlib.pyplot as plt
import pymc as pm

import pymc as pm
import arviz as az
# Taking draws from a normal distribution
seed = 42

exp_p_dist = pm.Dirichlet.dist(2*(1,))
exp_p = pm.draw(exp_p_dist, random_seed=seed)
print("Experiment Distribution:", exp_p)

trust_p_agents = [1, 0.99, 0.6]

true_if_trust = 0.99


N = 2000
exp_p_dist = pm.Dirichlet.dist(2*(1,), shape=(N, 2))
exp_p = pm.draw(exp_p_dist, random_seed=seed)
print(exp_p)
E = pm.draw(pm.Categorical.dist(exp_p), random_seed=seed)

def response_p(E, T, true_if_trust):
    return (1-E)*T*(1-true_if_trust) + (1-E)*(1-T)*true_if_trust + E*T*true_if_trust + E*(1-T)*(1-true_if_trust)

Rs = []
for agent_trust_p in trust_p_agents:
    T = pm.draw(pm.Bernoulli.dist(agent_trust_p, shape=N))
    out_agent = []
    agent_response_p = response_p(E, T, true_if_trust)
    print("E        ", E)
    print("Trust:   ", T)
    R = pm.draw(pm.Bernoulli.dist(agent_response_p))
    Rs.append(R)
    print("Response:", R)

# Define generative model
with pm.Model() as model:
    E_p = pm.Dirichlet("E_p", (1, 1))
    E = pm.Categorical("E", E_p, observed=E)
    for i in range(len(trust_p_agents)):
        a1_trust_p_prior = pm.Beta(f"agent{i}_p_prior", 2, 1)
        T = pm.Bernoulli(f"agent{i}_trust", a1_trust_p_prior)
        a1_response_p = pm.Deterministic(f"response{i}", (1-E)*T*(1-true_if_trust) + (1-E)*(1-T)*true_if_trust + E*T*true_if_trust + E*(1-T)*(1-true_if_trust))
        a1_obs = pm.Bernoulli(f"agent{i}_response", a1_response_p, observed=Rs[i])

pm.model_to_graphviz(model, save='graph.png')

with model:
    idata = pm.sample(5000, tune=2000, random_seed=42)
    az.plot_energy(idata)
    az.plot_trace(idata, combined=True)
    az.plot_trace(idata, var_names=["agent0_trust", "agent1_trust", "agent2_trust"])
    plt.show(block=True)