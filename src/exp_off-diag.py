import autograd.numpy as anp
import pandas as pd
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

from tqdm import tqdm

from optimizers import ConjugateGradient, MemorylessQuasiNewton


anp.random.seed(99999)

beta_rules = [
    'HagerZhang',
    "DaiYuan",
]

formulas = [
    'BFGS',
    'preconvex'
]

reqularization_methods = [
    'LiFukushima',
    'Powell',
]

xi_list = [1., 0.8, 0.1]

if __name__ == '__main__':
    algorithms = []
    for formula in formulas:
        for regularization in reqularization_methods:
            for xi in xi_list:
                algorithms.append(f'{formula}+{regularization}+(xi={xi})')
    for beta_rule in beta_rules:
        algorithms.append(f'CG({beta_rule})')
    
    iterations_list = []
    elapsed_list = []

    n = 10
    p = 5
    
    for _ in tqdm(range(100)):
        manifold = pymanopt.manifolds.Oblique(n, p)

        matrices = []
        for k in range(5):
            B = anp.random.randn(n, n)
            C = (B + B.T) / 2
            matrices.append(C)

        @pymanopt.function.autograd(manifold)
        def cost(X):
            _sum = 0.
            for matrix in matrices:
                Y = X.T @ matrix @ X
                _sum += anp.linalg.norm(Y - anp.diag(anp.diag(Y))) ** 2
            return _sum

        iterations_row = []
        elapsed_row = []

        for formula in formulas:
            for regularization in reqularization_methods:
                for xi in xi_list:
                    problem = pymanopt.Problem(manifold, cost)
                    optimizer = MemorylessQuasiNewton(formula=formula, regularization_method=regularization, xi=xi, verbosity=0)
                    results, elapsed_time = optimizer.run(problem)

                    iterations_row.append(results.iterations)
                    elapsed_row.append(elapsed_time)

        for beta_rule in beta_rules:
            problem = pymanopt.Problem(manifold, cost)
            optimizer = ConjugateGradient(beta_rule=beta_rule, verbosity=0)

            results, elapsed_time = optimizer.run(problem)

            iterations_row.append(results.iterations)
            elapsed_row.append(elapsed_time)
        
        iterations_list.append(iterations_row)
        elapsed_list.append(elapsed_row)

    iterations_df = pd.DataFrame(iterations_list, columns=algorithms)
    elapsed_df = pd.DataFrame(elapsed_list, columns=algorithms)

    iterations_df.to_csv('results/off-diag/iterations.csv', index=None)
    elapsed_df.to_csv('results/off-diag/elapsed.csv', index=None)
