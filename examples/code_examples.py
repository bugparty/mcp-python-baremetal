# Example Python code snippets for MCP Python Baremetal

# 1. Basic NumPy operations (auto-import)
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", np.mean(arr))
print("Standard deviation:", np.std(arr))

# 2. PyTorch tensor operations (auto-import)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = torch.dot(x, y)
print("Dot product:", z)

# 3. JAX operations (auto-import)
key = jax.random.PRNGKey(42)
random_data = jax.random.normal(key, (5,))
print("Random JAX array:", random_data)

# 4. SymPy symbolic math (auto-import)
x = sympy.Symbol("x")
expr = x**2 + 2*x + 1
derivative = sympy.diff(expr, x)
print("Expression:", expr)
print("Derivative:", derivative)

# 5. CVXPy optimization (auto-import)
x = cp.Variable()
objective = cp.Minimize(cp.square(x - 2))
constraints = [x >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
print("Optimal value:", x.value)

# 6. Multiple libraries together (all auto-imported)
# Mathematical computation with different libraries
data = np.random.randn(10)
tensor_data = torch.tensor(data)
jax_data = jnp.array(data)

print("NumPy mean:", np.mean(data))
print("PyTorch mean:", torch.mean(tensor_data))
print("JAX mean:", jnp.mean(jax_data))
