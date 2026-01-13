## Accept Reject

$$f(x) = \frac{d}{dx}F(x)$$
$$f(x) <= M * g(x)$$
$$M >= \frac{f(x)}{g(x)}$$

$M$ Needs to be the lowest value higher than the maximum of $\frac{f(x)}{g(x)}$

$$g(x) = \frac{1}{b-a}$$

Now draw u and x from the distribution and check

$$u <= \frac{f(x)}{M * g(x)}$$

## Hull-Dobell M, a, b

$$M = 2^{31}$$
$$a = 1103515245$$
$$b = 12345$$

Calculate:
$$D(x) = (ax + b) \mod M$$

## Hoeffding confidence interval

$$
\epsilon = (b-a) * \sqrt{\frac{\ln\left(\frac{2}{\alpha}\right)}{2n}}
$$

Where $[a, b]$ is from the interval that the random variables $X_i$ are chosen, and $\alpha$ is $1 - \text{confidence interval}$. If the interval is $[0, 1]$ then we get:

$$
\epsilon = \frac{1}{\sqrt{n}} * \sqrt{0.5 * \ln\left(\frac{2}{\alpha}\right)}
$$

Generates the tuple:
$$[\overline{X}_n - \epsilon, \overline{X}_n  + \epsilon]$$

## Accuracy, Precision and Recall

True Positive (TP)  
False Positive (FP)  
True Negative (TN)
False Negative (FN)

### Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP } + \text{ FP}}$$
$$n = \text{TP } + \text{ FP}$$

### Recall

$$\text{Recall} = \frac{\text{TP}}{\text{TP } + \text{ FN}}$$
$$n = \text{TP } + \text{ FN}$$

### Accuracy

$$\text{Accuracy} = \frac{\text{TP } + \text( TN)}{\text{TP } + \text{ TN} + \text{ FP} + \text{ FN}}$$
$$n = \text{TP } + \text{ TN} + \text{ FP} + \text{ FN}$$

## Splits

```py
# 40%, 20%, 40%
split1 = int(0.4 * df.shape[0])
split2 = int(0.6 * df.shape[0])

problem2_X_train = problem2_X[:split1]
problem2_X_calib = problem2_X[split1:split2]
problem2_X_test = problem2_X[split2:]
```

# Markov Chains

## General Notes

A Chain is Irreducible if you can get from any state to any other state.  
A Chain is Aperiodic if the GCD of all cycles that start and end a path is equal to 1, if higher, Periodic.

Quick check:  
If Chain is Irreducible and there is a self-loop somewhere: Aperiodic.  
If Chain is Reducible but there is a self-loop inside the "trap": Aperiodic.

## Irreducible

```py
def is_irreducible(P):
  n = len(P)
  I = np.eye(n)
  return np.all(np.linalg.matrix_power(I + P, n-1) > 0)
```

## Stationary distribution

```py
def find_stationary(P):
  eigenvalues, eigenvectors = np.linalg.eig(P.T)
  idx = np.argmin(np.abs(eigenvalues - 1.0))
  steady_state = np.real(eigenvectors[:, idx])
  return steady_state / steady_state.sum()
```

## Reversible

```py
def is_reversible(P, stat_dist):
  for i in range(len(P)):
    for j in range(len(P)):
      if not np.isclose(P[i][j] * stat*dist[i], P[j][i] \* stat_dist[j]):
        return False
  return True
```

## State after n iterations

```py
state = np.array([1.0, 0.0, 0.0]) # Start in A
for _ in range(n):
  state = np.dot(state, transition_matrix)
state[i] # Probability of being in state i after n iterations
```

## First time state after n iterations

```py
state = np.array([1.0, 0.0, 0.0]) # Start in A
for _ in range(n):
  state[i] = 0 # Zero the state we want to end up in
  state = np.dot(state, transition_matrix)
state[i] # Probability of being in state i after n iterations
```

## Expected number of steps

"What is the expected number of steps until the first time one enters B having started in A"

Coefficients are derived from:

$$E_A = 1 + P(A \rightarrow A)E_A + P(A \rightarrow B)0 + P(A \rightarrow C)E_C$$

$$E_C = 1 + P(C \rightarrow A)E_A + P(C \rightarrow B)0 + P(C \rightarrow C)E_C$$

```py
coefficients = [
  [a, b],
  [c, d]
]
constants = [1, 1]
solution = np.linalg.solve(coefficients, constants)
solution[0] # We are interested in the solution for A
```

## Period

```py
P_power = P.copy()
for n in range(1, state**2 + 1):
  if P_power[state, state] > 1e-10:
    return_time.append(n)
  P_power = np.dot(P_power, P)
period = reduce(gcd, return_time)
```

# Matplotlib

```py
plt.whateverplot(x, y, alpha=, label="")

plt.xlabel("X label")
plt.ylabel("Y label")
plt.title("Title")
plt.grid(True)

plt.plot([x1, y1], [x2, y2], color="", linestyle="--")
plt.show()
```
