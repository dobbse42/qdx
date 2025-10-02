import numpy as np
import stim
import itertools 
from more_itertools import distinct_permutations

import scipy.special as ss
import jax
import jax.numpy as jnp
import chex
from itertools import combinations
from qdx.simulators import TableauSimulator

import jax.lax

class Utils():
    
    def __init__(self,
                 n,
                 k,
                 gates,
                 softness,
                ):
        
        self.n_qubits_physical = n # Number of physical qubits
        self.n_qubits_logical = k # Number of logical qubits
        
        # Initialize the tableau simulator with the number of physical qubits and apply gates
        tableau_simulator = TableauSimulator(self.n_qubits_physical)
        for g in gates:
            # Dynamically execute the gate commands on the tableau simulator
            eval("tableau_simulator%s" % g)
            
        # Extract the current tableau state after applying gates   
        self.tableau = tableau_simulator.current_tableau[0]
        
        # Generate the stabilizer group structure for stabilizer calculations
        self.generate_S_structure(softness) # This generates self.S_struct

        self.S = self.generate_S(self.tableau[self.n_qubits_physical + self.n_qubits_logical:])
        
        # Symplectic metric Omega
        self.Omega = jnp.kron(jnp.array([[0,1],[1,0]], dtype=jnp.uint8), jnp.eye(self.n_qubits_physical, dtype=jnp.uint8))
        
        
    def error_operators(self, eval_weight) -> chex.Array:
        # Generate error operators based on the evaluation weight.
        
        error_list = [1,2,3] # Corresponds to X,Y,Z
        
        # Generate all combinations of errors with repetition based on eval_weight
        prod = list(list(tup) for tup in itertools.combinations_with_replacement(error_list, eval_weight))
        
        # Pad the error list to match the number of physical qubits
        prod = [pr + [0] * (self.n_qubits_physical - eval_weight) for pr in prod]
        
        # Generate distinct permutations of each product to form the error structure
        result = list(list(distinct_permutations(pr, self.n_qubits_physical)) for pr in prod)
        error_structure = list(itertools.chain(*result))
        
        # Convert error structure to a JAX array for efficient computation
        E_mu = jnp.array([jnp.array(stim.PauliString(p).to_numpy()).flatten() for p in error_structure], dtype=jnp.uint8)
        
        return E_mu
    
    def generate_S_structure(self, softness):
        # Generate the structure of the stabilizer group (S) based on the softness parameter.
        
        num = self.n_qubits_physical - self.n_qubits_logical
        
        # Calculate the number of stabilizer elements
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))

        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Generate combinations of indices to place ones in the S structure
            comb = list(combinations(range(num),m)) # ???
            indices = np.array(comb)

            # Fill the S structure with ones at the specified indices
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1
            
        # Ensure there are no rows with all zeroes in the S structure
        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes" # comment this out for speed
        
        # Convert the S structure to a JAX array for efficient computation
        self.S_struct = jnp.array(S_struct, dtype=jnp.uint8)

        return
    
    def generate_S(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        print(f"self.S_struct shape: {jnp.shape(self.S_struct)}")
        print(f"tableau shape: {jnp.shape(tableau)}")
        return (self.S_struct @ tableau) % 2
    
    def remove_rows(self, arr, row_indices):
        mask = jnp.ones(arr.shape, dtype=int)
        mask = mask.at[jnp.array(row_indices)].set(0)
        mask = mask.astype(bool)

        a = jnp.array(row_indices).shape[0]
        b,c = arr.shape
        # return arr * mask
        return arr[mask].reshape(b-a,c)
    
    def check_KL(self, E_mu):
        # Check the Knill-Laflamme conditions for error correction.
        num_KL = 0
        print(f"shape of E_mu: {jnp.shape(E_mu)}, shape of self.S: {jnp.shape(self.S)}")
        # half_index = int(len(E_mu)/2)
        quarter_index = int(len(E_mu)/4)
        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        inS_1 = jax.vmap(jnp.logical_xor, in_axes=(None,0))(self.S, E_mu[:quarter_index])
        inS_1 = jnp.prod(jnp.logical_not(inS_1), axis=-1)
        num_KL += len(E_mu[:quarter_index]) - jnp.sum(jnp.any(((E_mu[:quarter_index] @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, axis=1), axis=0) - jnp.sum(inS_1)
        inS_2 = jax.vmap(jnp.logical_xor, in_axes=(None,0))(self.S, E_mu[quarter_index:2*quarter_index])
        inS_2 = jnp.prod(jnp.logical_not(inS_2), axis=-1)
        num_KL += len(E_mu[quarter_index:2*quarter_index]) - jnp.sum(jnp.any(((E_mu[quarter_index:2*quarter_index] @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, axis=1), axis=0) - jnp.sum(inS_2)
on $[[14, 2, 3]]$:        inS_3 = jax.vmap(jnp.logical_xor, in_axes=(None,0))(self.S, E_mu[2*quarter_index:3*quarter_index])
        inS_3 = jnp.prod(jnp.logical_not(inS_3), axis=-1)
        num_KL += len(E_mu[2*quarter_index:3*quarter_index]) - jnp.sum(jnp.any(((E_mu[2*quarter_index:3*quarter_index] @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, axis=1), axis=0) - jnp.sum(inS_3)
        inS_4 = jax.vmap(jnp.logical_xor, in_axes=(None,0))(self.S, E_mu[3*quarter_index:])
        inS_4 = jnp.prod(jnp.logical_not(inS_4), axis=-1)
        num_KL += len(E_mu[3*quarter_index:]) - jnp.sum(jnp.any(((E_mu[3*quarter_index:] @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, axis=1), axis=0) - jnp.sum(inS_4)

        # Calculate the number of Knill-Laflamme conditions that are not satisfied
        # num_KL = len(E_mu) - jnp.sum(jnp.any(((E_mu @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, axis=1), axis=0) - jnp.sum(inS)
        return num_KL # np.min(KLs), np.argmin(KLs)
    
    def check_KL_cZ(self, E_mu, cZ):
        # Check the Knill-Laflamme conditions specifically considering a biased error channel with bias parameter cZ
        
        # Separate X,Y,Z part of errors for effective weight computation
        H_X = E_mu[:,:self.n_qubits_physical]
        H_Z = E_mu[:,self.n_qubits_physical:]
        H_Y = H_X * H_Z
        
        # Calculate the effective weights
        effective_weights = jnp.sum(H_X - H_Y, axis=1) + jnp.sum(H_Y, axis=1) + cZ*jnp.sum(H_Z - H_Y, axis=1)
        
        # Extract the stabilizer generators
        tableau = self.tableau[self.n_qubits_physical + self.n_qubits_logical:]
        
        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        inS = jax.vmap(jnp.logical_xor, in_axes=(None,0))(self.S, E_mu)
        inS = jnp.prod(jnp.logical_not(inS), axis=-1)
        
        # Check which error operators don't satisfy the Knill-Laflamme conditions
        KL1 = effective_weights * jnp.logical_not(jnp.any((E_mu @ self.Omega @ tableau.T)%2, axis=1))
        KL2 = effective_weights * jnp.sum(inS, axis=1)
        KLs = KL1 - KL2
        
        # Identify non-zero Knill-Laflamme values (the ones that don't satisfy KL)
        non_zero_ids = jnp.nonzero(KLs)[0]
        
        # Determine the smallest undetected effective weight
        if len(KLs[non_zero_ids]) > 0:
            smallest_undetected_effective_weight = jnp.min(KLs[non_zero_ids])
            
        # If all errors are detected, return a large number (it will be discarded anyway)
        else:
            smallest_undetected_effective_weight = 100
        
        return round(float(smallest_undetected_effective_weight),2)
