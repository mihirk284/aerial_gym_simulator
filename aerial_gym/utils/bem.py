
import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BEMParams:
    """Parameters required for BEM theory calculations"""
    rho: float    # Air density [kg/m^3]
    R: float      # Rotor radius [m]
    b: int        # Number of blades
    c: float      # Chord length [m]
    cd0: float    # Zero-lift drag coefficient
    cl0: float    # Lift coefficient slope
    theta0: float # Blade root pitch angle [rad]
    theta1: float # Blade twist angle [rad]
    k_beta: float # Hinge spring stiffness [N⋅m/rad]
    e: float      # Hinge offset [m]
    I_b: float    # Blade moment of inertia about flapping hinge [kg⋅m²]
    m_b: float    # Mass of single blade [kg]


def bem_algorithm(params: BEMParams, 
                 omega: float,   # Propeller angular velocity [rad/s]
                 v_hor: float,   # Horizontal velocity [m/s]
                 v_ver: float,   # Vertical velocity [m/s]
                 p: float,       # Roll rate [rad/s]
                 q: float,       # Pitch rate [rad/s]
                 clockwise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Blade Element Momentum (BEM) theory algorithm as described in the paper.
    
    The algorithm follows these steps:
    1. Initialize with zero flapping angles
    2. Find induced velocity using momentum theory and blade element theory
    3. Calculate flapping angles using moment equilibrium
    4. Recalculate forces with final angles
    5. Return forces and torques in propeller frame
    """
    
    def momentum_thrust(v_i: float) -> float:
        """
        Calculates thrust using momentum theory (Equation 5).
        T = 2v_i⋅ρ⋅A⋅sqrt(v_hor² + (v_ver - v_i)²)
        """
        return 2 * v_i * params.rho * (np.pi * params.R**2) * \
               np.sqrt(v_hor**2 + (v_ver - v_i)**2)

    def calculate_moments(r: float, psi: float, a0: float, a1: float, b1: float, v_i: float) -> tuple:
        """
        Calculates all moments acting on the blade for flapping equation (Equation 16).
        Returns tuple of (M_w, M_gyro, M_inertial, M_cf, M_aero, M_spring)
        """
        # Current flapping angle and its derivatives
        beta = a0 + a1 * np.cos(psi) + b1 * np.sin(psi)
        beta_dot = -a1 * omega * np.sin(psi) + b1 * omega * np.cos(psi)
        beta_ddot = -a1 * omega**2 * np.cos(psi) - b1 * omega**2 * np.sin(psi)
        
        # Weight moment - gravity effect on blade
        M_w = params.m_b * 9.81 * params.e * np.cos(psi)
        
        # Gyroscopic moment due to aircraft angular rates
        M_gyro = params.I_b * omega * (p * np.sin(psi) - q * np.cos(psi))
        
        # Inertial moment from flapping motion
        M_inertial = params.I_b * beta_ddot
        
        # Centrifugal force moment
        M_cf = -params.m_b * omega**2 * params.e * params.R * np.sin(beta)
        
        # Calculate aerodynamic moment
        # Equations 6-7: Velocities at blade element
        U_T = omega * r + v_hor * np.sin(psi)
        U_P = (v_ver - v_i - 
               r * omega * (a1 * np.sin(psi) + b1 * np.cos(psi)) +
               v_ver * (a0 - a1 * np.cos(psi) - b1 * np.sin(psi)) * np.cos(psi))
        
        # Equations 8-9: Flow angle and angle of attack
        phi = np.arctan2(U_P, U_T)
        alpha = params.theta0 + (r/params.R) * params.theta1 + phi
        
        # Equation 12: Lift and drag coefficients
        cl = params.cl0 * np.sin(alpha) * np.cos(alpha)
        cd = params.cd0 * np.sin(alpha)**2
        
        # Equations 10-11: Differential lift and drag
        U_sq = U_T**2 + U_P**2
        dL = params.c * cl * U_sq
        dD = params.c * cd * U_sq
        
        M_aero = r * (dL * np.cos(phi) + dD * np.sin(phi))
        
        # Equation 17: Spring moment
        M_spring = params.k_beta * beta
        
        return M_w, M_gyro, M_inertial, M_cf, M_aero, M_spring

    def blade_element_thrust(v_i: float, a0: float, a1: float, b1: float) -> Tuple[float, float, float]:
        """
        Calculates thrust, horizontal force, and torque using blade element theory
        (Equations 13-15 with numerical integration)
        """
        dr = params.R / 12.5  # Radial discretization
        dpsi = 2 * np.pi / 6  # Azimuthal discretization
        
        T, H, Q = 0, 0, 0
        
        for r in np.arange(0, params.R, dr):
            for psi in np.arange(0, 2*np.pi, dpsi):
                # Calculate velocities, angles, and forces as in calculate_moments()
                U_T = omega * r + v_hor * np.sin(psi)
                U_P = (v_ver - v_i - 
                      r * omega * (a1 * np.sin(psi) + b1 * np.cos(psi)) +
                      v_ver * (a0 - a1 * np.cos(psi) - b1 * np.sin(psi)) * np.cos(psi))
                
                phi = np.arctan2(U_P, U_T)
                alpha = params.theta0 + (r/params.R) * params.theta1 + phi
                
                cl = params.cl0 * np.sin(alpha) * np.cos(alpha)
                cd = params.cd0 * np.sin(alpha)**2
                
                U_sq = U_T**2 + U_P**2
                dL = params.c * cl * U_sq * dr * dpsi
                dD = params.c * cd * U_sq * dr * dpsi
                
                # Integrate forces and torque
                T += dL * np.cos(phi) + dD * np.sin(phi)
                H += (-dL * np.sin(phi) + dD * np.cos(phi)) * np.sin(psi)
                Q += (-dL * np.sin(phi) + dD * np.cos(phi)) * r
        
        # Scale by number of blades and air density
        return (params.b * params.rho / (4 * np.pi)) * np.array([T, H, Q])
    
    def flapping_equilibrium(x: np.ndarray) -> np.ndarray:
        """
        Solves for flapping angles by enforcing moment equilibrium (Equation 16)
        Uses Fourier decomposition to solve for a0, a1, and b1
        """
        a0, a1, b1 = x
        
        n_psi = 72  # Number of azimuthal points
        psi_points = np.linspace(0, 2*np.pi, n_psi)
        residuals = np.zeros(3)
        
        for psi in psi_points:
            # Sum all moments at current azimuth
            moments = calculate_moments(params.R/2, psi, a0, a1, b1, v_i)
            M_total = sum(moments)
            
            # Fourier decomposition
            residuals[0] += M_total            # Constant term (a0)
            residuals[1] += M_total * np.cos(psi)  # Cosine term (a1)
            residuals[2] += M_total * np.sin(psi)  # Sine term (b1)
            
        return residuals / n_psi
    
    # Step 1: Initialize with zero flapping
    a0, a1, b1 = 0, 0, 0
    
    # Step 2: Find induced velocity
    def thrust_residual(v_i):
        """Balance momentum theory thrust with blade element thrust"""
        T, _, _ = blade_element_thrust(v_i, a0, a1, b1)
        return momentum_thrust(v_i) - T
    
    # Initial guess based on hover induced velocity
    v_i_initial = np.sqrt(omega * params.R**2 / 2)
    v_i = fsolve(thrust_residual, v_i_initial)[0]
    
    # # Check for vortex ring state (Equation 18)
    # if 0 < v_ver/v_i < 2:
    #     # Equation 19: Empirical approximation for vortex ring state
    #     v_ratio = v_ver / v_i
    #     v_i_tilde = v_i * (1 + 1.125*v_ratio - 1.372*v_ratio**2 + 
    #                       1.718*v_ratio**3 - 0.655*v_ratio**4)
    #     v_i = max(v_i_tilde, v_i)
    
    # Step 3: Calculate flapping angles
    flap_angles = fsolve(flapping_equilibrium, [0, 0, 0])
    a0, a1, b1 = flap_angles
    
    # Steps 4-5: Final forces and torques
    T, H, Q = blade_element_thrust(v_i, a0, a1, b1)
    
    # Set direction based on propeller rotation
    sign = -1 if clockwise else 1
    
    # Final forces and torques in propeller frame
    force = np.array([
        -(H + np.sin(a1) * T),
        sign * np.sin(b1) * T,
        -T * np.cos(a0)
    ])
    
    torque = np.array([
        sign * params.k_beta * b1,
        params.k_beta * a1,
        -sign * Q
    ])
    
    return force, torque

