import numpy as np

class ConvexMPCMatrices:
    def __init__(self, mass, inertia_body):
        self.mass = mass
        # inertia_body is a 3x3 diagonal matrix
        self.I_body = inertia_body 
        self.g = 9.81

    def get_discrete_matrices(self, psi, r_feet, dt):
        """
        Constructs the discrete-time A and B matrices for the QP.
        
        State vector x (13x1): [Theta(3), p(3), omega(3), v(3), g_const(1)]
        Control vector u (12x1): [f1(3), f2(3), f3(3), f4(3)]
        
        Args:
            psi (float): Current Yaw angle (for rotation approx).
            r_feet (np.array): (4, 3) matrix of foot positions relative to CoM.
            dt (float): Prediction time step (e.g., 0.03s).
            
        Returns:
            A_d (13, 13): Discrete state transition matrix.
            B_d (13, 12): Discrete control input matrix.
        """
        
        # --- 1. Rotation Approximation ---
        # We rotate the inertia tensor by Yaw (psi) only.
        c = np.cos(psi)
        s = np.sin(psi)
        R_z = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Approximate World Inertia: R * I_body * R.T
        I_world = R_z @ self.I_body @ R_z.T
        I_inv = np.linalg.inv(I_world)

        # --- 2. Construct Continuous Matrix A_c (13x13) ---
        A_c = np.zeros((13, 13))
        
        # derivatives of orientation (Theta) depend on angular velocity (omega)
        # For small angles, d(Theta)/dt approx = R_z.T * omega
        # (This maps world angular velocity back to body frame Euler rates)
        A_c[0:3, 6:9] = R_z.T 
        
        # derivatives of position (p) depend on linear velocity (v)
        A_c[3:6, 9:12] = np.eye(3)
        
        # --- The Gravity Trick ---
        # We want v_dot = F/m + g.
        # Since the 13th state is fixed at "1", we put "g" in the column corresponding to it.
        # This adds [0, 0, -9.81] * 1 to the velocity update.
        A_c[11, 12] = -self.g # Z-velocity is affected by gravity
        
        # --- 3. Construct Continuous Matrix B_c (13x12) ---
        B_c = np.zeros((13, 12))
        
        for i in range(4): # For each leg (0 to 3)
            r = r_feet[i] # Position of foot i relative to CoM
            
            # Cross product skew-symmetric matrix [r]x
            # r x f = [r]x * f
            r_skew = np.array([
                [0,    -r[2],  r[1]],
                [r[2],  0,    -r[0]],
                [-r[1], r[0],  0]
            ])
            
            # Angular acceleration: I_inv * (r x f)
            # Rows 6:9 are omega_dot
            # Cols i*3 : i*3+3 are forces for leg i
            B_c[6:9, i*3 : i*3+3] = I_inv @ r_skew
            
            # Linear acceleration: f / m
            # Rows 9:12 are v_dot
            B_c[9:12, i*3 : i*3+3] = np.eye(3) / self.mass

        # --- 4. Discretization (Forward Euler) ---
        # A_d = I + A_c * dt
        # B_d = B_c * dt
        
        A_d = np.eye(13) + A_c * dt
        B_d = B_c * dt
        
        return A_d, B_d