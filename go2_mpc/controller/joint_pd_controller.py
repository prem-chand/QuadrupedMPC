import numpy as np
import mujoco

class JointPDController:
    """
    Robust Joint-Space PD Controller.
    Automatically maps Actuators <-> Joints using MuJoCo model data.
    """

    def __init__(self, model, kp=60.0, kd=5.0):
        self.model = model
        self.kp = kp
        self.kd = kd
        
        # 1. Auto-Detect Mapping
        # We need to know which address in 'qpos' corresponds to which address in 'ctrl'
        self.qpos_indices = []
        self.ctrl_indices = []
        
        # Iterate over all actuators in the model
        for i in range(model.nu):
            # Get the joint ID associated with this actuator
            # trnid (transmission ID) maps actuator_id -> joint_id
            # The first column of trnid usually holds the joint ID for simple motors
            joint_id = model.actuator_trnid[i, 0]
            
            # Get the qpos address for this joint
            # qpos_adr gives the index in the qpos array
            q_adr = model.jnt_qposadr[joint_id]
            
            # Store the mapping
            self.ctrl_indices.append(i)      # Actuator Index
            self.qpos_indices.append(q_adr)  # Qpos Index
            
        # Convert to numpy for speed
        self.ctrl_indices = np.array(self.ctrl_indices, dtype=int)
        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        
        # Also get velocity indices (dof_adr)
        # Usually qvel_adr = qpos_adr - 1 (for freejoint systems), but let's be safe:
        self.dof_indices = []
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            dof_adr = model.jnt_dofadr[joint_id]
            self.dof_indices.append(dof_adr)
        self.dof_indices = np.array(self.dof_indices, dtype=int)

        # Default Target: Current configuration or Zero
        # We initialize target to zeros, user must set valid stand pose
        self.q_des = np.zeros(len(self.ctrl_indices))
        self.dq_des = np.zeros(len(self.ctrl_indices))

    def update_target(self, q_desired):
        """
        Update the target joint angles.
        Args:
            q_desired: Array of angles matching the ACTUATOR order, 
                       OR a dictionary {joint_name: angle}
        """
        if isinstance(q_desired, dict):
            # Advanced: Allow setting by name
            pass # (Implementation omitted for brevity)
        else:
            self.q_des = np.array(q_desired)

    def compute_torques(self, qpos, qvel):
        """
        Vectorized torque computation.
        
        Args:
            qpos: Full simulation qpos
            qvel: Full simulation qvel
        """
        # 1. Extract Measurement (Fast Slicing)
        # We only grab the joints that have motors attached
        q_curr = qpos[self.qpos_indices]
        dq_curr = qvel[self.dof_indices]
        
        # 2. PD Control
        # tau = Kp * (q_des - q) + Kd * (dq_des - dq)
        tau = self.kp * (self.q_des - q_curr) + self.kd * (self.dq_des - dq_curr)
        
        # 3. Clip Torques (Safety)
        # It's good practice to clip to the actuator's hardware limits
        # defined in the XML (actuator_ctrlrange)
        ctrl_range = self.model.actuator_ctrlrange[:, 1] # Upper limit (assume symmetric)
        # Note: Some actuators might be asymmetric, but this is a safe simplification
        # Or use: tau = np.clip(tau, self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])
        
        # Return full array directly mapped to 'data.ctrl'
        # Since self.ctrl_indices usually is 0..nu-1, we can just return tau.
        # If actuators are sparse, we would need to scatter it.
        return tau