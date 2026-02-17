class Robot:
    """
    Simulator-agnostic robot interface.
    """

    def step(self):
        raise NotImplementedError

    def get_time(self):
        raise NotImplementedError

    def get_state_vector(self):
        raise NotImplementedError

    def get_foot_positions_world(self):
        raise NotImplementedError

    def get_foot_jacobian(self, foot_index):
        raise NotImplementedError

    def set_torques(self, tau):
        raise NotImplementedError
