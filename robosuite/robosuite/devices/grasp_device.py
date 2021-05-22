import abc  # for abstract base class definitions


class GraspDevice(metaclass=abc.ABCMeta):
    """
    Base class for all robot grasp devices.
    Defines basic interface for all grasp devices to adhere to.
    """

    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can 
        start receiving commands. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_grasp_action(self, grasp_state, hand_model):
        raise NotImplementedError

    @abc.abstractmethod
    def get_grasp_state(self):
        """Returns the current grasp state of the device, a list of grasp values per device joint"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_haptic_feedback(self, feedback_dict, force_scale):
        """Sends haptic feedback to the device"""
        raise NotImplementedError
