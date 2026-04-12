class ActionHandler:
    VALID_ACTIONS = {"administer_drug", "adjust_ventilator", "request_lab", "escalate_care"}

    @staticmethod
    def is_valid(action: str) -> bool:
        return action in ActionHandler.VALID_ACTIONS