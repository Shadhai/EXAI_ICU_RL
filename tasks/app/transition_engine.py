from app.state_manager import StateManager

class TransitionEngine:
    def __init__(self, state_manager: StateManager, environment):
        self.state = state_manager
        self.env = environment

    def apply_action(self, action: str) -> bool:
        action_valid = True

        if action == "administer_drug":
            self.state.update_vital("BP", +25)
            self.state.update_lab("lactate", -2.5)
            self.state.update_vital("SpO2", +5)
            self.state.update_vital("GCS", +1.5)

        elif action == "adjust_ventilator":
            if self.state.resources["ventilators"] > 0:
                self.state.update_vital("SpO2", +12)
                self.state.update_vital("RR", -6)
                self.state.update_resource("ventilators", -1)
            else:
                action_valid = False

        elif action == "request_lab":
            pass

        elif action == "escalate_care":
            if self.state.resources["icu_beds"] > 0:
                # Force vitals to perfect
                self.state.vitals["BP"] = 120.0
                self.state.vitals["SpO2"] = 96.0
                self.state.vitals["RR"] = 16.0
                self.state.vitals["GCS"] = 15.0
                self.state.labs["lactate"] = 1.5
                self.state.labs["pH"] = 7.40
                self.state.update_resource("icu_beds", -1)
                # Force survival to 95%
                self.state.survival_probability = 0.95
            else:
                action_valid = False
        else:
            action_valid = False

        # Deterioration (very slow)
        if self.state.alive and action != "escalate_care":
            self.state.update_vital("SpO2", -0.05)
            self.state.update_vital("BP", -0.1)
            self.state.update_lab("lactate", +0.01)

        # Only recalc survival if not forced by escalate_care
        if action != "escalate_care":
            self.state._update_survival_from_labs()

        # Death condition
        if (self.state.survival_probability < 0.05 or
            self.state.vitals["SpO2"] < 65 or
            self.state.vitals["BP"] < 45 or
            self.state.vitals["GCS"] <= 3):
            self.state.alive = False
            self.state.survival_probability = 0.0

        self.state.step += 1
        return action_valid