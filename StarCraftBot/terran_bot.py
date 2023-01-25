import random
import numpy as np
import pandas as pd
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop


class QLearningTable:
    def __init__(self, possible_actions, learning_rate=0.01, reward_decay=0.9, filename=None):
        self.actions = possible_actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        if filename is None:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        else:
            self.q_table = pd.read_csv(filename, index_col=0)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, prev_state, action_performed, reward, new_state):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[prev_state, action_performed]
        if new_state != 'terminal':
            q_target = reward + self.reward_decay * self.q_table.loc[new_state, :].max()
        else:
            q_target = reward
        self.q_table.loc[prev_state, action_performed] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns,
                                                         name=state))

    def save(self, filename):
        self.q_table.to_csv(filename)


class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.base_top_left = None
        self.attack_xy = None
        self.actions = ("do_nothing", "harvest_minerals", "build_supply_depot", "build_barracks", "train_marine",
                        "attack")

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, the_units, xy):
        units_xy = [(unit.x, unit.y) for unit in the_units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
            # enemy_base = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)[0]
            # self.attack_xy = (enemy_base.x, enemy_base.y)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and len(scvs) > 0:
            supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
            distances = self.get_distances(scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 0
                and obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100 and free_supply > 0:
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            # attack_xy = (38, 44) if self.base_top_left else (19, 23)
            if self.attack_xy is None:
                enemy_base = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)[0]
                self.attack_xy = (enemy_base.x, enemy_base.y)

            distances = self.get_distances(marines, self.attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag,
                                                   (self.attack_xy[0] + x_offset, self.attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


class SmartAgent(Agent):
    def __init__(self, filename=None):
        super(SmartAgent, self).__init__()
        self.q_table = QLearningTable(self.actions, filename=filename)
        self.clear()

    def reset(self):
        super(SmartAgent, self).reset()
        self.clear()

    def clear(self):
        self.base_top_left = None
        self.attack_xy = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.q_table.choose_action(state)
        if self.previous_action is not None:
            self.q_table.learn(self.previous_state, self.previous_action, obs.reward,
                               'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

    def save(self, filename):
        self.q_table.save(filename)


def main(unused_argv):
    agent1 = SmartAgent(filename="terran_table.csv")
    agent2 = RandomAgent()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=48,
                disable_fog=True,
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=5)  # normally 1000 episodes
            agent1.save("terran_table.csv")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
