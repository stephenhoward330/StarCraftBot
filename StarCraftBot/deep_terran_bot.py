import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('prev_state', 'action', 'state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv1 = nn.Linear(state_size, 32)
        self.conv2 = nn.Linear(32, 32)
        self.conv3 = nn.Linear(32, 16)
        self.head = nn.Linear(16, num_actions)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size=5, stride=2):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
        #
        # conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = conv_w * conv_h * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x)  # self.head(x.view(x.size(0), -1))


# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])
#
#
# def get_cart_location(screen_width):
#     world_width = env.x_threshold * 2
#     scale = screen_width / world_width
#     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
#
#
# def get_screen():
#     # Returned screen requested by gym is 400x600x3, but is sometimes larger
#     # such as 800x1200x3. Transpose it into torch order (CHW).
#     screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#     # Cart is in the lower half, so strip off the top and bottom of the screen
#     _, screen_height, screen_width = screen.shape
#     screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
#     view_width = int(screen_width * 0.6)
#     cart_location = get_cart_location(screen_width)
#     if cart_location < view_width // 2:
#         slice_range = slice(view_width)
#     elif cart_location > (screen_width - view_width // 2):
#         slice_range = slice(-view_width, None)
#     else:
#         slice_range = slice(cart_location - view_width // 2,
#                             cart_location + view_width // 2)
#     # Strip off the edges, so that we have a square image centered on a cart
#     screen = screen[:, :, slice_range]
#     # Convert to float, rescale, convert to torch tensor
#     # (this doesn't require a copy)
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0).to(device)


# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 200
# TARGET_UPDATE = 2  # 10  right now doing every 1
n_actions = 6
state_size = 21
new_model = True

policy_net = DQN(state_size, n_actions).to(device)
if not new_model:
    policy_net.load_state_dict(torch.load('terran_net.pt'))
target_net = DQN(state_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


episodes_done = 0


def select_action(state):
    global episodes_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)  # policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# episode_durations = []

# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state)),
    #                               device=device, dtype=torch.bool)
    non_final_next_states = torch.cat(batch.state)
    # state_batch = torch.cat([s for s in batch.prev_state if s is not None])
    state_batch = torch.cat(batch.prev_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()


class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.base_top_left = None
        self.attack_xy = None
        self.actions = ("do_nothing", "harvest_minerals", "build_supply_depot", "build_barracks", "train_marine",
                        "attack")

    def reset(self):
        super(Agent, self).reset()
        self.base_top_left = None
        self.attack_xy = None

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


def save_model(filename):
    torch.save(target_net.state_dict(), filename)


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        # self.q_table = QLearningTable(self.actions, filename=filename)
        self.clear()
        self.results = []
        self.episode_durations = []

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
        can_afford_supply_depot = 1 if obs.observation.player.minerals >= 100 else 0
        can_afford_barracks = 1 if obs.observation.player.minerals >= 150 else 0
        can_afford_marine = 1 if obs.observation.player.minerals >= 100 else 0

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return [[len(command_centers),
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
                len(enemy_marines)]]

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        # prev state already saved
        # for t in count():

        # _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([obs.reward], device=device)

        # Observe new state
        state = self.get_state(obs)
        state = torch.Tensor(state).to(device)

        # Select and perform an action
        action = select_action(state)

        # last_screen = current_screen
        # current_screen = get_screen()
        # if not done:
        #     next_state = current_screen - last_screen
        # else:
        #     next_state = None

        # Store the transition in memory
        # memory.push(state, action, next_state, reward)
        if self.previous_state is not None:
            memory.push(self.previous_state, self.previous_action, state, reward)

        # Move to the next state
        # state = next_state
        self.previous_state = state
        self.previous_action = action

        # Perform one step of the optimization (on the policy network)
        # optimize_model()
        if obs.last():
            global episodes_done
            episodes_done += 1
            self.episode_durations.append(self.steps + 1)
            # plot_durations()
            # break
            # Update the target network, copying all weights and biases in DQN
            # if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            self.results.append(obs.reward)
        else:
            optimize_model()

        # action = self.q_table.choose_action(state)
        # if self.previous_action is not None:
        #     self.q_table.learn(self.previous_state, self.previous_action, obs.reward,
        #                        'terminal' if obs.last() else state)
        # self.previous_state = state
        # self.previous_action = action
        return getattr(self, self.actions[action.item()])(obs)

    # def save(self, filename):
    #     self.q_table.save(filename)


def accuracy_every(my_list, window):
    chunks = [my_list[x:x + window] for x in range(0, len(my_list), window)]
    accuracies = [c.count(1)/len(c) for c in chunks]
    return accuracies


def plot(a, el):
    new_el = []
    for i in range(len(el)):
        if i == 0:
            new_el.append(el[i])
        else:
            new_el.append(el[i] - el[i-1])

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(a)
    ax2.plot(new_el)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Rolling Win Percentage")
    ax1.set_title("Change in Win Percentage Over Time")
    ax2.set_xlabel("Episode Number")
    ax2.set_ylabel("Episode Length (# of steps)")
    ax2.set_title("Change in Episode Length Over Time")
    plt.show()


def main(unused_argv):
    agent1 = SmartAgent()
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
            run_loop.run_loop([agent1, agent2], env, max_episodes=200)  # normally 1000 episodes
            save_model("terran_net.pt")
            
            print(agent1.results)
            accuracies = accuracy_every(agent1.results, 10)
            print(accuracies)

            plot(accuracies, agent1.episode_durations)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
