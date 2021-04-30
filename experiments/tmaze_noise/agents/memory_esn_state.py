import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import agent
# from replay_buffer_episodic import ReplayMemory, Transition
from agents.replay.replay_buffer import ReplayMemory, Transition

criterion = torch.nn.SmoothL1Loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# recurrent trace
class SimpleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, beta):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tanh = nn.ReLU()
        # self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        # # nn.init.sparse_(self.i2h.weight, 0.5)
        # torch.nn.init.xavier_uniform_(self.i2h.weight)
        # self.i2h.weight.requires_grad_(False)

        # beta = 0.05
        # beta = 0.2
        self.beta = beta
        self.num_factor = 10

        # 1)
        # self.w = torch.empty(1, self.num_factor, hidden_size)
        # self.w.requires_grad_(False)
        # decay_factors = np.random.uniform(0.8, .99, size=(10))
        # with torch.no_grad():
        #     for i, d in enumerate(decay_factors):
        #         self.w[0][i] = d
        # 2)
        self.w = torch.empty(1, self.num_factor, hidden_size).to(device)
        self.w.requires_grad_(False)
        decay_factors = 1 - np.clip(np.random.exponential(self.beta, self.num_factor), a_min=0, a_max=1-sys.float_info.epsilon)
        with torch.no_grad():
            for i, d in enumerate(decay_factors):
                self.w[0][i] = d
        # 3)
        # self.w = torch.tensor(1 - np.clip(np.random.exponential(beta, (self.num_factor, hidden_size)), a_min=0, a_max=1)).float().to(device)

        # self.pool_w = nn.Conv1d(self.num_factor, 2, 1)
        # self.pool_w = nn.Conv1d(self.num_factor, 1, 3, padding=1)
        self.pool_w = nn.Conv1d(self.num_factor, 1, 1)

        # self.i2o = nn.Linear(input_size+hidden_size, output_size)
        # self.i2o = nn.Linear(input_size+hidden_size*2, (input_size+hidden_size)//2)
        self.i2o = nn.Linear(input_size+hidden_size, (input_size+hidden_size)//2)
        self.o = nn.Linear((input_size+hidden_size)//2, output_size)
        # torch.nn.init.xavier_uniform_(self.i2o.weight)
        # self.actions = nn.Parameter(torch.normal(0, .01, (output_size, hidden_size)))

    def reset_memory(self):
        self.avg_feature = torch.zeros(1, self.input_size+self.hidden_size).to(device)
        self.steps = 1

    def forward(self, inp, hidden):
        output = []
        hiddens = []
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]

            # 1)
            # with torch.no_grad():
            #     hidden_w = self.w(hidden)
            #     hidden_w = hidden_w.reshape(-1, self.hidden_size, 10).mean(-1)
            #     hidden_in = self.i2h(x)
            #     hidden = self.tanh(hidden_in+hidden_w)
            #  combined = torch.cat((x, hidden), 1)

            # 2)
            # with torch.no_grad():
            #     hidden_w = self.w(hidden)
            # hidden_w = hidden_w.reshape(-1, 10, self.hidden_size)
            # hidden_w = self.pool_w(hidden_w).squeeze(dim=1)
            # hidden_in = self.i2h(x)
            # # hidden_in = x
            # with torch.no_grad():
            #     hidden = self.tanh(0.2*hidden_in+0.8*hidden_w)
            # combined = torch.cat((x, hidden), 1)

            # 3)
            # hidden_i = self.pool_w(hidden).squeeze(dim=1)
            # combined = torch.cat((x, hidden_i), 1)
            # pred = self.o(self.tanh(self.i2o(combined)))

            # with torch.no_grad():
            #     hidden_w = self.w * hidden
            #     # a)
            #     hidden_in = self.i2h(x)
            #     # b)
            #     # hidden_in = x
            #     hidden_in = hidden_in.repeat(1, 10, 1)
            #     hidden = hidden_in+hidden_w

            # output.append(pred)

            # 4)
            # hidden_i = self.pool_w(torch.sigmoid(hidden)).squeeze(dim=1)
            # hidden_i = self.pool_w(torch.log(hidden+1)).squeeze(dim=1)
            hidden_i = self.pool_w(hidden).squeeze(dim=1)
            # hidden_i = self.pool_w(hidden).flatten(start_dim=1)
            # x = self.i2h(x)
            combined = torch.cat((x, hidden_i), 1)
            self.avg_feature = (self.avg_feature + combined) / self.steps
            pred = self.o(self.tanh(self.i2o(self.avg_feature)))
            self.avg_feature *= self.steps
            self.steps += 1

            with torch.no_grad():
                hidden_w = self.w * hidden
                hidden_in = x
                hidden_in = hidden_in.repeat(1, self.num_factor, 1)
                hidden = hidden_w + hidden_in

            output.append(pred)

            hiddens.append(hidden)

        return output[-1], hiddens[-1]

    def batch(self, inp, hidden, discount_batch, action_batch):
        output = []
        hiddens = []
        avg_feature = torch.zeros(1, self.input_size+self.hidden_size).to(device)
        steps = 1
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]

            # 1)
            # with torch.no_grad():
            #     hidden_w = self.w(hidden)
            #     hidden_w = hidden_w.reshape(-1, self.hidden_size, 10).mean(-1)
            #     hidden_in = self.i2h(x)
            #     hidden = self.tanh(hidden_in+hidden_w)
            #  combined = torch.cat((x, hidden), 1)

            # 2)
            # with torch.no_grad():
            #     hidden_w = self.w(hidden)
            # hidden_w = hidden_w.reshape(-1, 10, self.hidden_size)
            # hidden_w = self.pool_w(hidden_w).squeeze(dim=1)
            # hidden_in = self.i2h(x)
            # # hidden_in = x
            # with torch.no_grad():
            #     hidden = self.tanh(0.2*hidden_in+0.8*hidden_w)
            # combined = torch.cat((x, hidden), 1)

            # 3)
            # hidden_i = self.pool_w(hidden).squeeze(dim=1)
            # combined = torch.cat((x, hidden_i), 1)
            # pred = self.o(self.tanh(self.i2o(combined)))

            # with torch.no_grad():
            #     hidden_w = self.w * hidden
            #     hidden_in = self.i2h(x)
            #     # hidden_in = x
            #     hidden_in = hidden_in.repeat(1, 10, 1)
            #     hidden = hidden_in+hidden_w

            # output.append(pred)

            # 4)
            # hidden_i = self.pool_w(torch.sigmoid(hidden)).squeeze(dim=1)
            # hidden_i = self.pool_w(torch.log(hidden+1)).squeeze(dim=1)
            hidden_i = self.pool_w(hidden).squeeze(dim=1)
            # hidden_i = self.pool_w(hidden).flatten(start_dim=1)
            # x = self.i2h(x)
            combined = torch.cat((x, hidden_i), 1)
            combined = (avg_feature + combined) / steps
            pred = self.o(self.tanh(self.i2o(combined)))
            avg_feature = combined.detach() * steps
            steps += 1

            with torch.no_grad():
                hidden_w = self.w * hidden
                hidden_in = x
                hidden_in = hidden_in.repeat(1, self.num_factor, 1)
                hidden = hidden_w + hidden_in

            output.append(pred)

            hiddens.append(hidden.detach())

            if discount_batch[i].item() == 0:
                hidden = self.initHidden()
        return torch.cat(output), hiddens

    def initHidden(self):
        return torch.zeros(1, self.num_factor, self.hidden_size).to(device)


class RNNAgent(agent.BaseAgent):
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        torch.manual_seed(agent_init_info["seed"])
        self.T = agent_init_info.get("T",10)
        self.hidden_size = agent_init_info.get("hidden_size", self.num_states+1)

        self.beta = agent_init_info["beta"]
        self.rnn = SimpleRNN(self.num_states+1, self.hidden_size, self.num_actions, self.beta).to(device)
        self.target_rnn = SimpleRNN(self.num_states+1, self.hidden_size, self.num_actions, self.beta).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(10000)
        self.tau = .5
        self.flag = False
        self.train_steps = 0

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        self.rnn.reset_memory()
        # Choose action using epsilon greedy.
        self.is_door = None
        self.feature = None
        self.hidden = self.rnn.initHidden()

        self.prev_hidden = self.hidden

        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps = 0

        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        self.buffer.push(self.prev_state, self.prev_action, reward, self.prev_hidden.detach(), self.discount)

        self.prev_hidden = self.hidden

        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        if len(self.buffer) > self.T+1:# and self.steps % 5 == 0:# and self.epsilon == .1:
            self.batch_train()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        if append_buffer:
            self.buffer.push(self.prev_state, self.prev_action, reward, self.prev_hidden.detach(), 0)
            self.flag = True

        if len(self.buffer) > self.T+1:
            self.batch_train()

    def batch_train(self):
        self.train_steps += 1
        self.rnn.train()
        transitions = self.buffer.sample_successive(self.T+1)
        batch = Transition(*zip(*transitions))
        next_discount_batch = torch.FloatTensor(batch.discount[1:]).to(device)
        state_batch = torch.cat(batch.state[:-1])
        next_state_batch = torch.cat(batch.state[1:])

        action_batch = torch.LongTensor(batch.action[:-1]).view(-1, 1).to(device)
        next_action_batch = torch.LongTensor(batch.action[1:]).view(-1, 1).to(device)

        reward_batch = torch.FloatTensor(batch.reward[:-1]).to(device)
        hidden_batch = batch.hidden[0]
        next_hidden_batch = batch.hidden[1] # or after rnn next_hidden_batch[0]

        discount_batch = torch.FloatTensor(batch.discount[:-1]).to(device)

        current_q, _ = self.rnn.batch(state_batch, hidden_batch, discount_batch, action_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            # new_q, _ = self.target_rnn.batch(next_state_batch, next_hidden_batch, next_discount_batch, next_action_batch)
            new_q, _ = self.rnn.batch(next_state_batch, next_hidden_batch, next_discount_batch, next_action_batch)
        max_q = new_q.max(1)[0]
        # max_q = new_q.gather(1, next_action_batch).squeeze_()
        target = reward_batch
        target += discount_batch * max_q

        target = target.view(-1, 1)
        loss = criterion(q_learning_action_values, target)
        # loss = criterion(q_learning_action_values[-1], target[-1])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.rnn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.train_steps % 100 == 0:
            self.update()

    # def update(self):
    #     # target network update
    #     for target_param, param in zip(self.target_rnn.parameters(), self.rnn.parameters()):
    #         target_param.data.copy_(
    #             self.tau * param + (1 - self.tau) * target_param)

    def update(self):
        self.target_rnn.load_state_dict(self.rnn.state_dict())

    def update_target(self):
        self.target_rnn.load_state_dict(self.rnn.state_dict())
