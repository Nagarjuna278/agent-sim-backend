import numpy as np
import random

# environment for a grid with obstacles where a agent tries to reach the goal 

#grid env with 8x8 with obstacles and walls and a goal, and the agent can move in 4 directions (up, down, left, right)
class Environment:
    def __init__(self, size=8, obstacles=None, agent_start=None, goal=None):
        self.size = size
        # self.obstacles0 = [(0,1),(2,1),(1,3),(3,2),(3,3)]
        self.obstacles0 = [(0,3),(1,3),(3,1),(2,5),(3,6),(4,6),(4,2),(4,4),(5,3),(6,2)]
        self.obstacles = obstacles if obstacles is not None else self.obstacles0
        self.agent_pos = None
        self.goal = None
        self.grid = None
        self.reset(obstacles=self.obstacles, start=agent_start,goal=goal)

    def reset(self,obstacles = None, start = None, goal = None):
        isreachable = False
        self.previous_catcher_actions = []
        self.previous_runner_actions = []

        while isreachable == False:
    
            self.grid = np.zeros((self.size,self.size))

            if obstacles is None:
                obstacles = []
                self.obstacles =[]
            
            if len(obstacles) == 0:
                self.agent_pos = start if start is not None else (random.randint(0,self.size-1),random.randint(0,self.size-1))
                self.goal = goal if goal is not None else (random.randint(0,self.size-1),random.randint(0,self.size-1))

                while len(self.obstacles) < self.size:
                    obs= (random.randint(0,self.size-1),random.randint(0,self.size-1))
                    if obs not in self.obstacles and obs != self.agent_pos and obs != self.goal:
                        self.obstacles.append(obs)
            
            else:
                self.obstacles = obstacles

            while True:
                pos = (random.randint(0,self.size-1),random.randint(0,self.size-1))
                if pos not in self.obstacles:
                    self.agent_pos = pos
                    break

            while True:
                pos = (random.randint(0,self.size-1),random.randint(0,self.size-1))
                if pos not in self.obstacles and pos != self.agent_pos:
                    self.goal = pos
                    break

            # print(self.obstacles,self.agent_pos,self.goal)
        
            for obs in self.obstacles:
                self.grid[obs] = -1
            self.grid[self.agent_pos] = 1
            self.grid[self.goal] = 2
            
            isreachable = False if self.distance_to_goal(self.agent_pos,self.goal)==-1 else True

        return self.grid.copy()


    def goalactions(self):
        actions =[]
        for pos in [(1,0),(0,1),(-1,0),(0,-1)]:
            dx = pos[0]
            dy = pos[1]
            x, y = self.goal[0] + dx, self.goal[1] + dy
            if 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles:
                actions.append((dx, dy))
        return actions

    def actions(self):
        actions = []
        for pos in [(1,0),(0,1),(-1,0),(0,-1)]:
            dx = pos[0]
            dy = pos[1]
            x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles:
                actions.append((dx, dy))
        return actions

    def step(self,action):
        dx, dy = action
        x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        if 0<=x<self.size and 0<=y<self.size:
            if (x,y) == self.goal:
                reward = 100
                done = True
            else:
                if self.isnexttome(start=(x,y),goal=self.goal):
                    reward = 20
                    done = True
                    return self.grid.copy(), reward, done
                old_distance = self.distance_to_goal(self.agent_pos,self.goal)
                new_distance = self.distance_to_goal((x,y),self.goal)
                if new_distance < old_distance:
                    reward = 0.5
                else:
                    reward = -0.5
                self.grid[self.agent_pos] = 0
                self.agent_pos = (x,y)
                self.grid[self.agent_pos] = 1
                done = False
        else:
            reward = -10
            done = True
        return self.grid.copy(), reward, done
    
    def runner_step(self,action):
        dx, dy = action
        x, y = self.goal[0] + dx, self.goal[1] + dy
        if 0<=x<self.size and 0<=y<self.size:
            if (x,y) == self.agent_pos:
                reward = 100
                done = True
            else:
                if self.isnexttome(start=(x,y),goal=self.agent_pos):
                    reward = 20
                    done = True
                    return self.grid.copy(), reward, done
                old_distance = self.distance_to_goal(self.agent_pos,self.goal)
                new_distance = self.distance_to_goal((x,y),self.agent_pos)
                if new_distance < old_distance:
                    reward = 0.5
                else:
                    reward = -0.5
                self.grid[self.goal] = 0
                self.goal = (x,y)
                self.grid[self.goal] = 2
                done = False
        else:
            reward = -10
            done = True
        return self.grid.copy(), -reward, done

    
    def distance_to_goal(self, start, goal):
        # Breadth-First Search (BFS) to find shortest path distance
        from collections import deque
        queue = deque([(start, 0)])
        visited = set()

        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == goal:
                return dist
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and 0 <= new_y < self.size and 
                    (new_x, new_y) not in self.obstacles and 
                    (new_x, new_y) not in visited):
                    queue.append(((new_x, new_y), dist + 1))

        return -1  # Goal not reachable
    
    def isnexttome(self,start,goal):
        for pos in [(1,0),(0,1),(-1,0),(0,-1)]:
            dx = pos[0]
            dy = pos[1]
            x, y = start[0] + dx, start[1] + dy
            if (x,y) == goal:
                return True
        return False