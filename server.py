import asyncio
import socketio
import torch
import numpy as np
from environment import RunnerCatcherEnv

sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio)

# Load models
runner_network = torch.load('runner_model.pth')
catcher_network = torch.load('catcher_model.pth')

@sio.event
async def connect(sid, environ):
    print(f'Connected: {sid}')

@sio.event
async def disconnect(sid):
    print(f'Disconnected: {sid}')

@sio.event
async def start_simulation(sid, grid):
    await run_simulation(sid, grid)

async def run_simulation(sid, grid):
    """Run simulation and emit grid states"""
    try:
        # Initialize environment
        env = RunnerCatcherEnv()
        env.grid = np.array(grid)
        
        # Find initial positions
        runner_pos = np.where(env.grid == 3)
        catcher_pos = np.where(env.grid == 2)
        
        if len(runner_pos[0]) == 0 or len(catcher_pos[0]) == 0:
            await sio.emit('error', {'message': "Missing runner or catcher on grid"}, room=sid)
            return
            
        env.runner_pos = (runner_pos[0][0], runner_pos[1][0])
        env.catcher_pos = (catcher_pos[0][0], catcher_pos[1][0])
        
        # Send initial grid
        await sio.emit('grid_update', {
            'grid': env.grid.tolist(),
            'done': False
        }, room=sid)
        
        # Run simulation
        done = False
        while not done:
            await asyncio.sleep(0.5)  # Delay for visualization
            
            # Get state tensor
            state_tensor = torch.FloatTensor(env.grid.flatten()).unsqueeze(0)
            
            # Get actions from models
            with torch.no_grad():
                runner_probs = runner_network(state_tensor)
                catcher_probs = catcher_network(state_tensor)
                runner_action = runner_probs.argmax().item()
                catcher_action = catcher_probs.argmax().item()
            
            # Take step and get next state
            next_state, rewards, done = env.step(runner_action, catcher_action)
            
            # Send updated grid
            await sio.emit('grid_update', {
                'grid': next_state.tolist(),
                'done': done,
                'runner_action': runner_action,
                'catcher_action': catcher_action
            }, room=sid)
            
    except Exception as e:
        print(f"Error in simulation: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
