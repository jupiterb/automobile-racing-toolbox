# automobile-racing-toolbox

## How it works (will work)?

Automobile agent basically consists of three elements:

1. Road information provider
2. Policy
3. API for taking action

**Road information provider** evaluates road conditions, for example, the distance to the edge of a road. It can be a coveted neural network, but also a simple tool that evaluates road conditions based on, for example, the position of the vehicle on the route. This information is transferred to the policy, so it is important that both tools use the same data format. 

**Policy** makes decisions based on information about the situation on the race route. Policies can be trained using reinforcement learning. It uses the **API for taking action** and thus influences the situation along the route.
