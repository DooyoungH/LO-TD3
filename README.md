# LO-TD3

This page is the implementation of Lifelong Offline Twin Delayed Deep Deterministic policy gradient (LO-TD3) in a paper "Energy-Efficient Drone Delivery System through Reinforcement and Continual Learning".

<img width="2105" alt="Screen Shot 2022-05-19 at 9 21 34 PM" src="https://user-images.githubusercontent.com/40784671/169291988-29a78f81-66bc-4f47-bc5d-064a9e1333ca.png">

Drone delivery systems require a suitable RL algorithm since the training process of the drone through the conventional RL algorithm places too much burden on the hardware and when they consider changes like the climate or operation environments.
Therefore, we propose the Lifelong Offline Twin Delayed Deep Deterministic policy gradient (LO-TD3) for the drone delivery inspired by Offline RL and CL.

Because of Offline RL that is initial networks of LO-TD3, we higly recommand use above 64GB RAM system on the running computer if you train the overall model.  


LO-TD3 requires huge calculation resources. So, we also present the toy example using OpenAI GYM. 
It can be running by:
