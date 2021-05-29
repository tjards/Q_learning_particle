# Embracing the Value of Hallucinations: Neural Dyna Q-Learning for Easy Control Problems**(Status)**: *This is still just a research proposal. Q-Learning works well for tuning parameters and the neural network works well(ish) for modelling dynamics. Need to integrate into a Neural Dyna Q-Learning framework.*As descibed in Ref. 1, dyna-type reinforcement learning (RL) strategies greatly improve sample efficiency by learning from "simulated" experience. When an exact first-principles model is not available, these simulations are often implemented using data-driven models. In such cases, even small modelling errors can result in poor learning outcomes. These misleading simulated experiences can be considered "hallucinations" and, for certain applications, can be a big problem.In this project, we propose that by appropriately structuring the learning process, we can avoid the pitfalls of such hallucinations in automatic control problems. By constraining our RL problem within the well-established principles of proportion-integral-derivative (PID) control, we show that our proposed approach is sufficiently robust to absorb modelling errors in a Dyna Q-Learning framework. **In short, we reduce the complexity of the RL space to just filtering out really bad ideas, which is still feasible when the model isn't very good.** Since most PID controllers are poorly tuned anyway, we propose this is potentially useful.We implement a simple Neural Network to model a dynamic agent in (oh gosh) real-time and use this model as the basis for dyna planning. An off-policy Dyna Q-Learning approach (similar to the on-policy Learning Automata approach at Ref. 2) is then used to tune the gain parameters for the agent controller.## References1. Taher Jafferjee, Ehsan Imani, Erin J. Talvitie, Martha White, and Michael Bowling. [Hallucinating Value: A Pitfall of Dyna-style Planning with Imperfect Environment Models](https://arxiv.org/pdf/2006.04363.pdf)2. Peter T. Jardine, Sidney N. Givigi, and Shahram Yousefi. [Leveraging Data Engineering to Improve Unmanned Aerial Vehicle Control Design](https://ieeexplore.ieee.org/document/9130726)## CitingThe code is opensource but, if you reference this work in your own reserach, please cite me. I have provided an example bibtex citation below:`@techreport{Jardine-2021,  title={Embracing the Value of Hallucinations: Neural Dyna Q-Learning for Easy Control Problems},  author={Jardine, P.T.},  year={2021},  institution={Royal Military College of Canada, Kingston, Ontario},  type={Research Proposal},}`## Initial results: Q-Learning Here is an animation of the agent learning:<p float="center">  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/animation_05.gif" width="80%" /></p>Note that the cost decreases with time. The noise is generated by the random motion of the target. In practice, what is important is that the variance of this noise decreases with time (because the agent is getting better at tracking the target as it moves).<p float="center">  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/cost_05.png" width="80%" /></p>As we reduce the exploration rate, the rewards grow with time (because the agent is exploiting the good parameters more often):<p float="center">  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/rewards_05.png" width="80%" /></p>## Initial results: Neural NetworkHere is the neural network getting better at predicting the agent dynamics (i.e. error goes down):<p float="center">  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/batch1.png" width="30%" />  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/batch2.png" width="30%" />    <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/batch3.png" width="30%" /></p>Here is a comparison of the neural network tracking ("ghost") versus the actual agent ("states") given the same inputs. State updates are provided between successive RL trials (i.e. ever 2 seconds).<p float="center">  <img src="https://github.com/tjards/Q_learning_particle/blob/master/Figs/posError.png" width="80%" /></p>## Next steps- Integrate the Q-Learning with the Neural Network (i.e. Neural Dyna Q-Learning). - More detailed formulations, documentation, ... etc coming soon.