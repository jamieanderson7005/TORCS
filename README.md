# README

# AI Enhanced F1 Simulator With IBM Granite

### SHU-Undecided Team Members:
1. Jamie Anderson
2. Alexander Adeyemo
3. Jake Harrison-Bell
4. Joshua Odetoyinbo

## Project Brief
This project that we have been working on aims to transform The Open Race Car Simulator (TORCS) environment into an intelligent, self learning, immersive racing experience. By integrating IBM Granite into elements like the chatbot and live race commentary and reinforcement learning to the AI race bot, we as a team have developed an advanced system that includes a high performance race car that receives life like commentary and proper advice from the AI engineer.

## Our Core Features
**AI Race Bot Using Reinforcement Learning:**<br>
The main focus we had in this project was to improve the efficiency of the model through reinforcement learning so it can navigate the track through trial and error.
- **RL Framework:** Built using a reward based system that learns to make decisions based on interacting with its environment and improves cornering, braking and overall lap-time.
- **Training:** The model optimises the racing line and the speed it approaches corners and straights on the track by processing low level sesnor data to maximise its rewards from the RL.

**AI Race Engineer:**<br>
While the bot handles the driving of the model, IBM Granite is acting almost as the brains in the pit lane as it gives advice to the driver/model.
- **Real Time Interaction:** A chatbot interface that is able to interpret complex telemetry data and display them to the user in real time.
- **Strategic Advice:** Displays information and tips such as "Car off track. Immediate correction needed." and "Complementing on the sector 1 lap, you're maintaining a smooth --- RPM".

**Procedural Commentary**<br>
To make the simulation feel more realistic, we used IBM Granite's language generation to produce commentary for the user wo they are able to see how the model is performing and to provide real time entertainment.
- **Dynamic Narrative:** It converts raw telemetry events into life like commentary that allows the user to follow the race with ease, this also has a text to speech feature to improve the accessability of the simulation.
- **Contextual Awareness:** Granite is able to generate descriptions of the models behaviour providing a step by step of the RL agent's performance.

## Tech Stack

| Component | Technology | Role |
| :---| :--- | :--- |
| **Simulator** | TORCS | Racing Environment |
| **Race Bot** | Reinforcement Learning | Driving and Vehicle Control |
| **Intelligence** | IBM Granite | LLM for Race Engineer and Commentary |
| **Integration and Devlopment** | Python (VisualStudio Code), Github | Data Flow and Collaboration |

## Project Objectives
- To enhance an open-source Formula 1-style racing simulator using IBM Granite foundation models. This includes making an AI Race Engineer (Chatbot), telemetry analysis, procedural commentary and an AI race bot.
- Maintain project momentum and adaptability despite changes in team structure.

## How To Run The Project Sections

### Reinforcement Learning / Main Model:
To run the machine learning there are 3 modes: Optimise, Continuous and Drive. To run each mode firstly open wtorcs.exe. Once wtorcs.exe is running and you have selected the race with server1 as the driver:

Run either of these commands for the modes in the console:

python "directory of torcs_continuous" --optimise
python "directory of torcs_continuous" --continuous
python "directory of torcs_continuous" --drive

**Optimise:**
Optimise is a training loop that has an end which can be changed. It will run multiple laps of which is specified and once those have completed it will take the best and move onto generation 2 in which the best from generation 1 is used and improved upon and repeats in generations until an end term.

**Continuous:**
Continuous will run the exact same as optimise however it doesnt have an end condition and will run generations indefinetly.

**Drive:**
Drive will deploy the best trained model for demo or racing.

### Live Commentary:
Please run the following line in your terminal to ensure you see live commentary on torcs:<b>
python -m pip install -r requirements.txt

Please also ensure you install Ollama from this link:<b>
https://ollama.com/download

And install the model specifially from this link by running this line in your terminal:<b>
https://ollama.com/library/granite4:3b 

**Requirements to be installed:**
- torch
- ollama
- numpy
- matplotlib

**Download Ollama for MacOS:**<b>
https://ollama.com/download

**Granite 4 Download:**<b.>
https://ollama.com/library/granite4:3b

### Chatbot / Race Engineer:
**Dependencies:**
* TORCS
* Python
* Ollama, numpy

**Installing:**
* Download and install [Ollama](https://ollama.com/download/windows)
* Run in terminal
```
pip install ollama numpy
```

* Pull AI model
```
ollama pull granite4:3b
```

**Executing program:**
* Run main.py
```
python main.py
```
* To run in mock mode choose option 1; To run with AI car choose option 3