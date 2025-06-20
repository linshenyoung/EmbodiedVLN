# EmbodiedVLN
The repository of graduate dian competition : Embodied Intelligent Problem - LLM in VLN

# Structure

--imgs
|-- n.jpg                    images captured from the horizontal and top camera, with both RGB images and depth images.

--prompts
|-- prompt_NAV.py             prompt build function for navigation
|-- prompt_PAR.py             prompt build function for parsing
|-- prompt_SUM.py             prompt build function for summarize
|-- prompt_template.py        including four templates in our project, however, due to the disability of langchain, we finally choose these three functions above.

---- BaseAgent.py             This python file defines a base agent for EmbodiedCity Tasks, its main functions include define client , load data and parse files.
---- EmbodiedNavAgent.py      It is the main agent we used in this project, it consists of multimodal inputs understanding, history storing and decision making.
---- MyEmbodiedCityVLN.py     The main class of this task, taking control of the agent through getting input\parse files\get actions and thoughts... and so on.
---- embodied_vln.py          The raw executable python files provided by organizer, we modified our code based on this code.
---- utils.py                 A lot of interesting and necessary tools for development.
---- airsim_test.py           A simple code to test whether your airsim env is correctly installed.
