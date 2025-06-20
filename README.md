# EmbodiedVLN
The repository of graduate dian competition : Embodied Intelligent Problem - LLM in VLN


# Structure
```
│  airsim_test.py                               A simple code to test whether your airsim env is correctly installed.
│  API.py
│  BaseAgent.py                                 This python file defines a base agent for EmbodiedCity Tasks, its main functions include define client , load data and parse files.
│  EmbodiedNavAgent.py                          It is the main agent we used in this project, it consists of multimodal inputs understanding, history storing and decision making.
│  embodied_vln.py                              The raw executable python files provided by organizer, we modified our code based on this code.
│  list.txt
│  MyEmbodiedCityVLN.py                         The main class of this task, taking control of the agent through getting input\parse files\get actions and thoughts... and so on.
│  utils.py                                     A lot of interesting and necessary tools for development.
│  
├─.vscode
│      launch.json
│      
├─embodiedcity
│      client.py
│      __init__.py
│      
├─imgs                                           images captured from the horizontal and top camera, with both RGB images and depth images.
│      1.jpg
│      2.jpg
│      3.jpg
│      4.jpg
│      
├─prompts
│  │  .DS_Store
│  │  prompt1.txt
│  │  prompt2.py
│  │  prompt_NAV.py                              prompt build function for navigation
│  │  prompt_PAR.py                              prompt build function for parsing
│  │  prompt_SUM.py                              prompt build function for summarize
│  │  prompt_template.py                         including four templates in our project, however, due to the disability of langchain, we finally choose these three functions above.
│  │  requirements.txt
│  │  utils.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          prompt2.cpython-310.pyc
│          prompt_NAV.cpython-310.pyc
│          prompt_PAR.cpython-310.pyc
│          prompt_SUM.cpython-310.pyc
│          prompt_template.cpython-310.pyc
│          __init__.cpython-310.pyc
│          
├─results
├─vln
│  │  agent.py
│  │  base_navigator.py
│  │  clip.py
│  │  coord_transformation.py
│  │  dataset.py
│  │  env.py
│  │  evaluate.py
│  │  graph_loader.py
│  │  landmarks.py
│  │  navigator.py
│  │  prompt_builder.py
│  │  utils.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          agent.cpython-310.pyc
│          base_navigator.cpython-310.pyc
│          coord_transformation.cpython-310.pyc
│          env.cpython-310.pyc
│          evaluate.cpython-310.pyc
│          graph_loader.cpython-310.pyc
│          prompt_builder.cpython-310.pyc
│          __init__.cpython-310.pyc
│          
└─__pycache__
        BaseAgent.cpython-310.pyc
        EmbodiedNavAgent.cpython-310.pyc
        utils.cpython-310.pyc
```
