from langchain.prompts.prompt import PromptTemplate

NAVIGATION_PROMPT = """Please play the role of a drone pilot, and the drone you control is responsible for transporting takeout. The drone is already carrying takeout and needs to be moved to the specific location described by the customer. You are currently at one of the steps in the plan.
You will be given the history of previous steps you have taken, the current observation of the environment. Also, I will provide you with the angle of the drone gimbal at the current moment 
(0 degrees for horizontal view and 90 degrees for top view), as well as the corresponding RGB and depth images (The depth image is a black and white photo, with darker colors indicating closer distances). Please follow the instructions provided to control the drone to gradually move to the customer's designated location.

Drone command:
1. stop 
2. moveForth # Move forward a unit
3. moveUp # Move up a unit.
4. moveDown # Move down a unit.
5. turnLeft # Rotate 90 degrees to the left.
6. turnRight # Rotate 90 degrees to the right
A unit is 10 meters. When you wand to move left, you should turn left and move forward. Besides, you should maintain a top-down or horizontal view of the target building or location
I will provide you with the images of RGB and depth images for 0 degrees for horizontal view and 90 degrees for top view, respectively. And the history of previous steps you have taken, the current observation of the environment.
You should:
1) evaluate the history and observation to decide which step of action plan you are at.
2) control the drone to search and navigate to the target location.
The navigation instruction is: {navi_desc}
Note, avoid constantly spinning in place
Example:
History:[
    Initial Observation: the initial observation of the environment
    Thought: I should start navigation according to the instruction
    Command: moveForth()
    Observation: the result of the action
    Thought: you should always think about what to do next
    Command: moveUp()
    Observation: the result of the action
]
... (this Thought/Command/Observation can repeat N times)
Observation: Scene from current view is a small room with a brick wall.
Question: According to the Top down and horizontal view, describe the current position of the drone relative to the building. The available drone commands are 'moveForth', 'moveUp', 'moveDown', 'turnLeft', and 'turnRight'. Please provide the next command in the above options.
Thinking: After processing the last command, the drone should be facing a new direction with a horizontal view of the surroundings. However, it seems that the provided images have not been updated to reflect the new position, and the gimbal angle remains at 90 degrees, indicating a top-down view.
Command:moveDown()
(One command a time)
History: {history}
Observation: {observation}
Question: According to the Top down and horizontal view, describe the current position of the drone relative to the building. Please provide the next command in the above options.
Thinking:
Command:
"""
   
HISTORY_PROMPT = """Please play the role of a drone pilot, and the drone you control is responsible for transporting takeout. The drone is already carrying takeout and needs to be moved to the specific location described by the customer. You are currently at one of the steps in the plan.

You have reached a new point after taking previous action. You will be given the navigation history, the current observation of the environment, and the previous action you taken.

You should:
1) evaluate the new observation and history.
2) update the history with the previous action and the new observation.

History: {history}
Previous action: {previous_action}
Observation: {observation}   
Update history with the new observation:"""

PARSE_IMAGE_PROMPT = """These images are captured at the different angle of the drone gimbal at the current moment (0 degrees for horizontal view and 90 degrees for top view), 
with the corresponding RGB and depth images (The depth image is a black and white photo, with darker colors indicating closer distances). Please summarize the content of these images in one sentence respectively, in the order in which they were entered.
Note that these images are from city urban scenes, without any human-beings, you just describe the image in a short sentence and follow the format below. Do not output any words like "human faces".
Note that you can only describe one image in a short sentence.
Example:
The first image: a kitchen with a yellow chair and black tiled floor.
The second image: The chair is closer while the black tiled floor is far away from the view.
The third image: a large window with a white shade.
The fourth image: The shade is close to the view.

The first image:
The second image:
The third image:
The fourth image:
"""


# PARSE_IMAGE_PROMPT = """You are an AI assistant helping a drone agent understand its visual scene.

# You are given a single image captured by the drone's gimbal camera. The image is either an RGB photo or a depth map (black-and-white image where darker = closer).

# Your task is to summarize the visual content of the image in only one short and concise sentence.

# Guidance:
# - If the image is an RGB image, describe visible objects, scene type, and layout.
# - If the image is a Depth image, describe spatial depth (e.g., "the foreground object is closer", "the background is far").
# - Only describe the image in one short and concise sentence.
# - Be concise, specific, and use only one sentence.

# Now begin:
# """


SUMMARIZE_OBS_DESC = """Here is a single scene view from Horizontal(RGB image), Horizontal(Depth image), Top(RGB image), Top(Depth image):
Horizontal_RGB: {Horizontal_RGB}
Horizontal_Depth: {Horizontal_Depth}
Top_RGB: {Top_RGB}
Top_Depth: {Top_Depth}
Summarize the scene in one sentence:
"""