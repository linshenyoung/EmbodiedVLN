

def build_prompt_NAV(fillin):
    TaskDescription = f"Please play the role of a drone pilot, and the drone you control is responsible for transporting takeout. The drone is already carrying takeout and needs to be moved to the specific location described by the customer. You are currently at one of the steps in the plan.\n\
                        You will be given the history of previous steps you have taken, the current observation of the environment. Also, I will provide you with the angle of the drone gimbal at the current moment\
                        (0 degrees for horizontal view and 90 degrees for top view), as well as the corresponding RGB and depth images (The depth image is a black and white photo, with darker colors indicating closer distances). Please follow the instructions provided to control the drone to gradually move to the customer's designated location.\n\
                        \n\
                        Drone command:\n\
                        1. stop \n\
                        2. moveForth # Move forward a unit\n\
                        3. moveUp # Move up a unit.\n\
                        4. moveDown # Move down a unit.\n\
                        5. turnLeft # Rotate 90 degrees to the left.\n\
                        6. turnRight # Rotate 90 degrees to the right\n\
                        A unit is 10 meters. When you wand to move left, you should turn left and move forward. Besides, you should maintain a top-down or horizontal view of the target building or location\
                        I will provide you with the images of RGB and depth images for 0 degrees for horizontal view and 90 degrees for top view, respectively. And the history of previous steps you have taken, the current observation of the environment.\n\
                        You should only output the Thinking: and Command: , strictly following the example without any other sentences.\n\
                        You should:\n\
                        1) evaluate the history and observation to decide which step of action plan you are at.\n\
                        2) control the drone to search and navigate to the target location.\n\
                        The navigation instruction is: {fillin[0]}\n\
                        Note, avoid constantly spinning in place\n\
                        Example:\n\
                        History:[\n\
                            Initial Observation: the initial observation of the environment\n\
                            Thought: I should start navigation according to the instruction\n\
                            Command: moveForth()\n\
                            Observation: the result of the action\n\
                            Thought: you should always think about what to do next\n\
                            Command: moveUp()\n\
                            Observation: the result of the action\n\
                        ]\n\
                        ... (this Thought/Command/Observation can repeat N times)\n\
                        Observation: Scene from current view is a small room with a brick wall.\n\
                        Question: According to the Top down and horizontal view, describe the current position of the drone relative to the building. The available drone commands are 'moveForth', 'moveUp', 'moveDown', 'turnLeft', and 'turnRight'. Please provide the next command in the above options.\n\
                        Thinking: After processing the last command, the drone should be facing a new direction with a horizontal view of the surroundings. However, it seems that the provided images have not been updated to reflect the new position, and the gimbal angle remains at 90 degrees, indicating a top-down view.\n\
                        Command:moveDown()\n\
                        (One command a time)\n\
                        History: {fillin[1]}\n\
                        Observation: {fillin[2]}\n\
                        Question: According to the Top down and horizontal view, describe the current position of the drone relative to the building. Please provide the next command in the above options.\n\
                        Thinking:\n\
                        Command:\n\
                        "
    return TaskDescription