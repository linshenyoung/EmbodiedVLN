

def build_prompt_SUM(obs_desc):
    TaskDescription = f"Here is a single scene view from Horizontal(RGB image), Horizontal(Depth image), Top(RGB image), Top(Depth image):\
                        Horizontal_RGB: {obs_desc[0]}\n\
                        Horizontal_Depth: {obs_desc[1]}\n\
                        Top_RGB: {obs_desc[2]}\n\
                        Top_Depth: {obs_desc[3]}\n\
                        Summarize the scene in one sentence:\n\
                        "
    return TaskDescription