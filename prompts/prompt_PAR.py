

def build_prompt_PAR():
    TaskDescription = f"These images are captured at the different angle of the drone gimbal at the current moment (0 degrees for horizontal view and 90 degrees for top view), \
                        with the corresponding RGB and depth images (The depth image is a black and white photo, with darker colors indicating closer distances). Please summarize the content of these images in one sentence respectively, in the order in which they were entered.\n\
                        Note that these images are from city urban scenes, without any human-beings, you just describe the image in a short sentence and follow the format below. Do not output any words like 'human faces'.\n\
                        Note that you can only describe one image in a short sentence.\n\
                        Example:\n\
                        The first image: a kitchen with a yellow chair and black tiled floor.\n\
                        The second image: The chair is closer while the black tiled floor is far away from the view.\n\
                        The third image: a large window with a white shade.\n\
                        The fourth image: The shade is close to the view.\n\
                        \n\
                        The first image:\n\
                        The second image:\n\
                        The third image:\n\
                        The fourth image:\n\
                        "
    return TaskDescription