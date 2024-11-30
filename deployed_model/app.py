import sys
import pickle
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
import sdl2
import sdl2.ext
import sdl2.sdlimage as sdlimage
import sdl2.sdlttf as ttf
from picture import Picture
from button import Button
from textbox import TextBox

# Classes needed by the pickled object

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        return xs[self.columns]

class TransformData(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        result = xs.apply(self.func)
        return result
    
# Functions needed by the pickled object
    
def get_identity(x):
    return x

def get_sqrt(x):
    return np.where(x > 0, np.sqrt(x), 0)

# Deserialize and return serialized objects

def unpickle(model_file : str):
    try:
        with open(model_file, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
            return model
    except Exception as e:
        print(f"Error unpickling: {e}")
        sys.exit(1)

# Obtain new data to test the model with.
# The new data will be modified in the same way that the original data was for training.
# TODO eventually this data will be taken in from a GUI.

def get_test_data(xs):

    base_egg_steps = random.randint(10000, 20000)

    test_data = {
        'abilities' : [['Pressure', 'Pickpocket']],
        'base_egg_steps' : [base_egg_steps],
        'capture_rate' : [50],
        'hp' : [50],
        'speed' : [100],
    }

    # create a dataframe and get it in the expected format
    test_df = pd.DataFrame(test_data)
    mlb = MultiLabelBinarizer()
    abilities_data = pd.DataFrame(  mlb.fit_transform(test_df['abilities']), columns=mlb.classes_, index=test_df.index)
    test_df = test_df.drop(columns=['abilities'], axis=1)
    test_df = pd.concat([test_df, abilities_data], axis=1)
    test_df = test_df.reindex(columns=xs.columns, fill_value=0)
    return test_df

# Given a pandas dataframe in the a format used for training, predict if the pokemon is legendary

def predict_is_legendary(df, model):
    result = model.predict(df)
    if(result == [0]):
        print("Predicted not legendary!")
    else:
        print("Predicted legendary!")

def run_app(model, xs):

    # initialize SDL and core objects
    sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING)
    sdl2.ext.init()
    ttf.TTF_Init()
    sdlimage.IMG_Init(sdlimage.IMG_INIT_PNG)
    window = sdl2.ext.window.Window(title="Legendary Pokemon Predictor", size=(640, 480), flags=sdl2.SDL_WINDOW_RESIZABLE)
    renderer = sdl2.ext.renderer.Renderer(window)

    # open fonts
    font_big = ttf.TTF_OpenFont("data/Ubuntu-M.ttf".encode('utf-8'), 32)
    font = ttf.TTF_OpenFont("data/Ubuntu-M.ttf".encode('utf-8'), 20)
    font_i = ttf.TTF_OpenFont("data/Ubuntu-LI.ttf".encode('utf-8'), 20)

    # create Picture/Button/TextBox instances

    title = Picture(renderer, font=font_big, text="YAML-G: Legendary Pokemon Predictor", x=20, y=10)
    bg = Picture(renderer, "data/bg.png", 0, 0)

    reset = Button(renderer, "data/reset.png", 100, 420)
    retry = Button(renderer, "data/retry.png", 250, 420)
    predict = Button(renderer, "data/predict.png", 275, 420)
    exit_b = Button(renderer, "data/exit.png", 450, 420)
    try_another = Button(renderer, "data/try_another.png", 500, 420)
    
    box1 = TextBox(renderer, font, "data/box.png", 50, 60, "Ability 1", font2=font_i)
    box2 = TextBox(renderer, font, "data/box.png", 50, 120, "Ability 2 (optional)", font2=font_i)
    box3 = TextBox(renderer, font, "data/box.png", 50, 180, "Ability 3 (optional)", font2=font_i)
    box4 = TextBox(renderer, font, "data/box.png", 50, 240, "Ability 4 (optional)", font2=font_i)
    box5 = TextBox(renderer, font, "data/box.png", 50, 300, "Ability 5 (optional)", font2=font_i)
    box6 = TextBox(renderer, font, "data/box.png", 50, 360, "Ability 6 (optional)", font2=font_i)
    box7 = TextBox(renderer, font, "data/box.png", 350, 60, "Speed", font2=font_i)

    # other variables with an extended scope

    keep_running = True
    left_pressed = False
    boxes = [box1, box2, box3, box4, box5, box6, box7]
    mouse_rect = sdl2.SDL_Rect(0, 0, 1, 1)

    while(keep_running): # main loop
        
        input_text = ""
        backspace_pressed = False
        
        # obtaining user input/actions from the event queue

        events = sdl2.ext.common.get_events()
        for event in events:
            if(event.type == sdl2.SDL_QUIT):
                keep_running = False
            elif(event.type == sdl2.SDL_MOUSEBUTTONDOWN):
                if(event.button.button == sdl2.SDL_BUTTON_LEFT):
                    left_pressed = True
            elif(event.type == sdl2.SDL_MOUSEBUTTONUP):
                if(event.button.button == sdl2.SDL_BUTTON_LEFT):
                    left_pressed = False
            elif(event.type == sdl2.SDL_MOUSEMOTION):
                mouse_rect.x, mouse_rect.y = event.motion.x, event.motion.y
            elif(event.type == sdl2.SDL_TEXTINPUT):
                input_text = event.text.text.decode("utf-8")
            elif(event.type == sdl2.SDL_KEYDOWN):
                if(event.key.keysym.sym == sdl2.SDLK_BACKSPACE):
                    backspace_pressed = True

        # Using the logic and rendering methods for our GUI components

        bg.render()
        title.render()

        for box in boxes:
            box.update(left_pressed, mouse_rect, input_text, backspace_pressed)
            box.render()

        reset.update_button(left_pressed, mouse_rect)
        predict.update_button(left_pressed, mouse_rect)
        exit_b.update_button(left_pressed, mouse_rect)
        reset.render()
        predict.render()
        exit_b.render()

        renderer.present() # Actually apply the changes to the screen

def main():

    model = unpickle("../models/model.pickle")
    xs = unpickle("../models/xs.pickle")
    run_app(model, xs)

    # for _ in range(0, 5):
    #     print("Testing another pokemon...")
    #     test_df = get_test_data(xs)
    #     predict_is_legendary(test_df, model)
    #     time.sleep(2)


        
if __name__ == '__main__':
    main()
