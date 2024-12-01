import sys
import pickle
import pandas as pd
import numpy as np
import random
import time
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

def get_test_data(xs, boxes : list[TextBox]):

    # Set default values to the mean of each column.

    hp = 69
    attack = 78
    defense = 73
    sp_attack = 72
    sp_defense = 70
    speed = 65
    base_egg_steps = xs['base_egg_steps'].mean()
    capture_rate = xs['capture_rate'].mean()

    # height = xs['height_m'].mean()
    # weight = xs['weight_kg'].mean()
    # xp = xs['experience_growth'].mean()
    # base_happiness = xs['base_happiness'].mean()
    
    # Obtain the values entered by the user.

    try:
        n=0
        for box in boxes:
            contents = box.get_text()
            if not contents and contents != 0:
                n += 1
                continue
            if(n==0):
                hp = int(contents)
            elif(n==1):
                attack = int(contents)
            elif(n==2):
                defense = int(contents)
            elif(n==3):
                sp_attack = int(contents)
            elif(n==4):
                sp_defense = int(contents)
            elif(n==5):
                speed = int(contents)
            elif(n==6):
                base_egg_steps = int(contents)
            elif(n==7):
                capture_rate = int(contents)
            # elif(n==8):
            #     xp = int(contents)
            # elif(n==9):
            #     weight = float(contents)
            # elif(n==10):
            #     height = float(contents)
            # elif(n==11):
            #     base_happiness = int(contents)
            n += 1

        
        
        # create a dataframe and get it in the expected format

        base_total = hp + attack + defense + sp_attack + sp_defense + speed
        #print(base_total, base_egg_steps, capture_rate)

        test_data = {
            'base_egg_steps' : [base_egg_steps],
            'capture_rate' : [capture_rate],
            'base_total' : [base_total],
            # 'base_happiness' : [base_happiness],
            # 'height_m': [height],
            # 'weight_kg' : [weight],
            # 'experience_growth' : [xp],
        }

        test_df = pd.DataFrame(test_data)
        return test_df
    except Exception as e:
        print(f"Input error: {e}")
        error_df = pd.DataFrame()
        return error_df

# Given a pandas dataframe in the a format used for training, predict if the pokemon is legendary

def predict_is_legendary(df, model) -> bool:
    result = model.predict(df)
    return (result == [1])

def run_app(model, xs):

    # initialize SDL and core objects
    sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING)
    sdl2.ext.init()
    ttf.TTF_Init()
    sdlimage.IMG_Init(sdlimage.IMG_INIT_PNG)
    window = sdl2.ext.window.Window(title="Legendary Pokemon Predictor", size=(640, 480), flags=sdl2.SDL_WINDOW_RESIZABLE)
    renderer = sdl2.SDL_CreateRenderer(window.window, -1, sdl2.SDL_RENDERER_ACCELERATED)

    icon = sdl2.ext.load_image("data/pikachu2.png")
    sdl2.SDL_SetWindowIcon(window.window, icon)

    # open fonts
    font_big = ttf.TTF_OpenFont("data/Ubuntu-M.ttf".encode('utf-8'), 32)
    font = ttf.TTF_OpenFont("data/Ubuntu-M.ttf".encode('utf-8'), 20)
    font_i = ttf.TTF_OpenFont("data/Ubuntu-LI.ttf".encode('utf-8'), 20)

    # create Picture/Button/TextBox instances

    title = Picture(renderer, font=font_big, text="YAML-G: Legendary Pokemon Predictor", x=20, y=10)
    bg = Picture(renderer, "data/bg.png", 0, 0)
    legendary = Picture(renderer, font=font_big, text="Your pokemon is LEGENDARY!", x=85, y=225)
    not_legendary = Picture(renderer, font=font_big, text="Your pokemon is NOT LEGENDARY!", x=60, y=225)
    
    legendary_error = Picture(renderer, font=font_big, text="Your pokemon is an ERROR!", x=85, y=225)
    legendary_error_2 = Picture(renderer, font=font_big, text="Check your input!", x=85, y=260)
    legendary_error_3 = Picture(renderer, "data/pikachu2.png", 500, 225)


    reset = Button(renderer, "data/reset.png", 100, 420)
    retry = Button(renderer, "data/retry.png", 250, 420)
    predict = Button(renderer, "data/predict.png", 275, 420)
    exit_b = Button(renderer, "data/exit.png", 450, 420)
    try_another = Button(renderer, "data/try_another.png", 100, 420)
    
    box1 = TextBox(renderer, font, "data/box.png", 50, 120, "Hitpoints", font2=font_i)
    box2 = TextBox(renderer, font, "data/box.png", 50, 180, "Attack", font2=font_i)
    box3 = TextBox(renderer, font, "data/box.png", 50, 240, "Defense", font2=font_i)
    box4 = TextBox(renderer, font, "data/box.png", 50, 300, "Special attack", font2=font_i)
    box5 = TextBox(renderer, font, "data/box.png", 350, 120, "Special defense", font2=font_i)
    box6 = TextBox(renderer, font, "data/box.png", 350, 180, "Speed", font2=font_i)
    box7 = TextBox(renderer, font, "data/box.png", 350, 240, "Base egg steps", font2=font_i)
    box8 = TextBox(renderer, font, "data/box.png", 350, 300, "Capture rate", font2=font_i)
    # box9 = TextBox(renderer, font, "data/box.png", 350, 180, "Experience growth", font2=font_i)
    # box10 = TextBox(renderer, font, "data/box.png", 350, 240, "Weight (kg)", font2=font_i)
    # box11 = TextBox(renderer, font, "data/box.png", 350, 300, "Height (m)", font2=font_i)
    # box12 = TextBox(renderer, font, "data/box.png", 350, 360, "Base Happiness", font2=font_i)
    
    
    
    # other variables with an extended scope

    keep_running = True
    left_pressed = False
    #boxes = [box1, box2, box3, box4, box5, box6, box7, box8, box9, box10, box11, box12]
    boxes = [box1, box2, box3, box4, box5, box6, box7, box8]
    mouse_rect = sdl2.SDL_Rect(0, 0, 1, 1)
    mouse_rect_logical = sdl2.SDL_Rect(0, 0, 1, 1)
    results_screen_on = False
    results_error = False
    is_legendary = False

    x_scale = 1
    y_scale = 1

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
                mouse_rect_logical.x = int(mouse_rect.x / x_scale)
                mouse_rect_logical.y = int(mouse_rect.y / y_scale)
            elif(event.type == sdl2.SDL_TEXTINPUT):
                input_text = event.text.text.decode("utf-8")
            elif(event.type == sdl2.SDL_KEYDOWN):
                if(event.key.keysym.sym == sdl2.SDLK_BACKSPACE):
                    backspace_pressed = True

        window_w, window_h = window.size
        x_scale = (window_w / 640)
        y_scale = (window_h / 480)

        # Using the logic and rendering methods for our GUI components

        bg.render(x_scale, y_scale)
        title.render(x_scale, y_scale)

        if not results_screen_on:

            for box in boxes:
                box.update(left_pressed, mouse_rect_logical, input_text, backspace_pressed)
                box.render(x_scale, y_scale)

            reset.update_button(left_pressed, mouse_rect_logical)
            predict.update_button(left_pressed, mouse_rect_logical)
            exit_b.update_button(left_pressed, mouse_rect_logical)
            reset.render(x_scale, y_scale)
            predict.render(x_scale, y_scale)
            exit_b.render(x_scale, y_scale)

            if(reset.clicked()):
                for box in boxes:
                    box.reset()

            if(predict.clicked()):
                results_screen_on = True
                test_df = get_test_data(xs, boxes)
                if not test_df.empty:
                    is_legendary = predict_is_legendary(test_df, model)
                else:
                    results_error = True

            if(exit_b.clicked()):
                keep_running = False
        else:
            
            exit_b.update_button(left_pressed, mouse_rect_logical)
            try_another.update_button(left_pressed, mouse_rect_logical)
            exit_b.render(x_scale, y_scale)
            try_another.render(x_scale, y_scale)
            if is_legendary and not results_error:
                legendary.render(x_scale, y_scale)
            elif not is_legendary and not results_error:
                not_legendary.render(x_scale, y_scale)
            else:
                legendary_error.render(x_scale, y_scale)
                legendary_error_2.render(x_scale, y_scale)
                legendary_error_3.render(x_scale, y_scale)

            if(try_another.clicked()):
                for box in boxes:
                    box.reset()
                results_screen_on = False
                results_error = False
            if(exit_b.clicked()):
                keep_running = False
            
        sdl2.SDL_RenderPresent(renderer) # Actually apply the changes to the screen

def main():

    model = unpickle("../models/model.pickle")
    xs = unpickle("../models/xs.pickle")
    run_app(model, xs)
     
if __name__ == '__main__':
    main()
