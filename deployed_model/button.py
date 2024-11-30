import sdl2
import sdl2.ext
from picture import Picture

class Button():
    def __init__(self, renderer, path, x, y):
        self.sprite_sheet = Picture(renderer, path, x, y)
        self.is_clicked = False
        self.is_focus = False
        self.status = "normal"
        self.prev_status = "normal"
        self.button_height = int(self.sprite_sheet.dest_rect.h / 3)
        self.sprite_sheet.dest_rect.h = self.button_height
        self.sprite_sheet.src_rect.h = self.button_height

    def update_button(self, left_pressed, mouse_pos):
        self.prev_status = self.status
        if(sdl2.SDL_HasIntersection(mouse_pos, self.sprite_sheet.dest_rect)):
            if(left_pressed):
                self.status = "clicked"
                self.sprite_sheet.src_rect.y = self.button_height*2
                if(self.prev_status == "hovered"):
                    self.is_focus = True
            else:
                self.status = "hovered"
                self.sprite_sheet.src_rect.y = self.button_height
        else:
            self.status = "normal"
            self.sprite_sheet.src_rect.y = 0
            self.is_focus = False

    def clicked(self):
        if(self.status == "hovered" and self.prev_status == "clicked" and self.is_focus):
            self.is_focus = False
            return True
        return False
    
    def render(self, x_scale_factor=1, y_scale_factor=1):
        self.sprite_sheet.render(x_scale_factor, y_scale_factor)
        