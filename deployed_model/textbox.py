import sdl2
import sdl2.ext
import sdl2.sdlttf
from picture import Picture



class TextBox():
    def __init__(self, renderer, font,  bg_path, x=0, y=0, default=" ", font2=None, ):
        self.bg = Picture(renderer, bg_path, x, y)
        self.font = font
        self.is_selected = False
        self.was_selected = False
        self.contents = " "
        self.default = " " + default
        if not font2: font2 = font
        self.font2 = font2
        self.text_picture = Picture(renderer, font=font2, text=self.default, x=x, y=y+12)

    def update(self, left_pressed, mouse_pos, input, backspace_pressed):
        self.was_selected = self.is_selected
        if(left_pressed and sdl2.SDL_HasIntersection(mouse_pos, self.bg.dest_rect)):
            self.is_selected = True
        elif left_pressed:
            self.is_selected = False

        if self.is_selected and (input or backspace_pressed):
            if backspace_pressed:
                if len(self.contents) > 1:
                    self.contents = self.contents[:-1]
            elif input and len(self.contents) < 17 and self.text_picture.dest_rect.w < self.bg.dest_rect.w - 14:
                self.contents = self.contents + input
            temp_contents = self.contents
            temp_contents = temp_contents + '|'
            self.text_picture = Picture(self.bg.renderer, font=self.font, text=temp_contents, x=self.bg.dest_rect.x, y=self.bg.dest_rect.y+12)
        elif self.was_selected and not self.is_selected:
            if(len(self.contents)>1):
                self.text_picture = Picture(self.bg.renderer, font=self.font, text=self.contents, x=self.bg.dest_rect.x, y=self.bg.dest_rect.y+12)
            else:
                self.text_picture = Picture(self.bg.renderer, font=self.font2, text=self.default, x=self.bg.dest_rect.x, y=self.bg.dest_rect.y+12)
        elif self.is_selected and not self.was_selected:
            temp_contents = self.contents + '|'
            self.text_picture = Picture(self.bg.renderer, font=self.font, text=temp_contents, x=self.bg.dest_rect.x, y=self.bg.dest_rect.y+12)

    def render(self, x_scale_factor=1, y_scale_factor=1):
        self.bg.render(x_scale_factor, y_scale_factor)
        self.text_picture.render(x_scale_factor, y_scale_factor)

    def get_text(self):
        return self.contents[1:]
        
