import sdl2
import sdl2.sdlimage
import sdl2.ext
import sdl2.sdlttf as ttf

class Picture():
    def __init__(self, renderer, path=None, x:int=0, y:int=0, font=None, text:str=None):

        if path:
            self.renderer = renderer
            path_encoded = path.encode('utf-8')
            img_surface = sdl2.sdlimage.IMG_Load(path_encoded)
            #self.texture = sdl2.ext.renderer.Texture(renderer, img_surface)
            self.texture = sdl2.SDL_CreateTextureFromSurface(renderer, img_surface)
            self.src_rect = sdl2.SDL_Rect(0,0,img_surface.contents.w, img_surface.contents.h)
            self.dest_rect = sdl2.SDL_Rect(x, y, img_surface.contents.w, img_surface.contents.h)
        elif font and text:
            self.renderer = renderer
            try:
                surface = ttf.TTF_RenderText_Blended(font, text.encode('utf-8'), sdl2.SDL_Color(0,0,0))
                #self.texture = sdl2.ext.renderer.Texture(renderer, surface)
                self.texture = sdl2.SDL_CreateTextureFromSurface(renderer, surface)
                self.src_rect = sdl2.SDL_Rect(0,0,surface.contents.w, surface.contents.h)
                self.dest_rect = sdl2.SDL_Rect(x, y, surface.contents.w, surface.contents.h)
            except Exception as e:
                raise ValueError(f"Error: {e}")
        else:
            raise ValueError("Error creating picture: a valid path or font/text must be specified!")

    def render(self, x_scale_factor=1, y_scale_factor=1):
        rect = sdl2.SDL_Rect(int(self.dest_rect.x*x_scale_factor),
                             int(self.dest_rect.y*y_scale_factor),
                             int(self.dest_rect.w*x_scale_factor),
                             int(self.dest_rect.h*y_scale_factor) )
        #self.renderer.blit(self.texture, self.src_rect, rect)
        sdl2.SDL_RenderCopy(self.renderer, self.texture, self.src_rect, rect)
