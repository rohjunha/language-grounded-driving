import numpy as np
import pygame


def destroy_actor(actor):
    if actor is not None and actor.is_alive:
        actor.destroy()


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def np_from_carla_image(image, reverse: bool = True):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    if reverse:
        array = array[:, :, ::-1]
    return array


def draw_image(surface, image, blend=False):
    array = np_from_carla_image(image)
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def show_game(
        display,
        font,
        image,
        clock,
        road_option = None,
        is_intersection = None,
        extra_str: str = ''):
    draw_image(display, image)
    strs = ['{:5.3f}'.format(clock.get_fps())]
    if road_option is not None:
        strs += [road_option.name.lower()]
    if is_intersection is not None:
        strs += [str(is_intersection)]
    if extra_str:
        strs += [extra_str]
    text_surface = font.render(', '.join(strs), True, (255, 255, 255))
    display.blit(text_surface, (8, 10))
    pygame.display.flip()
