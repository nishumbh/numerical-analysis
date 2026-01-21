from manim import FadeOut


def fadeOut(scene):
    scene.play(*[FadeOut(obj) for obj in scene.mobjects])

def fadeOutToNextSection(scene, wait_time=1, next_section_name="Next Scene"):
    scene.wait(wait_time)
    fadeOut(scene)
    scene.next_section(next_section_name)
    pass
