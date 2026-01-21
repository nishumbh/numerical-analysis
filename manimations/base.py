
from manim import *

from utils import fadeOutToNextSection

class TestManim(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen
        self.play(Write(Text("Hello World")))

        fadeOutToNextSection(self)

        next_circle = Circle()
        next_circle.set_fill(YELLOW, opacity=0.4)
        self.play(Create(next_circle))
        self.play(Write(Text("PooPoo")))


