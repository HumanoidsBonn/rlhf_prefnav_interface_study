from igibson.objects.visual_marker import VisualMarker


class VisualMarkerHRL(VisualMarker):

    def __init__(
        self,
        **kwargs,
    ):
        super(VisualMarkerHRL, self).__init__(**kwargs)
        self.hidden = False
        self.hide_pos = [0, 0, -1]
        self.last_pos = None

    def hide(self):
        if self.hidden:
            return
        self.last_pos = self.get_position()
        self.set_position(self.hide_pos)
        self.hidden = True

    def unhide(self):
        if not self.hidden:
            return
        if self.last_pos is not None:
            self.set_position(self.last_pos)
            self.last_pos = self.get_position()
        self.hidden = False

    def set_position(self, pos):
        if self.hidden:
            return
        else:
            return super().set_position(pos)

    def set_position_orientation(self, pos, orn):
        if self.hidden:
            return
        else:
            return super().set_position_orientation(pos, orn)
