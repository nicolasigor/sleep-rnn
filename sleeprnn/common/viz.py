from IPython.core.display import display, HTML

from . import constants


DPI = 200
FONTSIZE_TITLE = 9
FONTSIZE_GENERAL = 8

PALETTE = {
    constants.RED: '#c62828',
    constants.GREY: '#455a64',
    constants.BLUE: '#0277bd',
    constants.GREEN: '#43a047',
    constants.DARK: '#1b2631'}


def notebook_full_width():
    display(HTML("<style>.container { width:100% !important; }</style>"))
