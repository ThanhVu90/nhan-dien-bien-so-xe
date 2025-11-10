"""
Views Package - MVC Architecture
Chứa các view để hiển thị và tương tác với người dùng
"""

from .console_view import ConsoleView
from .gui_view import GUIView

__all__ = [
    'ConsoleView',
    'GUIView'
]
