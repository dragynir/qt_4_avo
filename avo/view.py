from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QApplication, QWidget, QRadioButton, QButtonGroup
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton, QFileDialog, QDialog, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIcon, QIntValidator

class ViewForms():

    def __init__(self):
        pass


    @staticmethod
    def create_labeled_input(name, validator=None):

        hbox = QHBoxLayout()
        text_input = QLineEdit()
        label = QLabel(name)

        if validator:
            text_input.setValidator(validator)

        hbox.addWidget(label, stretch=30)
        hbox.addWidget(text_input, stretch=70)

        return text_input, hbox

    
    @staticmethod
    def create_labeled_slider(name, min_v, max_v):

        hbox = QHBoxLayout()

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_v)
        slider.setMaximum(max_v)

        label = QLabel(name)
        value_label = QLabel()
        value_label.setText(str(min_v))

        on_slider_changed = lambda x: value_label.setText(str(x))
        slider.valueChanged.connect(on_slider_changed)


        hbox.addWidget(label, stretch=15)
        hbox.addWidget(slider, stretch=75)
        hbox.addWidget(value_label, stretch=10)

        return slider, hbox
    
    @staticmethod
    def create_button_group(label_name, names):

        vbox = QVBoxLayout()
        label = QLabel(label_name)
        vbox.addWidget(label)

        group = QButtonGroup()
        for name in names:
            btn = QRadioButton(name)
            group.addButton(btn)
            vbox.addWidget(btn)
        
        group.buttons()[0].setChecked(True)

        return group, vbox