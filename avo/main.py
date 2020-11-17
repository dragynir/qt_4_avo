from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QApplication, QWidget, QRadioButton, QButtonGroup, QGroupBox
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton, QFileDialog, QDialog, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QDoubleValidator, QIcon, QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import os
from functools import partial
import configparser

from script import run_avo_script
from view import ViewForms




class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon(os.path.join('res', 'frequency-64.png')))

        self.root = QVBoxLayout()

        self.path_dict = dict()

        self.SETTINGS_FILE = 'settings.ini'

        self.__setup_ui()

        self.__parse_ini()

        self.__check_and_create_folders()

        self.setLayout(self.root)
        
        self.show()


    def __check_and_create_folders(self):
        dirs = ["OUT_metrics",
                "OUT_pics", "OUT_pics/Impulse",
                "OUT_pics/QC", "OUT_pics/Seismic"]
        for dir_ in dirs:
            if (not os.path.exists(dir_)):
                os.mkdir(dir_)

    def __parse_ini(self):

        config = configparser.ConfigParser()
        config.read(self.SETTINGS_FILE)

        for name, edit_text in self.path_dict.items():
            edit_text.setText(config.get('Paths', name))
        
        
        self.int_time1.setText(config.get('Arguments', 'int_time1'))
        self.int_time2.setText(config.get('Arguments', 'int_time2'))
        self.win_len_smooth.setText(config.get('Arguments', 'win_len_smooth'))

        self.a1.setText(config.get('Arguments', 'a1'))
        self.b1.setText(config.get('Arguments', 'b1'))

        self.win.setText(config.get('Arguments', 'win'))
        self.time1.setText(config.get('Arguments', 'time1'))
        self.time2.setText(config.get('Arguments', 'time2'))

        self.threshold.setValue(int(config.get('Arguments', 'threshold')))

        self.offset_range.setText(config.get('Arguments', 'offset_range'))
        
        self.config = config

    
    
    
    def __setup_ui(self):

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)

        self.root.setSpacing(5)


        paths_group_box = QGroupBox('Input files:')
        # self.root.addWidget(ViewForms.create_section_label('Input files:', font), alignment=Qt.AlignCenter)
        paths_group_box.setLayout(self.__create_path_inputs())
        self.root.addWidget(paths_group_box)



        analysis_group_box = paths_group_box = QGroupBox('Interval time analysis:')
        analysis_group_box.setLayout(self.__create_analysis_group_params())
        self.root.addWidget(analysis_group_box)

        synth_group_box = paths_group_box = QGroupBox('Phase estimation parameters:')
        synth_group_box.setLayout(self.__create_synth_group_params())
        self.root.addWidget(synth_group_box)

        metrics_group_box = paths_group_box = QGroupBox('Metrics parameters:')
        metrics_group_box.setLayout(self.__create_metrics_group_params())
        self.root.addWidget(metrics_group_box)


        run_button = QPushButton('Run')

        run_button.clicked.connect(self.__run_script)

        self.root.addWidget(run_button)
    

    
    def __button_group_selected(self, group):
        for b in group.buttons():
            if b.isChecked():
                return b.text()
    

    def __write_config(self):
        with open(self.SETTINGS_FILE, 'w') as configfile:
            self.config.write(configfile)


    def __run_script(self):
        
        for name, edit_text in self.path_dict.items():
            self.config['Paths'][name] = edit_text.text()


        self.config['Arguments']['int_time1'] = self.int_time1.text()
        self.config['Arguments']['int_time2'] = self.int_time2.text()

        self.config['Arguments']['approx'] = self.__button_group_selected(self.approx)
        self.config['Arguments']['vsp'] = self.__button_group_selected(self.vsp)

        self.config['Arguments']['win_len_smooth'] = self.win_len_smooth.text()


        self.config['Arguments']['on'] = self.__button_group_selected(self.on)[:1]

        self.config['Arguments']['a1'] = self.a1.text()
        self.config['Arguments']['b1'] = self.b1.text()

        self.config['Arguments']['win'] = self.win.text()


        self.config['Arguments']['time1'] = self.time1.text()
        self.config['Arguments']['time2'] = self.time2.text()

        self.config['Arguments']['param'] = self.__button_group_selected(self.param)


        self.config['Arguments']['threshold'] = str(self.threshold.value())

        self.config['Arguments']['offset_range'] = self.offset_range.text()

        self.__write_config()

        run_avo_script()


    
    def __create_path_block(self, name, key, container, path_dict):
        hbox = QHBoxLayout()
        label = QLabel(name)
        text_input = QLineEdit()
        button = QPushButton()
        button.setIcon(QIcon(os.path.join('res', 'file_folder.png')))
        button.clicked.connect(partial(self.__on_choose_folder, key))

        hbox.addWidget(label, stretch=15)
        hbox.addWidget(text_input, stretch=75)
        hbox.addWidget(button)

        container.addLayout(hbox)
        path_dict[key] = text_input

    def __create_path_inputs(self):

        vbox = QVBoxLayout()

        self.__create_path_block('segy:', 'segy', vbox, self.path_dict)
        self.__create_path_block('las:', 'las', vbox, self.path_dict)
        self.__create_path_block('impulse:', 'impulse', vbox, self.path_dict)

        return vbox
    
    
    def __on_choose_folder(self, name):
        path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if path:
            self.path_dict[name].setText(path)

        
    
    def __create_analysis_group_params(self):

        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.int_time1, l = ViewForms.create_labeled_input('Time1 (ms): ', QIntValidator())
        vbox.addLayout(l)
        self.int_time2, l = ViewForms.create_labeled_input('Time2 (ms): ', QIntValidator())
        vbox.addLayout(l)


        self.approx, l1 = ViewForms.create_button_group('Ampl approx:', ['richards', 'zoeppritz'])
        self.vsp, l2 = ViewForms.create_button_group('Extremum:', ['on', 'off'])

        groups_layout = QVBoxLayout()
        groups_layout.addLayout(l1)
        groups_layout.addLayout(l2)
        vbox.addLayout(groups_layout)

        return vbox

    def __muting_changed(self, state):

        if state != Qt.Checked:
            self.a1.setText('')
            self.b1.setText('')
            self.a1.setReadOnly(True)
            self.b1.setReadOnly(True)
        else:
            self.a1.setText('0')
            self.b1.setText('0')
            self.a1.setReadOnly(False)
            self.b1.setReadOnly(False)

    
    def __create_synth_group_params(self):

        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.win_len_smooth, l = ViewForms.create_labeled_input('Smooth velocity (odd): ', QIntValidator())
        vbox.addLayout(l)
        

        self.on, l = ViewForms.create_button_group('Muting:', ['yes', 'no'])

        muting_cb = QCheckBox('Muting')
        muting_cb.stateChanged.connect(self.__muting_changed)

        self.a1, l1 = ViewForms.create_labeled_input('a1: ', QDoubleValidator())
        self.b1, l2 = ViewForms.create_labeled_input('b1: ', QDoubleValidator())
    
        
        params_layout = QHBoxLayout()
        params_layout.addLayout(l1)
        params_layout.addLayout(l2)
        params_layout.addWidget(muting_cb)

        vbox.addLayout(params_layout)

        return vbox


     
    def __create_metrics_group_params(self):

        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.win, l = ViewForms.create_labeled_input('Window length: ', QIntValidator())
        vbox.addLayout(l)

        self.time1, l = ViewForms.create_labeled_input('Time1 (ms): ', QIntValidator())
        vbox.addLayout(l)

        self.time2, l = ViewForms.create_labeled_input('Time2 (ms): ', QIntValidator())
        vbox.addLayout(l)
        
        self.param, l = ViewForms.create_button_group('Amplitude:', ['max', 'rms'])
        vbox.addLayout(l)


        self.threshold, l = ViewForms.create_labeled_slider('Threshold', 0, 100)
        vbox.addLayout(l)

        self.offset_range, l = ViewForms.create_labeled_input('Offset range: ', QIntValidator())
        vbox.addLayout(l)

        return vbox


    
app = QApplication([])

appStyle = open("appStyle", "r").read().replace('\n', '')
app.setStyleSheet(appStyle)


application = Window()
application.show()
sys.exit(app.exec())